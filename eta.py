"""Report how much of the seed-1 loop queue is left, and when it should finish.

The loop renders progress with rich's track(), which writes nothing when stdout
is redirected, so the queue log carries no progress at all. The one live signal
is logs/llm_cost.jsonl: every judge and simulated-user call appends a record
whose `tag` carries the item id, so counting distinct ids gives exact progress.

Estimates are built on awake time only. This machine has Modern Standby and
will suspend mid-run if the sleep timeout lets it, and wall clock keeps counting
through a suspend. Averaging over wall clock therefore reports a pace several
times worse than the hardware is actually managing, so any gap longer than
IDLE_GAP is treated as a stall and excluded from the pace.

Usage:  python eta.py
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"
COST_LOG = REPO / "logs" / "llm_cost.jsonl"

# The queue runs these in order, one at a time.
CONDITIONS = ("off", "gate", "always", "prompt")
SPLIT_N = 256

# A pause longer than this means the box was suspended, not working.
IDLE_GAP = dt.timedelta(minutes=10)


def result_path(cond: str) -> Path:
    return RESULTS / f"loop_qwen_{cond}_s1.json"


def load_calls() -> list[dict]:
    """Every API call, oldest first, with `ts` parsed into `t`."""
    if not COST_LOG.exists():
        return []
    out = []
    for line in COST_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            rec["t"] = dt.datetime.fromtimestamp(float(rec["ts"]))
            out.append(rec)
        except Exception:
            continue
    return sorted(out, key=lambda r: r["t"])


def relative_cost() -> dict[str, float]:
    """Per-condition cost factors, taken from the 48-task pilot.

    Conditions differ in how many turns they take: asking a question means a
    simulated-user reply and a second model turn. The pilot already measured
    that, so scale by its per-task API volume rather than guessing.
    """
    factors = {}
    for cond in CONDITIONS:
        pilot = RESULTS / f"loop_qwen_{cond}_s0.json"
        if not pilot.exists():
            continue
        rows = json.loads(pilot.read_text(encoding="utf-8"))["rows"]
        questions = sum(
            len(r["questions"]) if isinstance(r.get("questions"), list)
            else (1 if r.get("questions") else 0)
            for r in rows
        )
        factors[cond] = (2 * len(rows) + questions) / len(rows)
    base = factors.get("off")
    return {c: v / base for c, v in factors.items()} if base else {}


def main() -> None:
    now = dt.datetime.now()
    calls = load_calls()

    done = [c for c in CONDITIONS if result_path(c).exists()]
    todo = [c for c in CONDITIONS if c not in done]

    print(f"  now        {now:%a %d %b %H:%M:%S}")
    print(f"  complete   {len(done)} of {len(CONDITIONS)}   {', '.join(done) or 'none'}")

    if not todo:
        print("\n  ALL FOUR COMPLETE. The finalizer should have fired.")
        print("  Check FINALIZE_REVIEW.md in the session scratchpad.")
        return

    running = todo[0]

    # The running condition began when the previous one wrote its file.
    prev = done[-1] if done else None
    started = (dt.datetime.fromtimestamp(result_path(prev).stat().st_mtime)
               if prev else (calls[0]["t"] if calls else now))

    # Distinct item ids seen since then are the items this condition finished.
    seen: dict[str, dt.datetime] = {}
    for rec in calls:
        if rec["t"] < started:
            continue
        tag = rec.get("tag", "")
        if ":" in tag:
            seen.setdefault(tag.split(":", 1)[1], rec["t"])

    n_done = len(seen)
    stamps = [started] + sorted(seen.values()) + [now]

    awake = dt.timedelta()
    stalls: list[tuple[dt.datetime, dt.datetime]] = []
    for a, b in zip(stamps, stamps[1:], strict=False):
        if b - a > IDLE_GAP:
            stalls.append((a, b))
        else:
            awake += b - a

    lost = (now - started) - awake
    print(f"  running    {running}   {n_done} of {SPLIT_N}"
          f"   ({100 * n_done / SPLIT_N:.0f}%)")
    print(f"  elapsed    {(now - started).total_seconds() / 3600:.2f} h"
          f"   awake {awake.total_seconds() / 3600:.2f} h"
          f"   lost {lost.total_seconds() / 3600:.2f} h")

    if calls:
        quiet = (now - calls[-1]["t"]).total_seconds() / 60
        flag = "  <-- STALLED?" if quiet > 10 else ""
        print(f"  last call  {quiet:.1f} min ago{flag}")

    if n_done < 2:
        print("\n  Not enough items yet to estimate a pace.")
        return

    pace = awake.total_seconds() / n_done / 60  # minutes per item, awake only
    print(f"  pace       {pace:.2f} min/item (awake time only)")

    factors = relative_cost()
    here = factors.get(running, 1.0)

    print()
    total_h = 0.0
    for cond in todo:
        share = factors.get(cond, here) / here
        left = (SPLIT_N - n_done) if cond == running else SPLIT_N
        hours = left * pace * share / 60
        total_h += hours
        eta = now + dt.timedelta(hours=total_h)
        print(f"  {cond:<7} {left:>4} items   {hours:>5.1f} h   done ~{eta:%a %H:%M}")

    finish = now + dt.timedelta(hours=total_h)
    print(f"\n  REMAINING  {total_h:.1f} h   ->  finishes ~{finish:%a %d %b %H:%M}")

    if stalls:
        print(f"\n  {len(stalls)} stall(s) so far, most recent ended "
              f"{stalls[-1][1]:%H:%M}:")
        for a, b in stalls[-3:]:
            print(f"    {a:%H:%M} -> {b:%H:%M}   "
                  f"{(b - a).total_seconds() / 3600:.2f} h lost")
        print("  Stalls mean the machine suspended. Keep it awake:")
        print("    powercfg /change standby-timeout-ac 0")


if __name__ == "__main__":
    main()
