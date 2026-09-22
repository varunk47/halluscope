"""Where the running work is, from what it has written so far.

    python scripts/progress.py

Two parts. First the jobs running right now, the steering control and the
loop-seed queue, each read from the file it is filling rather than from
anything it claims. Then the older overnight chain, for history.

Nothing here talks to the jobs. A run that died leaves a checkpoint whose
timestamp stops moving, so a stale file is the signal, and the process table
is consulted only to confirm it.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "overnight.log"
RES = ROOT / "results"
COST = ROOT / "logs" / "llm_cost.jsonl"

STEER_FINAL = RES / "steer_qwen_last_minimal.json"
STEER_PARTIAL = RES / "steer_qwen_last_minimal.partial.json"
STEER_LOG = ROOT / "logs" / "steer_control.log"
LOOP_LOG = ROOT / "logs" / "loop_seeds.log"

# probe at five magnitudes, then three random directions at the four nonzero ones
STEER_SWEEPS = 17
LOOP_CONDITIONS = ("off", "gate", "always", "prompt")
LOOP_SEEDS = (0, 1, 2)


def _alive(pattern: str) -> int | None:
    """Python processes whose command line contains ``pattern``.

    Returns None if the process table cannot be read, so the caller can fall
    back to file timestamps rather than report a run dead on a failed lookup.
    """
    q = f"Name like '%python%' and CommandLine like '%{pattern}%'"
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-c",
             f"@(Get-CimInstance Win32_Process -Filter \"{q}\").Count"],
            capture_output=True, text=True, timeout=30,
        )
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _ago(path: Path) -> float:
    """Minutes since the file was last written."""
    return (time.time() - path.stat().st_mtime) / 60


def steer_status() -> None:
    print("steering control (random-direction)")
    if not STEER_PARTIAL.exists() and STEER_FINAL.exists():
        d = json.loads(STEER_FINAL.read_text(encoding="utf-8"))
        has = "random_controls" in d or any(
            r.get("direction", "probe") != "probe" for r in d.get("rows", [])
        )
        print(f"  finished, controls in the file: {has}")
        return
    if not STEER_PARTIAL.exists():
        print("  not started")
        return

    rows = json.loads(STEER_PARTIAL.read_text(encoding="utf-8")).get("rows", [])
    combos = {(r.get("direction", "probe"), float(r["alpha"])) for r in rows}
    done = len(combos)
    idle = _ago(STEER_PARTIAL)

    # Sweep durations come from the log, so the estimate uses this machine's
    # actual pace rather than a number I guessed once.
    times = [
        int(h) * 60 + int(m) + int(s) / 60
        for h, m, s in re.findall(r"\s(\d+):(\d\d):(\d\d)\s*$", STEER_LOG.read_text(
            encoding="utf-8", errors="replace"), re.M)
    ] if STEER_LOG.exists() else []
    per = sum(times[-5:]) / len(times[-5:]) if times else 16.0
    left = (STEER_SWEEPS - done) * per

    print(f"  {done} of {STEER_SWEEPS} sweeps, {len(rows)} rows")
    print(f"  {per:.0f} min per sweep lately, about {left / 60:.1f} h left")
    n = _alive("steer")
    if idle > 2 * per:
        print(f"  STALLED: checkpoint untouched for {idle:.0f} min")
    if n == 0:
        print("  DEAD: no process. Restart with --resume, it keeps what is done")
    elif n is None:
        print(f"  process table unreadable; checkpoint moved {idle:.0f} min ago")


def loop_status() -> None:
    print("\nloop seeds")
    have = {
        (c, s) for c in LOOP_CONDITIONS for s in LOOP_SEEDS
        if (RES / f"loop_qwen_{c}_s{s}.json").exists()
    }
    want = len(LOOP_CONDITIONS) * len(LOOP_SEEDS)
    print(f"  {len(have)} of {want} runs present")
    for s in LOOP_SEEDS:
        got = [c for c in LOOP_CONDITIONS if (c, s) in have]
        print(f"    seed {s}: {', '.join(got) if got else 'none'}")
    if not LOOP_LOG.exists():
        print("  queue not started")
        return
    tail = [ln for ln in LOOP_LOG.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    print(f"  queue: {tail[-1] if tail else 'no output yet'}")
    if STEER_PARTIAL.exists():
        print("  waiting for the card; the steering run still holds it")

STEPS = [
    ("behavior labels on the minimal build", 300),
    ("behavior probes on the minimal build", 40),
    ("capture Qwen3.5-2B on the minimal build", 60),
    ("gap probe on Qwen3.5-2B", 5),
    ("cross-model transfer qwen -> qwen2b", 5),
    ("activation steering on qwen", 90),
    ("semantic entropy on the train split, for the SEP probe", 200),
    ("semantic entropy and SEP on the test split", 200),
    ("report", 2),
]


def _rows(name: str) -> int:
    p = RES / name
    if not p.exists():
        return 0
    try:
        return len(json.loads(p.read_text(encoding="utf-8")).get("rows", []))
    except json.JSONDecodeError:
        return 0


def _rows_with(name: str, key: str) -> int:
    p = RES / name
    if not p.exists():
        return 0
    try:
        return sum(
            1 for r in json.loads(p.read_text(encoding="utf-8"))["rows"] if key in r["scores"]
        )
    except (json.JSONDecodeError, KeyError):
        return 0


def _calls(prefix: str, since: float) -> int:
    if not COST.exists():
        return 0
    n = 0
    for line in COST.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["ts"] >= since and r["tag"].startswith(prefix) and r["ok"]:
            n += 1
    return n


def _cache_count(model_slug: str) -> int:
    from halluscope.config import get_settings

    d = get_settings().resolved_cache_dir() / "activations" / model_slug
    return len(list(d.glob("*.safetensors"))) if d.exists() else 0


def _detail(step: str, started: float) -> str:
    n_items = 984  # approved items in the minimal build
    if step.startswith("behavior labels"):
        rows = _rows("behavior_qwen_minimal.json")
        if rows:
            return f"judging: {rows} of {n_items} answers labelled ({100 * rows // n_items}%)"
        return f"generating {n_items} answers; the judge phase starts when generation ends"
    if step.startswith("capture"):
        n = _cache_count("Qwen_Qwen3.5-2B")
        return f"{n} of about 1156 item-turns cached ({100 * n // 1156}%)"
    if step.startswith("activation steering"):
        n = _calls("steer:", started)
        return f"{n} of 160 steered replies judged ({100 * n // 160}%)"
    if step.startswith("semantic entropy on the train"):
        return f"{_rows_with('uq_qwen_train.json', 'semantic_entropy')} of 96 items done; {_calls('se:', started)} entailment calls"
    if step.startswith("semantic entropy and SEP"):
        return f"{_rows_with('uq_qwen_test.json', 'semantic_entropy')} of 96 items done; {_calls('se:', started)} entailment calls"
    return "short step"


def main() -> None:
    steer_status()
    loop_status()
    print("\novernight chain (history)")
    chain()


def chain() -> None:
    if not LOG.exists():
        print("no chain log; nothing running")
        return
    text = LOG.read_text(encoding="utf-8", errors="replace")
    marks = re.findall(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.+)$", text, re.M)
    if not marks:
        print("chain log exists but no step has started")
        return
    done_titles = [m[1] for m in marks]
    failed = [t for t in done_titles if t.startswith("FAILED")]
    finished = any(t.endswith("chain done") for t in done_titles)
    steps_only = [m for m in marks if not m[1].startswith("FAILED")]
    current_ts, current = steps_only[-1] if steps_only else marks[-1]
    started = datetime.strptime(current_ts, "%Y-%m-%d %H:%M:%S").timestamp()
    idx = next((i for i, (t, _) in enumerate(STEPS) if current.startswith(t)), None)

    total = sum(m for _, m in STEPS)
    print(f"chain started {marks[0][0]}")
    for i, (title, mins) in enumerate(STEPS):
        if finished or (idx is not None and i < idx):
            mark = "done"
        elif idx is not None and i == idx and failed:
            mark = "FAILED, see below"
        elif idx is not None and i == idx:
            mark = f"running {int((time.time() - started) // 60)} min, about {mins} expected"
        else:
            mark = f"queued, about {mins} min"
        print(f"  {i + 1}. {title:58} {mark}")
    if failed:
        print(f"\nSTOPPED: {failed[-1]}")
        return
    if finished:
        print("\nall steps done")
        return
    elapsed_before = sum(m for _, m in STEPS[:idx]) if idx is not None else 0
    in_step = min(STEPS[idx][1], (time.time() - started) / 60) if idx is not None else 0
    pct = 100 * (elapsed_before + in_step) / total
    left = max(0, total - elapsed_before - in_step)
    print(f"\ncurrent: {current}")
    print(f"  {_detail(current, started)}")
    print(f"\nabout {pct:.0f}% of the chain by expected time, roughly {left / 60:.1f} h left")


if __name__ == "__main__":
    main()
