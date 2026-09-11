"""Where the experiment chain is, from what it has written so far.

    python scripts/progress.py

Reads the chain log for the current step and start time, then counts the
things each step leaves behind: rows in a results file, entries in the
activation cache, judge calls in the cost log. Estimates are from the runs
already done on this machine and are rough.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "overnight.log"
RES = ROOT / "results"
COST = ROOT / "logs" / "llm_cost.jsonl"

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
