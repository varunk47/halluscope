"""Wait out a throttled provider, then start the verification run.

The NVIDIA NIM free tier answers a burst with 429 for far longer than any
sensible in-process backoff will wait, so the run cannot simply retry its way
through. This polls at a rate low enough not to keep the account throttled and
launches verify the moment a single call succeeds. The run itself is resumable,
so if the tier throttles again mid-way the next launch picks up where it
stopped rather than paying for every family twice.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
POLL_SECONDS = 300
MAX_HOURS = 12


def alive() -> bool:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import litellm

    litellm.drop_params = True
    litellm.suppress_debug_info = True
    try:
        litellm.completion(
            model="nvidia_nim/moonshotai/kimi-k3",
            messages=[{"role": "user", "content": "Reply with: ok"}],
            temperature=0.0,
            max_tokens=256,
        )
        return True
    except Exception as e:  # noqa: BLE001 - any failure means keep waiting
        print(f"  still throttled: {type(e).__name__}", flush=True)
        return False


def main(items: str, report: str) -> int:
    deadline = time.time() + MAX_HOURS * 3600
    while time.time() < deadline:
        print(f"[{time.strftime('%H:%M:%S')}] probing", flush=True)
        if alive():
            print("capacity is back, starting verify", flush=True)
            return subprocess.call(
                [
                    str(PY_EXE),
                    "-m",
                    "halluscope.cli",
                    "verify",
                    "--items",
                    items,
                    "--report",
                    report,
                    "--workers",
                    "1",
                    "--resume",
                ],
                cwd=str(ROOT),
            )
        time.sleep(POLL_SECONDS)
    print(f"gave up after {MAX_HOURS}h", flush=True)
    return 1


if __name__ == "__main__":
    items = sys.argv[1] if len(sys.argv) > 1 else "data/augmented/items_minimal.jsonl"
    report = sys.argv[2] if len(sys.argv) > 2 else "results/verify_report_minimal_kimi.json"
    raise SystemExit(main(items, report))
