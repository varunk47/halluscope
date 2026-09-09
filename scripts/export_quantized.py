"""One-time nf4 export so later loads need ~3 GB instead of the full bf16 shards.

Run when the machine has memory to spare (close editors and browsers first):
    uv run python scripts/export_quantized.py qwen
"""

from __future__ import annotations

import sys

from halluscope.config import get_settings
from halluscope.models.loader import export_quantized


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    spec = get_settings().model_spec(key)
    out = export_quantized(spec)
    print(f"exported {spec.id} (nf4) to {out}")


if __name__ == "__main__":
    main()
