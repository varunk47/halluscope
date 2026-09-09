"""Load the primary model on the GPU, capture one item, generate once, and
report memory and timing. Run: ``uv run python scripts/smoke_gpu.py [model_key]``."""

from __future__ import annotations

import sys
import time

import torch

from halluscope.capture.activations import capture_prefix
from halluscope.config import get_settings
from halluscope.data.io import load_items
from halluscope.models.chat import generate
from halluscope.models.loader import load_model
from halluscope.uq.entropy import logit_lens_entropy


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    cfg = get_settings()
    spec = cfg.model_spec(key)
    print(f"loading {spec.id} quant={spec.quant}")
    t0 = time.time()
    loaded = load_model(spec)
    print(f"loaded in {time.time() - t0:.1f}s; layers={loaded.n_layers} hidden={loaded.hidden}")
    if torch.cuda.is_available():
        print(
            f"vram allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )

    items = load_items(cfg.paths.seeds_dir)
    item = next(i for i in items if i.variant == "d")
    t0 = time.time()
    cap = capture_prefix(loaded, item.turns)
    print(f"capture {cap.last.shape} in {time.time() - t0:.2f}s, tokens={cap.n_tokens}")

    t0 = time.time()
    ent = logit_lens_entropy(loaded, item.turns)
    print(
        f"logit-lens entropy per layer (first 5, last 5): {ent[:5].round(2)} ... {ent[-5:].round(2)} in {time.time() - t0:.2f}s"
    )

    t0 = time.time()
    gen = generate(loaded, item.turns, max_new_tokens=120, temperature=0.0, n=1)[0]
    dt = time.time() - t0
    print(
        f"generated {len(gen.token_ids)} tokens in {dt:.1f}s ({len(gen.token_ids) / max(dt, 1e-6):.1f} tok/s)"
    )
    print("---- item ----")
    for t in item.turns:
        print(f"{t.role}: {t.content}")
    print("---- reply ----")
    print(gen.text)
    if torch.cuda.is_available():
        print(f"peak vram {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
