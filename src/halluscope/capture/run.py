"""CLI-facing capture runner: every approved item, every user turn, cached."""

from __future__ import annotations

from pathlib import Path

from rich.progress import track

from halluscope.capture.activations import capture_prefix, user_turn_indices
from halluscope.capture.cache import ActivationCache, CaptureKey, content_sha
from halluscope.config import get_settings
from halluscope.data.io import approved, load_items
from halluscope.models.loader import load_model


def run_capture(model_key: str, items_path: Path, approved_only: bool = True) -> int:
    cfg = get_settings()
    spec = cfg.model_spec(model_key)
    items = load_items(items_path)
    if approved_only:
        items = approved(items)
    cache = ActivationCache(cfg.resolved_cache_dir() / "activations")

    # Key on the dialogue prefix, not the item id: a rebuilt dataset reuses ids
    # for different text, and an id-only check would call that already captured.
    keys = {
        (it.id, k): CaptureKey(spec.id, it.id, k, content_sha(it.prefix(k)))
        for it in items
        for k in user_turn_indices(it)
    }
    todo = [
        (it, k) for it in items for k in user_turn_indices(it) if not cache.has(keys[(it.id, k)])
    ]
    if not todo:
        return 0
    loaded = load_model(spec)
    n = 0
    for it, k in track(todo, description=f"capture {spec.id}"):
        cap = capture_prefix(loaded, it.prefix(k), dtype=cfg.capture.dtype)
        cache.put(
            keys[(it.id, k)],
            cap,
            extra={
                "topic": it.topic,
                "label": it.label,
                "variant": it.variant,
                "family": it.family,
            },
        )
        n += 1
    return n
