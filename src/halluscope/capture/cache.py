"""On-disk activation cache.

Layout: ``{root}/{model_slug}/{item_id}__t{turn}.safetensors`` with tensors
``last`` and ``mean_user`` of shape ``(L+1, H)`` and JSON metadata in the
safetensors header. The cache is the boundary between GPU work and everything
else: probes, baselines, and figures only read from here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from halluscope.capture.activations import Capture


def model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


@dataclass(frozen=True)
class CaptureKey:
    model_id: str
    item_id: str
    turn_index: int

    def filename(self) -> str:
        return f"{self.item_id}__t{self.turn_index}.safetensors"


class ActivationCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, key: CaptureKey) -> Path:
        return self.root / model_slug(key.model_id) / key.filename()

    def has(self, key: CaptureKey) -> bool:
        return self._path(key).exists()

    def put(self, key: CaptureKey, cap: Capture, extra: dict | None = None) -> Path:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_id": key.model_id,
            "item_id": key.item_id,
            "turn_index": str(key.turn_index),
            "n_tokens": str(cap.n_tokens),
            "prompt_sha": cap.prompt_sha,
        }
        if extra:
            meta.update(
                {k: json.dumps(v) if not isinstance(v, str) else v for k, v in extra.items()}
            )
        save_file({"last": cap.last, "mean_user": cap.mean_user}, str(path), metadata=meta)
        return path

    def get(self, key: CaptureKey) -> Capture:
        path = self._path(key)
        if not path.exists():
            raise KeyError(f"no capture for {key}")
        tensors = load_file(str(path))
        meta = self.metadata(key)
        return Capture(
            last=tensors["last"],
            mean_user=tensors["mean_user"],
            n_tokens=int(meta.get("n_tokens", 0)),
            prompt_sha=meta.get("prompt_sha", ""),
        )

    def metadata(self, key: CaptureKey) -> dict[str, str]:
        from safetensors import safe_open

        with safe_open(str(self._path(key)), framework="np") as f:
            return dict(f.metadata() or {})

    def matrix(
        self,
        model_id: str,
        item_ids: list[str],
        turn_index: int | dict[str, int],
        layer: int,
        pooling: str = "last",
    ) -> np.ndarray:
        """Stack one layer's pooled vectors for the given items, in order. (N, H) float32."""
        rows = []
        for iid in item_ids:
            t = turn_index[iid] if isinstance(turn_index, dict) else turn_index
            cap = self.get(CaptureKey(model_id, iid, t))
            rows.append(cap.pooled(pooling)[layer].astype(np.float32))
        return np.stack(rows, axis=0)

    def n_layers(self, model_id: str) -> int:
        d = self.root / model_slug(model_id)
        first = next(d.glob("*.safetensors"), None)
        if first is None:
            raise KeyError(f"no captures for {model_id}")
        return int(load_file(str(first))["last"].shape[0])

    def list_items(self, model_id: str) -> list[tuple[str, int]]:
        d = self.root / model_slug(model_id)
        out = []
        for p in sorted(d.glob("*.safetensors")):
            stem = p.stem
            iid, _, t = stem.rpartition("__t")
            out.append((iid, int(t)))
        return out
