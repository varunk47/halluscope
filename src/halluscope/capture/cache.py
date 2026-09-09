"""On-disk activation cache.

Layout: ``{root}/{model_slug}/{item_id}__t{turn}__{content_sha}.safetensors``
with tensors ``last`` and ``mean_user`` of shape ``(L+1, H)`` and JSON metadata
in the safetensors header. The cache is the boundary between GPU work and
everything else: probes, baselines, and figures only read from here.

The content hash is part of the filename because item ids are not unique across
dataset builds: a regenerated dataset reuses ``rag-0001-p1-b`` for different
text. Keying on the id alone silently served activations captured from the old
wording, so every entry is keyed by the dialogue prefix it was actually produced
from. Lookups that supply no hash resolve by glob and refuse to guess when two
builds are cached side by side.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from halluscope.capture.activations import Capture


def model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def content_sha(turns) -> str:
    """Stable short hash of a dialogue prefix, independent of tokenizer or model."""
    blob = "\n".join(f"{t.role}:{t.content}" for t in turns)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def shas_for(items) -> dict[str, str]:
    """Item id to content hash, for the final user turn of each item."""
    return {i.id: content_sha(i.prefix(i.n_user_turns)) for i in items}


@dataclass(frozen=True)
class CaptureKey:
    model_id: str
    item_id: str
    turn_index: int
    content_sha: str = ""

    def filename(self) -> str:
        stem = f"{self.item_id}__t{self.turn_index}"
        if self.content_sha:
            return f"{stem}__{self.content_sha}.safetensors"
        return f"{stem}.safetensors"


class AmbiguousCaptureError(KeyError):
    """Two dataset builds cached under one item id; the caller must say which."""


class ActivationCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, key: CaptureKey) -> Path:
        d = self.root / model_slug(key.model_id)
        if key.content_sha:
            return d / key.filename()
        # No hash given: accept a single cached build, refuse to pick between two.
        matches = sorted(d.glob(f"{key.item_id}__t{key.turn_index}__*.safetensors"))
        if len(matches) > 1:
            raise AmbiguousCaptureError(
                f"{len(matches)} cached builds for {key.item_id} turn {key.turn_index}; "
                "pass content_sha, or clear the stale build from the cache"
            )
        if matches:
            return matches[0]
        return d / key.filename()

    def has(self, key: CaptureKey) -> bool:
        try:
            return self._path(key).exists()
        except AmbiguousCaptureError:
            return False

    def put(self, key: CaptureKey, cap: Capture, extra: dict | None = None) -> Path:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_id": key.model_id,
            "item_id": key.item_id,
            "turn_index": str(key.turn_index),
            "content_sha": key.content_sha,
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
        shas: dict[str, str] | None = None,
    ) -> np.ndarray:
        """Stack one layer's pooled vectors for the given items, in order. (N, H) float32.

        ``shas`` maps item id to content hash; supply it whenever the caller holds
        the items, so the rows provably come from the text being studied.
        """
        rows = []
        for iid in item_ids:
            t = turn_index[iid] if isinstance(turn_index, dict) else turn_index
            cap = self.get(CaptureKey(model_id, iid, t, (shas or {}).get(iid, "")))
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
            iid, _, rest = p.stem.rpartition("__t")
            out.append((iid, int(rest.split("__")[0])))
        return out
