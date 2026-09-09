"""Activation steering along the probe direction.

Adds ``alpha * direction`` to the residual stream at one layer during
generation, on every position, using a forward hook. This is the
"representation level" way to trigger clarifying behavior: if pushing the
state toward the underspecified side raises the asking rate, the direction
is causally relevant and not just a correlate.

Plain PyTorch hooks are used rather than nnsight so the same code path works
for the streaming-loaded and quantized models.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from halluscope.data.schema import Turn
from halluscope.models.chat import generate
from halluscope.models.loader import LoadedModel


def _decoder_layers(model: torch.nn.Module):
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    layers = getattr(base, "layers", None) or getattr(base, "h", None)
    if layers is None:
        raise RuntimeError("could not find decoder layers for steering")
    return layers


@contextmanager
def steer(loaded: LoadedModel, direction: np.ndarray, layer: int, alpha: float):
    """Context manager that adds alpha * direction to layer ``layer``'s output.

    ``layer`` indexes hidden_states (0 = embeddings); decoder layer i produces
    hidden_states[i + 1], so we hook decoder layer ``layer - 1``.
    """
    if layer <= 0:
        raise ValueError("steer at layer >= 1 (decoder outputs)")
    layers = _decoder_layers(loaded.model)
    dec = layers[layer - 1]
    d = torch.tensor(direction, dtype=torch.float32, device=loaded.device)
    d = d / (d.norm() + 1e-12)

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + (alpha * d).to(h.dtype)
        if isinstance(output, tuple):
            return (h, *output[1:])
        return h

    handle = dec.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def steered_generate(
    loaded: LoadedModel,
    turns: list[Turn],
    direction: np.ndarray,
    layer: int,
    alpha: float,
    max_new_tokens: int = 160,
) -> str:
    with steer(loaded, direction, layer, alpha):
        return generate(loaded, turns, max_new_tokens=max_new_tokens, temperature=0.0, n=1)[0].text


def asking_rate_curve(
    loaded: LoadedModel,
    items,
    direction: np.ndarray,
    layer: int,
    alphas: list[float],
    is_asking,
    max_new_tokens: int = 160,
) -> dict[float, float]:
    """Fraction of items where the steered reply asks, for each alpha.

    ``is_asking(item, text) -> bool`` is typically a judge call; a cheap
    heuristic (ends with a question mark and is short) works for smoke tests.
    """
    out: dict[float, float] = {}
    for a in alphas:
        asks = [
            is_asking(it, steered_generate(loaded, it.turns, direction, layer, a, max_new_tokens))
            for it in items
        ]
        out[a] = float(np.mean(asks)) if asks else float("nan")
    return out


def heuristic_is_asking(item, text: str) -> bool:
    t = text.strip()
    return t.endswith("?") and len(t) < 400
