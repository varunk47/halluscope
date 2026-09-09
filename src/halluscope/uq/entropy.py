"""Token-level and layer-level entropy signals that need one forward pass or
one generation.

- ``predictive_entropy``  length-normalized negative mean token log-probability
  of a single sampled or greedy generation (Malinin and Gales, 2021).
- ``logit_lens_entropy``  entropy of the next-token distribution read off every
  layer's last-token state through the final norm and unembedding
  (the logit-lens idea; TriLens 2026 uses per-layer entropy for white-box
  hallucination detection).
"""

from __future__ import annotations

import numpy as np
import torch

from halluscope.data.schema import Turn
from halluscope.models.chat import Generation, render_chat
from halluscope.models.loader import LoadedModel


def predictive_entropy(gen: Generation) -> float:
    if not gen.logprobs:
        return float("nan")
    return float(-np.mean(gen.logprobs))


def _final_norm_and_head(model: torch.nn.Module):
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    norm = None
    for name in ("norm", "final_layernorm", "ln_f"):
        norm = getattr(base, name, None)
        if norm is not None:
            break
    head = getattr(model, "lm_head", None)
    if norm is None or head is None:
        raise RuntimeError("could not locate final norm or lm_head for logit lens")
    return norm, head


@torch.no_grad()
def logit_lens_entropy(
    loaded: LoadedModel, turns: list[Turn], layers: list[int] | None = None
) -> np.ndarray:
    """Entropy (nats) of the next-token distribution at each layer's last position. Shape (L+1,)."""
    tok = loaded.tokenizer
    prompt = render_chat(tok, turns, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(loaded.device)
    out = loaded.model(**enc, output_hidden_states=True, use_cache=False)
    norm, head = _final_norm_and_head(loaded.model)
    hs = out.hidden_states
    idx = layers if layers is not None else list(range(len(hs)))
    ents = np.full(len(hs), np.nan, dtype=np.float64)
    for li in idx:
        h = hs[li][:, -1:, :]
        logits = head(norm(h)).float()[0, -1]
        logp = torch.log_softmax(logits, dim=-1)
        ents[li] = float(-(logp.exp() * logp).sum().item())
    return ents
