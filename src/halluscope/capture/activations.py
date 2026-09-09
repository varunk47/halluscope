"""Per-layer residual-stream capture at the point where the assistant would
start answering.

Two poolings per layer:

- ``last``       the last prompt token (the generation-prompt position)
- ``mean_user``  mean over the tokens of the final user turn

Both are taken from every layer output in ``output_hidden_states`` so the
shape is ``(n_layers + 1, hidden)``: index 0 is the embedding output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from halluscope.data.schema import Item, Turn
from halluscope.models.chat import render_chat
from halluscope.models.loader import LoadedModel


@dataclass
class Capture:
    last: np.ndarray  # (L+1, H) float16
    mean_user: np.ndarray  # (L+1, H) float16
    n_tokens: int
    prompt_sha: str

    def pooled(self, pooling: str) -> np.ndarray:
        if pooling == "last":
            return self.last
        if pooling == "mean_user":
            return self.mean_user
        raise ValueError(f"unknown pooling {pooling!r}")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def user_turn_indices(item: Item) -> list[int]:
    """1-indexed user turn numbers at which we capture (every user turn)."""
    return list(range(1, item.n_user_turns + 1))


def _final_user_span(tokenizer, turns: list[Turn]) -> tuple[int, int]:
    """Token span [start, end) of the final user message inside the rendered prompt."""
    full_no_gen = render_chat(tokenizer, turns, add_generation_prompt=False)
    before = (
        render_chat(tokenizer, turns[:-1], add_generation_prompt=False) if len(turns) > 1 else ""
    )
    n_before = len(tokenizer(before, add_special_tokens=False)["input_ids"]) if before else 0
    n_full = len(tokenizer(full_no_gen, add_special_tokens=False)["input_ids"])
    # The final user block may include template wrappers; the mean over it is still
    # dominated by the user text, which is what we want.
    start = min(n_before, n_full - 1)
    return start, n_full


@torch.no_grad()
def capture_prefix(loaded: LoadedModel, turns: list[Turn], dtype: str = "float16") -> Capture:
    tok = loaded.tokenizer
    prompt = render_chat(tok, turns, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(loaded.device)
    out = loaded.model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple(L+1) of (1, seq, H)
    seq_len = enc["input_ids"].shape[1]
    start, end = _final_user_span(tok, turns)
    end = min(end, seq_len)
    start = max(0, min(start, end - 1))

    np_dtype = np.float16 if dtype == "float16" else np.float32
    last = torch.stack([h[0, -1, :] for h in hs], dim=0).float().cpu().numpy().astype(np_dtype)
    mean_user = (
        torch.stack([h[0, start:end, :].mean(dim=0) for h in hs], dim=0)
        .float()
        .cpu()
        .numpy()
        .astype(np_dtype)
    )
    return Capture(
        last=last, mean_user=mean_user, n_tokens=int(seq_len), prompt_sha=prompt_hash(prompt)
    )


def capture_item(
    loaded: LoadedModel, item: Item, turn_indices: list[int] | None = None
) -> dict[int, Capture]:
    idx = turn_indices or user_turn_indices(item)
    return {k: capture_prefix(loaded, item.prefix(k)) for k in idx}
