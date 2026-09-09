"""Chat rendering and generation with per-token log probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from halluscope.data.schema import Turn
from halluscope.models.loader import LoadedModel

SYSTEM_PROMPT = (
    "You are an AI research assistant helping an engineer with machine learning "
    "and LLM work. Be concrete and brief."
)


@dataclass
class Generation:
    text: str
    token_ids: list[int]
    logprobs: list[float] = field(default_factory=list)

    @property
    def mean_logprob(self) -> float:
        return sum(self.logprobs) / max(1, len(self.logprobs))


def _messages(turns: list[Turn], system: str | None) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend({"role": t.role, "content": t.content} for t in turns)
    return msgs


def render_chat(
    tokenizer,
    turns: list[Turn],
    add_generation_prompt: bool = True,
    thinking: bool = False,
    system: str | None = SYSTEM_PROMPT,
) -> str:
    """Render a dialogue with the model's own chat template, thinking off by default."""
    msgs = _messages(turns, system)
    kwargs: dict = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(msgs, enable_thinking=thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, **kwargs)


@torch.no_grad()
def generate(
    loaded: LoadedModel,
    turns: list[Turn],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    n: int = 1,
    seed: int | None = None,
    system: str | None = SYSTEM_PROMPT,
) -> list[Generation]:
    tok = loaded.tokenizer
    prompt = render_chat(tok, turns, system=system)
    enc = tok(prompt, return_tensors="pt").to(loaded.device)
    if seed is not None:
        torch.manual_seed(seed)
    do_sample = temperature > 0
    out = loaded.model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        num_return_sequences=n,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tok.pad_token_id,
    )
    prompt_len = enc["input_ids"].shape[1]
    gens: list[Generation] = []
    # out.scores: tuple(steps) of (n, vocab) processed logits
    for i in range(n):
        ids = out.sequences[i, prompt_len:]
        lps: list[float] = []
        kept: list[int] = []
        for step, tid in enumerate(ids.tolist()):
            if tid == tok.pad_token_id and step > 0:
                break
            kept.append(tid)
            if step < len(out.scores):
                logits = out.scores[step][i].float()
                lps.append(torch.log_softmax(logits, dim=-1)[tid].item())
            if tid == tok.eos_token_id:
                break
        text = tok.decode(kept, skip_special_tokens=True).strip()
        gens.append(Generation(text=text, token_ids=kept, logprobs=lps))
    return gens


@torch.no_grad()
def generate_batch(
    loaded: LoadedModel,
    dialogues: list[list[Turn]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    batch_size: int = 8,
    system: str | None = SYSTEM_PROMPT,
) -> list[Generation]:
    """One generation per dialogue, batched with left padding. No logprobs."""
    tok = loaded.tokenizer
    tok.padding_side = "left"
    out_all: list[Generation] = []
    do_sample = temperature > 0
    for i in range(0, len(dialogues), batch_size):
        chunk = dialogues[i : i + batch_size]
        prompts = [render_chat(tok, t, system=system) for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(
            loaded.device
        )
        out = loaded.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tok.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        for row in out[:, prompt_len:]:
            ids = row.tolist()
            kept: list[int] = []
            for tid in ids:
                if tid == tok.pad_token_id or tid == tok.eos_token_id:
                    break
                kept.append(tid)
            out_all.append(
                Generation(text=tok.decode(kept, skip_special_tokens=True).strip(), token_ids=kept)
            )
    return out_all
