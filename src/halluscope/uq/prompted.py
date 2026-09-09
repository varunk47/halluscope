"""Self-report baselines: P(True) and verbalized confidence.

- ``p_true``      Kadavath et al. 2022: show the model its own answer and read
                  the probability it assigns to "True".
- ``verbalized``  Lin et al. 2022, Tian et al. 2023: ask for a 0 to 100
                  confidence in words.
Both return uncertainty as 1 - confidence to match the project convention.
"""

from __future__ import annotations

import re

import torch

from halluscope.data.schema import Turn
from halluscope.models.chat import generate, render_chat
from halluscope.models.loader import LoadedModel


@torch.no_grad()
def p_true(loaded: LoadedModel, turns: list[Turn], answer: str) -> float:
    tok = loaded.tokenizer
    check = list(turns) + [
        Turn(role="assistant", content=answer),
        Turn(
            role="user",
            content=(
                "Was the request above fully specified so that the reply could be given without "
                "guessing any missing detail, and is the reply a correct response to it? "
                "Answer with a single word: True or False."
            ),
        ),
    ]
    prompt = render_chat(tok, check, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(loaded.device)
    logits = loaded.model(**enc, use_cache=False).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    t_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("True", " True", "true")}
    f_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("False", " False", "false")}
    pt = torch.logsumexp(logp[list(t_ids)], dim=0).exp().item()
    pf = torch.logsumexp(logp[list(f_ids)], dim=0).exp().item()
    conf = pt / max(pt + pf, 1e-12)
    return 1.0 - conf


def verbalized_confidence(loaded: LoadedModel, turns: list[Turn], seed: int = 0) -> float:
    ask = list(turns) + [
        Turn(
            role="user",
            content=(
                "Before answering: how confident are you, from 0 to 100, that this request "
                "contains every detail you need to complete it without making assumptions? "
                "Reply with only the number."
            ),
        ),
    ]
    gen = generate(loaded, ask, max_new_tokens=8, temperature=0.0, n=1, seed=seed)[0]
    m = re.search(r"\d{1,3}", gen.text)
    if not m:
        return float("nan")
    conf = min(100.0, float(m.group(0))) / 100.0
    return 1.0 - conf
