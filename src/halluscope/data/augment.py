"""LLM augmentation of hand-written seed families.

The augmenter rewrites all nine text fields of a family spec: new surface
wording, new concrete numbers and names, same topic, same gap, same
contradiction logic. Output must validate as a ``FamilySpec`` or it is retried,
then skipped with a warning. Augmented items are ``review_status="pending"``
until a human approves them.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from pydantic import ValidationError

from halluscope.data.expand import FamilySpec, expand
from halluscope.data.io import load_seed_specs, save_items
from halluscope.data.schema import Item
from halluscope.judge.client import JudgeClient, JudgeError

log = logging.getLogger(__name__)

FIELDS = (
    "setup",
    "specified",
    "underspecified",
    "gap",
    "question",
    "assumption",
    "fill",
    "final",
    "contradiction",
)


def augment_prompt(spec: FamilySpec, k: int) -> list[dict[str, str]]:
    src = {f: getattr(spec, f) for f in FIELDS}
    return [
        {
            "role": "system",
            "content": (
                "You write realistic requests that a machine learning engineer sends to an AI "
                "assistant. You will receive a family of nine related text fields and must "
                "produce a paraphrased family with the same structure and the same logical "
                "relationships, but different wording, different concrete numbers, different "
                "tool or model names where sensible, and a plausible different product context. "
                "Rules: the 'specified' field must contain every detail needed to act; the "
                "'underspecified' field must omit exactly the kind of details described in 'gap'; "
                "'fill' must supply those details in prose; 'final' is a short follow-up "
                "instruction that alone would be underspecified; 'contradiction' must directly "
                "conflict with something in 'fill' while insisting the earlier plan also holds. "
                "Stay strictly within AI, LLM, and machine learning engineering. Never mention "
                "physics, mechanics, materials, or simulation. No em dashes."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {spec.topic}\nParaphrase variant number: {k}\n"
                f"Source family as JSON:\n{src}\n\n"
                "Return a JSON object with exactly these keys: " + ", ".join(FIELDS) + "."
            ),
        },
    ]


def augment_family(
    client: JudgeClient,
    spec: FamilySpec,
    n_paraphrases: int = 3,
    temperature_range: tuple[float, float] = (0.7, 1.0),
    seed: int = 0,
    alias: str = "augmenter",
) -> list[Item]:
    rng = random.Random(f"{spec.family}:{seed}")
    out: list[Item] = []
    for k in range(1, n_paraphrases + 1):
        temp = rng.uniform(*temperature_range)
        new_spec: FamilySpec | None = None
        for attempt in range(client.cfg.max_retries):
            try:
                raw = client.complete(
                    alias,
                    augment_prompt(spec, k),
                    _SpecFields,
                    temperature=temp,
                    tag=f"augment:{spec.family}:p{k}:{attempt}",
                )
                new_spec = FamilySpec(
                    family=f"{spec.family}-p{k}",
                    topic=spec.topic,
                    **raw.model_dump(exclude={"family", "topic"}),
                )
                break
            except (JudgeError, ValidationError) as e:
                log.warning("augment %s p%d attempt %d failed: %s", spec.family, k, attempt, e)
        if new_spec is None:
            log.warning("skipping %s p%d after retries", spec.family, k)
            continue
        fam = expand(new_spec)
        for it in fam.items:
            it.source = "augmented"
            it.review_status = "pending"
            out.append(it)
    return out


class _SpecFields(FamilySpec):
    """FamilySpec without family/topic, which the augmenter must not choose."""

    family: str = "x"
    topic: str = "rag"


def build_dataset(
    seeds_dir: Path | str,
    out_path: Path | str,
    n_paraphrases: int = 3,
    limit: int | None = None,
    client: JudgeClient | None = None,
) -> int:
    from halluscope.config import get_settings

    cfg = get_settings()
    client = client or JudgeClient(cfg.judge, cost_log=cfg.paths.cost_log)
    specs = load_seed_specs(seeds_dir)
    if limit is not None:
        specs = specs[:limit]
    items: list[Item] = []
    for spec in specs:
        items.extend(expand(spec).items)
        if n_paraphrases > 0:
            items.extend(
                augment_family(client, spec, n_paraphrases=n_paraphrases, seed=cfg.split.seed)
            )
    n = save_items(items, out_path)
    log.info("wrote %d items; %s", n, client.cost_summary())
    return n
