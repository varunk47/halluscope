"""Public underspecification benchmarks, mapped into our Item schema for
zero-shot transfer of the probe.

- AmbigQA (Min et al. 2020): questions with multiple valid interpretations
  are ``underspecified``; single-answer ones are ``specified``.
- AbstentionBench (2025): the ``underspecified`` slice is positive, a matched
  sample of answerable questions is negative. Loader is best-effort because
  the dataset layout on the Hub has changed over time; it degrades to a clear
  error rather than silently loading the wrong split.

Public items carry ``source="public"`` which relaxes the family/variant checks.
"""

from __future__ import annotations

import random

from halluscope.data.schema import Item, Turn


def _item(id_: str, topic: str, text: str, label: str) -> Item:
    return Item(
        id=id_,
        family=id_,
        topic=topic,
        variant="a" if label == "specified" else "b",
        label=label,  # type: ignore[arg-type]
        turns=[Turn(role="user", content=text)],
        gap="ambiguous question" if label != "specified" else None,
        expected_clarifying_question="Which interpretation did you mean?"
        if label != "specified"
        else None,
        source="public",
        review_status="approved",
    )


def load_ambigqa(n: int = 400, seed: int = 0, split: str = "validation") -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("sewon/ambig_qa", "light", split=split)
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)
    pos, neg = [], []
    for r in rows:
        types = r["annotations"]["type"]
        is_ambig = any(t == "multipleQAs" for t in types)
        (pos if is_ambig else neg).append(r)
    k = n // 2
    items = []
    for i, r in enumerate(pos[:k]):
        items.append(_item(f"ambigqa-amb-{i}", "public_ambigqa", r["question"], "underspecified"))
    for i, r in enumerate(neg[:k]):
        items.append(_item(f"ambigqa-clr-{i}", "public_ambigqa", r["question"], "specified"))
    return items


def load_abstentionbench_underspecified(n: int = 400, seed: int = 0) -> list[Item]:
    from datasets import load_dataset

    try:
        ds = load_dataset("facebook/AbstentionBench", split="test")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "AbstentionBench could not be loaded from the Hub with the expected layout; "
            "download it manually and adapt load_abstentionbench_underspecified"
        ) from e
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)
    pos = [
        r
        for r in rows
        if "underspec" in str(r.get("category", r.get("abstention_reason", ""))).lower()
    ]
    neg = [r for r in rows if not r.get("should_abstain", False)]
    k = n // 2
    items = []
    for i, r in enumerate(pos[:k]):
        items.append(
            _item(f"absbench-und-{i}", "public_abstentionbench", r["question"], "underspecified")
        )
    for i, r in enumerate(neg[:k]):
        items.append(
            _item(f"absbench-ans-{i}", "public_abstentionbench", r["question"], "specified")
        )
    return items
