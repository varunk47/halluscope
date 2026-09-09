"""Leakage-safe splits.

Rows are never split. The unit of assignment is the *family* (a seed and all
its paraphrases and variants), so no paraphrase or variant of a test item can
appear in training. Assignment is stratified by topic so every topic appears in
every split.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator

from halluscope.data.schema import Item

SplitName = str  # "train" | "val" | "test"


def family_root(item: Item) -> str:
    """Paraphrase families share their seed's root: ``rag-0037-p2`` -> ``rag-0037``."""
    parts = item.family.split("-p")
    return parts[0]


def grouped_split(
    items: list[Item],
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.6, 0.15, 0.25),
) -> dict[SplitName, list[Item]]:
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    rng = random.Random(seed)
    by_topic: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for it in items:
        root = family_root(it)
        if root not in seen:
            seen.add(root)
            by_topic[it.topic].append(root)

    assignment: dict[str, SplitName] = {}
    for topic in sorted(by_topic):
        roots = sorted(by_topic[topic])
        rng.shuffle(roots)
        n = len(roots)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        # guarantee at least one root per split that has a nonzero fraction
        wants_val = fractions[1] > 0
        wants_test = fractions[2] > 0
        reserve = int(wants_val) + int(wants_test)
        if n > reserve:
            n_train = max(1, min(n_train, n - reserve))
            n_val = max(1, min(n_val, n - n_train - int(wants_test))) if wants_val else 0
            if not wants_test:
                n_val = n - n_train
        for i, root in enumerate(roots):
            if i < n_train:
                assignment[root] = "train"
            elif i < n_train + n_val:
                assignment[root] = "val"
            else:
                assignment[root] = "test"

    out: dict[SplitName, list[Item]] = {"train": [], "val": [], "test": []}
    for it in items:
        out[assignment[family_root(it)]].append(it)
    return out


def leave_one_topic_out(items: list[Item]) -> Iterator[tuple[str, list[Item], list[Item]]]:
    """Yield (held_out_topic, train_items, test_items) for every topic present."""
    topics = sorted({it.topic for it in items})
    for topic in topics:
        train = [it for it in items if it.topic != topic]
        test = [it for it in items if it.topic == topic]
        yield topic, train, test


def assert_no_leakage(splits: dict[SplitName, list[Item]]) -> None:
    roots: dict[str, SplitName] = {}
    for name, its in splits.items():
        for it in its:
            r = family_root(it)
            if roots.setdefault(r, name) != name:
                raise AssertionError(f"family root {r} appears in {roots[r]} and {name}")
