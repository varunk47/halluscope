"""Position-swapped pairwise judging.

Every pairwise decision is asked twice with A and B exchanged. Agreement gives
the verdict; disagreement is reported as ``inconsistent`` rather than resolved
by a coin flip, so position bias is measured instead of hidden.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from halluscope.judge.client import JudgeClient
from halluscope.judge.rubrics import PairwiseVerdict

PromptBuilder = Callable[[str, str], list[dict[str, str]]]
Outcome = Literal["A", "B", "tie", "inconsistent"]


def pairwise_swapped(
    client: JudgeClient, alias: str, build: PromptBuilder, a: str, b: str, tag: str = ""
) -> Outcome:
    first = client.complete(alias, build(a, b), PairwiseVerdict, tag=f"{tag}:ab").winner
    second_raw = client.complete(alias, build(b, a), PairwiseVerdict, tag=f"{tag}:ba").winner
    second = {"A": "B", "B": "A", "tie": "tie"}[second_raw]
    if first == second:
        return first  # type: ignore[return-value]
    if "tie" in (first, second):
        return "tie"
    return "inconsistent"
