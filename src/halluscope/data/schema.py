"""Dataset schema for UnderspecAI.

One *family* is a task request written four ways from the same seed:

- ``a``  specified, single turn
- ``b``  underspecified, single turn (the gap is simply missing)
- ``c``  specified through context: the gap is filled in an earlier turn, the
         final user turn alone would be underspecified
- ``d``  inconsistent: the gap is filled, then contradicted in a later turn

Labels follow from the variant, and the validator enforces that.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Label = Literal["specified", "underspecified", "inconsistent"]
Variant = Literal["a", "b", "c", "d"]
Role = Literal["user", "assistant"]
Source = Literal["seed", "augmented", "public"]
ReviewStatus = Literal["pending", "approved", "edited", "rejected"]

TOPICS: tuple[str, ...] = (
    "finetuning",
    "rag",
    "evaluation",
    "agents",
    "prompting",
    "serving",
    "data",
    "safety",
)

VARIANT_LABEL: dict[str, Label] = {
    "a": "specified",
    "b": "underspecified",
    "c": "specified",
    "d": "inconsistent",
}


class Turn(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class Item(BaseModel):
    id: str
    family: str
    topic: str
    variant: Variant
    label: Label
    turns: list[Turn] = Field(min_length=1)
    gap: str | None = None
    expected_clarifying_question: str | None = None
    plausible_silent_assumption: str | None = None
    reference_specified_variant: str | None = None
    source: Source = "seed"
    review_status: ReviewStatus = "pending"
    reviewed_by: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Item:
        if self.source != "public":
            expected = VARIANT_LABEL[self.variant]
            if self.label != expected:
                raise ValueError(
                    f"{self.id}: variant {self.variant!r} must have label {expected!r}, got {self.label!r}"
                )
            if self.topic not in TOPICS:
                raise ValueError(f"{self.id}: unknown topic {self.topic!r}")
            if self.label != "specified":
                if not self.gap:
                    raise ValueError(f"{self.id}: non-specified items need a gap")
                if not self.expected_clarifying_question:
                    raise ValueError(
                        f"{self.id}: non-specified items need an expected clarifying question"
                    )
            if self.variant in ("c", "d") and len(self.turns) < 3:
                raise ValueError(f"{self.id}: variants c and d must be multi-turn")
        if self.turns[-1].role != "user":
            raise ValueError(f"{self.id}: dialogue must end on a user turn")
        return self

    @property
    def is_flag_target(self) -> bool:
        """True when the assistant *should* ask or flag rather than answer."""
        return self.label != "specified"

    @property
    def n_user_turns(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    def prefix(self, n_user_turns: int) -> list[Turn]:
        """Dialogue up to and including the n-th user turn (1-indexed)."""
        out: list[Turn] = []
        seen = 0
        for t in self.turns:
            out.append(t)
            if t.role == "user":
                seen += 1
                if seen == n_user_turns:
                    break
        return out


class Family(BaseModel):
    """Convenience wrapper used by seeds files and the augmenter."""

    family: str
    topic: str
    items: list[Item] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _check_variants(self) -> Family:
        variants = sorted(i.variant for i in self.items)
        if variants != ["a", "b", "c", "d"]:
            raise ValueError(f"{self.family}: needs exactly variants a, b, c, d; got {variants}")
        for i in self.items:
            if i.family != self.family or i.topic != self.topic:
                raise ValueError(f"{i.id}: family/topic mismatch")
        return self
