"""Expand compact hand-written family specs into the four dialogue variants.

A spec holds the hand-written pieces; this module assembles them so that the
four variants share wording wherever they should and differ only where the
label differs. Assistant acknowledgement turns are short, neutral, and drawn
by a hash of the family id so they do not correlate with the label.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from halluscope.data.schema import Family, Item, Turn

ACKS: tuple[str, ...] = (
    "Got it. What would you like me to do?",
    "Noted. What is the task?",
    "Understood. Ready when you are.",
    "Okay, I have that. What next?",
    "Thanks, that helps. What should I work on?",
    "Clear. Tell me what to build.",
)


class FamilySpec(BaseModel):
    family: str
    topic: str
    setup: str = Field(min_length=1)
    specified: str = Field(min_length=1)
    underspecified: str = Field(min_length=1)
    gap: str = Field(min_length=1)
    question: str = Field(min_length=1)
    assumption: str = Field(min_length=1)
    fill: str = Field(min_length=1)
    final: str = Field(min_length=1)
    contradiction: str = Field(min_length=1)


def _ack(family: str, salt: str) -> str:
    h = hashlib.sha256(f"{family}:{salt}".encode()).digest()
    return ACKS[h[0] % len(ACKS)]


def expand(spec: FamilySpec) -> Family:
    f = spec.family
    a = Item(
        id=f"{f}-a",
        family=f,
        topic=spec.topic,
        variant="a",
        label="specified",
        turns=[Turn(role="user", content=spec.specified)],
        source="seed",
        review_status="approved",
    )
    b = Item(
        id=f"{f}-b",
        family=f,
        topic=spec.topic,
        variant="b",
        label="underspecified",
        turns=[Turn(role="user", content=spec.underspecified)],
        gap=spec.gap,
        expected_clarifying_question=spec.question,
        plausible_silent_assumption=spec.assumption,
        reference_specified_variant=a.id,
        source="seed",
        review_status="approved",
    )
    c = Item(
        id=f"{f}-c",
        family=f,
        topic=spec.topic,
        variant="c",
        label="specified",
        turns=[
            Turn(role="user", content=f"{spec.setup} {spec.fill}"),
            Turn(role="assistant", content=_ack(f, "c")),
            Turn(role="user", content=spec.final),
        ],
        source="seed",
        review_status="approved",
    )
    d = Item(
        id=f"{f}-d",
        family=f,
        topic=spec.topic,
        variant="d",
        label="inconsistent",
        turns=[
            Turn(role="user", content=f"{spec.setup} {spec.fill}"),
            Turn(role="assistant", content=_ack(f, "d")),
            Turn(role="user", content=f"{spec.contradiction} {spec.final}"),
        ],
        gap=f"contradiction: {spec.contradiction}",
        expected_clarifying_question=(
            "Your latest instruction conflicts with what you specified earlier. Which one should I follow?"
        ),
        plausible_silent_assumption="silently follows the most recent instruction and drops the earlier constraint",
        reference_specified_variant=c.id,
        source="seed",
        review_status="approved",
    )
    return Family(family=f, topic=spec.topic, items=[a, b, c, d])
