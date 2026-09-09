"""Shared result type for every uncertainty method.

Convention: ``value`` is higher when the method thinks the assistant should
not answer as-is (more uncertain, more likely to be a hallucination or an
underspecified request). Methods that natively produce confidence return its
negation so the convention holds everywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class UQScore:
    method: str
    value: float
    n_generations: int = 0
    n_forwards: int = 1
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
