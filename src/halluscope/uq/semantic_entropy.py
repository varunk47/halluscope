"""Semantic entropy (Kuhn, Gal, Farquhar, ICLR 2023; Farquhar et al., Nature 2024).

Sample K answers, cluster them by bidirectional entailment, and take the
entropy of the cluster distribution. Clustering uses a judge call per pair
(bidirectional entailment through the LLM) or any user-supplied
``equivalent(a, b) -> bool``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Equiv = Callable[[str, str], bool]


def cluster_by_equivalence(texts: list[str], equivalent: Equiv) -> list[int]:
    """Greedy single-pass clustering: assign each text to the first cluster whose
    representative it is equivalent to, else start a new cluster."""
    reps: list[str] = []
    assign: list[int] = []
    for t in texts:
        for ci, r in enumerate(reps):
            if equivalent(t, r):
                assign.append(ci)
                break
        else:
            reps.append(t)
            assign.append(len(reps) - 1)
    return assign


def semantic_entropy(
    texts: list[str], equivalent: Equiv, weights: list[float] | None = None
) -> float:
    """Entropy (nats) over semantic clusters. ``weights`` may carry per-sample
    sequence probabilities; uniform when absent, which is the discrete estimator
    used in the Nature paper."""
    if not texts:
        return float("nan")
    assign = cluster_by_equivalence(texts, equivalent)
    n_clusters = max(assign) + 1
    w = np.ones(len(texts)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    p = np.zeros(n_clusters)
    for a, wi in zip(assign, w, strict=True):
        p[a] += wi
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def exact_match_equivalent(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def make_llm_equivalence(client, alias: str, context: str) -> Equiv:
    """Bidirectional entailment judged by an LLM through the JudgeClient."""
    from pydantic import BaseModel

    class Entail(BaseModel):
        a_entails_b: bool
        b_entails_a: bool

    cache: dict[tuple[str, str], bool] = {}

    def prefetch(texts: list[str], workers: int = 4) -> None:
        """Judge every distinct pair up front, a few at a time.

        The clustering below asks about pairs one after another, and each
        answer takes several seconds from a hosted model. The K samples give
        at most K(K-1)/2 pairs, so asking them all in parallel first costs a
        few extra calls and cuts the wall time per item several-fold.
        """
        from concurrent.futures import ThreadPoolExecutor

        uniq = sorted(set(texts))
        pairs = [(a, b) for i, a in enumerate(uniq) for b in uniq[i + 1 :] if (a, b) not in cache]
        if not pairs:
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda ab: equivalent(*ab), pairs))

    def equivalent(a: str, b: str) -> bool:
        key = (a, b) if a <= b else (b, a)
        if key in cache:
            return cache[key]
        msgs = [
            {
                "role": "system",
                "content": "You decide whether two assistant replies to the same request mean the same thing.",
            },
            {
                "role": "user",
                "content": (
                    f"Request:\n{context}\n\nReply A:\n{a}\n\nReply B:\n{b}\n\n"
                    "Does A entail B, and does B entail A, in the sense of proposing the same concrete plan or answer?"
                ),
            },
        ]
        v = client.complete(alias, msgs, Entail, tag="se:entail")
        cache[key] = bool(v.a_entails_b and v.b_entails_a)
        return cache[key]

    equivalent.prefetch = prefetch  # type: ignore[attr-defined]
    return equivalent
