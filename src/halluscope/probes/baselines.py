"""Surface-text baselines that see no internal state.

If a probe on hidden states cannot beat these, the internal state adds
nothing over the words, and the report must say so.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from halluscope.data.schema import Item
from halluscope.eval.metrics import MetricsWithCI, evaluate_scores


def dialogue_text(item: Item, final_turn_only: bool = False) -> str:
    if final_turn_only:
        return item.turns[-1].content
    return "\n".join(f"{t.role}: {t.content}" for t in item.turns)


def tfidf_baseline(
    train: list[Item],
    y_train: np.ndarray,
    test: list[Item],
    y_test: np.ndarray,
    final_turn_only: bool = False,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> MetricsWithCI:
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=seed)),
        ]
    )
    pipe.fit([dialogue_text(i, final_turn_only) for i in train], y_train)
    Xte = [dialogue_text(i, final_turn_only) for i in test]
    score = pipe.decision_function(Xte)
    prob = pipe.predict_proba(Xte)[:, 1]
    return evaluate_scores(y_test, score, prob=prob, n_bootstrap=n_bootstrap, seed=seed)


def length_baseline(
    test: list[Item], y_test: np.ndarray, n_bootstrap: int = 1000, seed: int = 0
) -> MetricsWithCI:
    """Shorter final turns tend to be underspecified. A sanity baseline."""
    score = -np.array([len(i.turns[-1].content) for i in test], dtype=float)
    return evaluate_scores(y_test, score, n_bootstrap=n_bootstrap, seed=seed)
