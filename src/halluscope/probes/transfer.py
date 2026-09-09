"""Cross-model transfer of a probe.

Hidden sizes differ across models, so a probe cannot be applied directly.
We test two honest notions of transfer:

1. ``relative_layer``  train and test the same probe *recipe* at the same
   relative depth on each model (no weight sharing): does the phenomenon
   transfer, not the weights.
2. ``direction_cka``    centered kernel alignment between the two models'
   representations of the same items at matched relative depth, as a measure
   of how similar the geometry is.
"""

from __future__ import annotations

import numpy as np

from halluscope.capture.cache import ActivationCache, shas_for
from halluscope.data.schema import Item
from halluscope.eval.metrics import MetricsWithCI, evaluate_scores
from halluscope.probes.linear import LinearProbe


def matched_layer(src_layer: int, src_n_layers: int, dst_n_layers: int) -> int:
    rel = src_layer / max(1, src_n_layers - 1)
    return int(round(rel * (dst_n_layers - 1)))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) between two (N, d) representations."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro") + 1e-12))


def recipe_transfer(
    cache: ActivationCache,
    model_id: str,
    layer: int,
    train: list[Item],
    y_train: np.ndarray,
    test: list[Item],
    y_test: np.ndarray,
    pooling: str = "last",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> MetricsWithCI:
    t_idx = {i.id: i.n_user_turns for i in train + test}
    shas = shas_for(train + test)
    Xtr = cache.matrix(model_id, [i.id for i in train], t_idx, layer, pooling, shas=shas)
    Xte = cache.matrix(model_id, [i.id for i in test], t_idx, layer, pooling, shas=shas)
    p = LinearProbe(seed=seed).fit(Xtr, y_train)
    return evaluate_scores(
        y_test, p.decision(Xte), prob=p.predict_proba(Xte), n_bootstrap=n_bootstrap, seed=seed
    )


def representation_similarity(
    cache: ActivationCache,
    model_a: str,
    model_b: str,
    items: list[Item],
    layer_a: int,
    layer_b: int,
    pooling: str = "last",
) -> float:
    t_idx = {i.id: i.n_user_turns for i in items}
    shas = shas_for(items)
    ids = [i.id for i in items]
    A = cache.matrix(model_a, ids, t_idx, layer_a, pooling, shas=shas)
    B = cache.matrix(model_b, ids, t_idx, layer_b, pooling, shas=shas)
    return linear_cka(A, B)
