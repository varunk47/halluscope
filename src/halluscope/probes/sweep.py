"""Layer sweep with nested selection.

The best layer is chosen on the validation split only. The test split is
touched once, for the chosen layer, and reported with bootstrap intervals.
Per-layer test curves are also recorded for the figure, but they are computed
after selection and never influence it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from halluscope.eval.metrics import MetricsWithCI, auroc, evaluate_scores

# get(layer) -> (X_train, X_val, X_test)
FeatureSource = Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass
class SweepResult:
    probe: str
    best_layer: int
    per_layer_val_auroc: dict[int, float]
    test: MetricsWithCI
    per_layer_test_auroc: dict[int, float] = field(default_factory=dict)
    test_prob: list[float] = field(default_factory=list)
    val_prob: list[float] = field(default_factory=list)
    fitted: object | None = None

    def to_dict(self) -> dict:
        return {
            "probe": self.probe,
            "best_layer": self.best_layer,
            "per_layer_val_auroc": {str(k): v for k, v in self.per_layer_val_auroc.items()},
            "per_layer_test_auroc": {str(k): v for k, v in self.per_layer_test_auroc.items()},
            "test": self.test.to_dict(),
        }


def layer_sweep(
    source: FeatureSource,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    layers: list[int],
    probe_factory: Callable[[], object],
    n_bootstrap: int = 1000,
    seed: int = 0,
    record_test_curve: bool = True,
) -> SweepResult:
    y_train, y_val, y_test = (np.asarray(v) for v in (y_train, y_val, y_test))
    val_scores: dict[int, float] = {}
    for layer in layers:
        Xtr, Xva, _ = source(layer)
        probe = probe_factory().fit(Xtr, y_train)
        val_scores[layer] = auroc(y_val, probe.decision(Xva))

    best = max(val_scores, key=lambda k: (val_scores[k], -k))
    Xtr, Xva, Xte = source(best)
    probe = probe_factory().fit(Xtr, y_train)
    test_metrics = evaluate_scores(
        y_test,
        probe.decision(Xte),
        prob=probe.predict_proba(Xte),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    result = SweepResult(
        probe=getattr(probe, "name", type(probe).__name__),
        best_layer=int(best),
        per_layer_val_auroc=val_scores,
        test=test_metrics,
        test_prob=probe.predict_proba(Xte).tolist(),
        val_prob=probe.predict_proba(Xva).tolist(),
        fitted=probe,
    )
    if record_test_curve:
        for layer in layers:
            Xtr, _, Xte = source(layer)
            p = probe_factory().fit(Xtr, y_train)
            result.per_layer_test_auroc[layer] = auroc(y_test, p.decision(Xte))
    return result
