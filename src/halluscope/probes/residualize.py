"""Probes on activations with surface format regressed out.

The 2026 critique of linear probing on hidden states is that a probe reported
as reading a semantic property can be reading the shape of the prompt instead:
how long it is, how many numbers it contains, how many turns it has. A probe
that beats a bag-of-words baseline has not answered that, because the two are
different measurements rather than the same measurement with one variable
removed.

The test here is stricter. Fit a linear map from surface features to the
residual-stream matrix on the training split only, subtract its prediction
from every split, and re-fit the probe on what is left. Anything a linear
readout of length, digit count and turn count could have supplied is gone from
the features before the probe sees them. If the probe still separates the
classes, the signal is not that linear format channel. If it collapses, the
headline was resting on format and the report has to say so.

Three feature blocks are removed in turn, from the narrowest to the widest:

``length_digits``        the two cues the critique names
``surface``             every handcrafted count below
``surface_lexical``     those plus fifty latent dimensions of the dialogue's
                        own bag of words, so the wording is partly removed too

The widest block is a conservative bound, not the headline: its features are
fit on the same text the label was written into, so it can remove real signal
along with the artifact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from halluscope.capture.cache import ActivationCache, CaptureKey, content_sha
from halluscope.config import get_settings
from halluscope.data.io import approved, load_items
from halluscope.data.schema import Item
from halluscope.data.splits import grouped_split
from halluscope.eval.metrics import auroc, evaluate_scores, paired_bootstrap_delta
from halluscope.probes.baselines import dialogue_text, tfidf_scores
from halluscope.probes.linear import LinearProbe
from halluscope.probes.run import _pair_masks, _source_factory, gap_label, probe_result_path
from halluscope.probes.sweep import layer_sweep

LENGTH_DIGIT_FEATURES = (
    "final_chars",
    "final_words",
    "dialogue_chars",
    "dialogue_words",
    "final_digits",
    "final_numbers",
    "dialogue_digits",
)

SURFACE_FEATURES = LENGTH_DIGIT_FEATURES + (
    "n_user_turns",
    "final_question_marks",
    "dialogue_question_marks",
    "final_mean_word_len",
    "final_punctuation",
    "final_uppercase",
)


def _features_for(item: Item) -> dict[str, float]:
    final = item.turns[-1].content
    full = dialogue_text(item)
    fw = final.split()
    return {
        "final_chars": len(final),
        "final_words": len(fw),
        "dialogue_chars": len(full),
        "dialogue_words": len(full.split()),
        "final_digits": sum(c.isdigit() for c in final),
        "final_numbers": len(re.findall(r"\d+", final)),
        "dialogue_digits": sum(c.isdigit() for c in full),
        "n_user_turns": float(item.n_user_turns),
        "final_question_marks": final.count("?"),
        "dialogue_question_marks": full.count("?"),
        "final_mean_word_len": float(np.mean([len(w) for w in fw])) if fw else 0.0,
        "final_punctuation": sum(c in ",.;:!-()" for c in final),
        "final_uppercase": sum(c.isupper() for c in final),
    }


def surface_matrix(items: list[Item], names: tuple[str, ...]) -> np.ndarray:
    rows = [_features_for(i) for i in items]
    return np.array([[float(r[n]) for n in names] for r in rows], dtype=float)


def _lexical_block(
    train: list[Item], others: list[list[Item]], k: int, seed: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Latent dimensions of the training split's own bag of words.

    Fitting the vectorizer and the decomposition on train only keeps the test
    split out of the transform, the same discipline the probe itself follows.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Z = vec.fit_transform([dialogue_text(i) for i in train])
    k = int(min(k, max(1, min(Z.shape) - 1)))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    return svd.fit_transform(Z), [
        svd.transform(vec.transform([dialogue_text(i) for i in g])) for g in others
    ]


def residualize_splits(
    F_tr: np.ndarray,
    F_other: list[np.ndarray],
    X_tr: np.ndarray,
    X_other: list[np.ndarray],
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Subtract the part of each activation a linear map of the features predicts.

    The map is fit on the training split and applied everywhere, so no test row
    contributes to what is removed from it.
    """
    mu, sd = F_tr.mean(axis=0), F_tr.std(axis=0) + 1e-8
    Ztr = (F_tr - mu) / sd
    model = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(Ztr, X_tr)
    res_tr = X_tr - model.predict(Ztr)
    res_other = [X - model.predict((F - mu) / sd) for F, X in zip(F_other, X_other, strict=True)]
    return res_tr, res_other


def _variance_removed(X: np.ndarray, R: np.ndarray) -> float:
    total = float(np.sum((X - X.mean(axis=0)) ** 2))
    left = float(np.sum((R - R.mean(axis=0)) ** 2))
    return 1.0 - left / total if total > 0 else float("nan")


def _residual_source(raw, F_tr: np.ndarray, F_va: np.ndarray, F_te: np.ndarray):
    """A layer-sweep feature source that residualises on the way through, once per layer.

    The map is refit for every layer because what format predicts about the
    residual stream is itself layer dependent.
    """
    cached: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def source(layer: int):
        if layer not in cached:
            X_tr, X_va, X_te = raw(layer)
            r_tr, (r_va, r_te) = residualize_splits(F_tr, [F_va, F_te], X_tr, [X_va, X_te])
            cached[layer] = (r_tr, r_va, r_te)
        return cached[layer]

    return source


def run_residual_probe(
    model_key: str,
    items_path: Path,
    target: str = "gap",
    pooling: str = "last",
    out_dir: Path = Path("results"),
    lexical_components: int = 50,
    tag: str = "",
) -> Path:
    """Re-fit the gap probe after removing linearly predictable surface format.

    Writes one JSON with, per feature block, the re-selected layer, the test
    AUROC with an interval, the same numbers split by variant pair, and paired
    bootstrap deltas against both the unmodified probe and the strongest
    bag-of-words baseline on identical test items.
    """
    if target != "gap":
        raise ValueError("residualisation is defined for the gap target")
    cfg = get_settings()
    spec = cfg.model_spec(model_key)
    cache = ActivationCache(cfg.resolved_cache_dir() / "activations")
    base_path = probe_result_path(out_dir, model_key, target, pooling, items_path)
    base = json.loads(base_path.read_text(encoding="utf-8"))

    items = approved(load_items(items_path))
    if base.get("exclude_seeds"):
        items = [i for i in items if i.source != "seed"]
    items = [
        i
        for i in items
        if cache.has(
            CaptureKey(spec.id, i.id, i.n_user_turns, content_sha(i.prefix(i.n_user_turns)))
        )
    ]
    splits = grouped_split(items, seed=cfg.split.seed, fractions=cfg.split.fractions)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    y_tr, y_va, y_te = (np.array([gap_label(i) for i in s]) for s in (tr, va, te))
    if [i.id for i in te] != base["probes"]["linear"]["test_ids"]:
        raise RuntimeError(
            f"test split does not match {base_path.name}; rerun the probe on this build first"
        )
    base_prob = np.array(base["probes"]["linear"]["test_prob"])

    n_layers = cache.n_layers(spec.id)
    layers = list(range(n_layers))
    masks = _pair_masks(te, y_te)

    surface = {
        name: tfidf_scores(tr, y_tr, te, final_turn_only=fo, seed=cfg.probe.seed)
        for name, fo in (("tfidf_dialogue", False), ("tfidf_final_turn", True))
    }
    best_surface = max(surface, key=lambda k: auroc(y_te, surface[k]))

    blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    for name, names in (("length_digits", LENGTH_DIGIT_FEATURES), ("surface", SURFACE_FEATURES)):
        blocks[name] = (
            surface_matrix(tr, names),
            surface_matrix(va, names),
            surface_matrix(te, names),
            list(names),
        )
    if lexical_components > 0:
        Ltr, (Lva, Lte) = _lexical_block([*tr], [va, te], lexical_components, cfg.probe.seed)
        blocks["surface_lexical"] = (
            np.hstack([blocks["surface"][0], Ltr]),
            np.hstack([blocks["surface"][1], Lva]),
            np.hstack([blocks["surface"][2], Lte]),
            [*SURFACE_FEATURES, f"tfidf_svd_{Ltr.shape[1]}"],
        )

    out: dict = {
        "model": spec.id,
        "model_key": model_key,
        "target": target,
        "pooling": pooling,
        "dataset": Path(items_path).stem,
        "tag": tag,
        "base_file": base_path.name,
        "base_layer": int(base["probes"]["linear"]["best_layer"]),
        "base_test": base["probes"]["linear"]["test"],
        "base_by_pair": base["probes"]["linear"].get("by_pair", {}),
        "n_items": len(items),
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "surface_baseline": best_surface,
        "blocks": {},
    }

    # How well the removed features predict the label on their own. If this is
    # near chance, the block was never a plausible shortcut in the first place.
    for name, (Ftr, _Fva, Fte, fnames) in blocks.items():
        probe = LinearProbe(seed=cfg.probe.seed).fit(Ftr, y_tr)
        feat_only = evaluate_scores(
            y_te,
            probe.decision(Fte),
            prob=probe.predict_proba(Fte),
            n_bootstrap=cfg.probe.n_bootstrap,
        ).to_dict()

        raw = _source_factory(cache, spec.id, pooling, tr, va, te)
        source = _residual_source(raw, Ftr, _Fva, Fte)

        res = layer_sweep(
            source,
            y_tr,
            y_va,
            y_te,
            layers,
            probe_factory=lambda: LinearProbe(seed=cfg.probe.seed),
            n_bootstrap=cfg.probe.n_bootstrap,
            seed=cfg.probe.seed,
            record_test_curve=True,
        )
        d = res.to_dict()
        prob = np.array(res.test_prob)
        d["by_pair"] = {
            pair: evaluate_scores(
                y_te[m], prob[m], prob=prob[m], n_bootstrap=cfg.probe.n_bootstrap
            ).to_dict()
            for pair, m in masks.items()
        }
        d["features"] = fnames
        d["n_features"] = int(blocks[name][0].shape[1])
        d["features_alone"] = feat_only
        d["variance_removed_at_base_layer"] = _variance_removed(
            raw(out["base_layer"])[0], source(out["base_layer"])[0]
        )
        d["vs_base_probe"] = paired_bootstrap_delta(
            auroc, y_te, prob, base_prob, n=cfg.probe.n_bootstrap
        )
        d["vs_surface"] = paired_bootstrap_delta(
            auroc, y_te, prob, surface[best_surface], n=cfg.probe.n_bootstrap
        )
        d["vs_surface_by_pair"] = {
            pair: paired_bootstrap_delta(
                auroc, y_te[m], prob[m], surface[best_surface][m], n=cfg.probe.n_bootstrap
            )
            for pair, m in masks.items()
        }
        # Also at the layer the unmodified probe chose, so the comparison is not
        # confounded by residualisation moving the selected layer.
        Xtr_b, Xva_b, Xte_b = source(out["base_layer"])
        at_base = LinearProbe(seed=cfg.probe.seed).fit(Xtr_b, y_tr)
        p_base = at_base.predict_proba(Xte_b)
        d["at_base_layer"] = evaluate_scores(
            y_te, at_base.decision(Xte_b), prob=p_base, n_bootstrap=cfg.probe.n_bootstrap
        ).to_dict()
        d["at_base_layer_vs_surface"] = paired_bootstrap_delta(
            auroc, y_te, p_base, surface[best_surface], n=cfg.probe.n_bootstrap
        )
        out["blocks"][name] = d

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else (f"_{Path(items_path).stem.removeprefix('items').lstrip('_')}")
    path = out_dir / f"residual_{model_key}_{target}_{pooling}{suffix.rstrip('_')}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path
