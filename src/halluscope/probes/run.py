"""End-to-end probe experiment from the activation cache.

For one model, one target, one pooling:
1. grouped split by family (and leave-one-topic-out),
2. layer sweep for linear, mass-mean, and MLP probes with validation-only
   layer selection and bootstrap intervals on test,
3. surface-text baselines on the same split,
4. JSON written to ``results/probe_{model}_{target}_{pooling}.json``.

Targets:
- ``gap``          dataset label: 1 when the item is underspecified or inconsistent
- ``will_assume``  behavioral label from ``results/behavior_{model}.json``
- ``will_ask``     behavioral label from the same file
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from halluscope.capture.cache import ActivationCache, CaptureKey, content_sha, shas_for
from halluscope.config import get_settings
from halluscope.data.io import approved, load_items
from halluscope.data.schema import Item
from halluscope.data.splits import grouped_split, leave_one_topic_out
from halluscope.eval.metrics import auroc, evaluate_scores, paired_bootstrap_delta
from halluscope.probes.baselines import (
    digit_count_baseline,
    length_baseline,
    tfidf_baseline,
    tfidf_scores,
)
from halluscope.probes.linear import LinearProbe, MassMeanProbe
from halluscope.probes.mlp import MLPProbe
from halluscope.probes.sweep import layer_sweep

PROBES = {
    "linear": LinearProbe,
    "massmean": MassMeanProbe,
    "mlp": MLPProbe,
}


def gap_label(item: Item) -> int:
    return int(item.label != "specified")


def load_behavior_labels(path: Path) -> dict[str, dict[str, int]]:
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    return {
        r["item_id"]: {"will_assume": int(r["silently_assumed"]), "will_ask": int(r["asked"])}
        for r in rows
    }


def labels_for(
    items: list[Item], target: str, behavior: dict[str, dict[str, int]] | None
) -> np.ndarray:
    if target == "gap":
        return np.array([gap_label(i) for i in items])
    if behavior is None:
        raise ValueError(f"target {target!r} needs behavior labels")
    return np.array([behavior[i.id][target] for i in items])


def _final_turn_index(items: list[Item]) -> dict[str, int]:
    return {i.id: i.n_user_turns for i in items}


def _source_factory(
    cache: ActivationCache,
    model_id: str,
    pooling: str,
    tr: list[Item],
    va: list[Item],
    te: list[Item],
):
    items = tr + va + te
    t_all = _final_turn_index(items)
    shas = shas_for(items)

    def source(layer: int):
        return (
            cache.matrix(model_id, [i.id for i in tr], t_all, layer, pooling, shas=shas),
            cache.matrix(model_id, [i.id for i in va], t_all, layer, pooling, shas=shas),
            cache.matrix(model_id, [i.id for i in te], t_all, layer, pooling, shas=shas),
        )

    return source


def _pair_masks(te: list[Item], y_te: np.ndarray) -> dict[str, np.ndarray]:
    """Test masks for the two variant pairs, dropping any that cannot be scored.

    The single-turn pair (a vs b) and the multi-turn pair (c vs d) ask different
    questions: c vs d is answerable only from the earlier context, so a number
    pooled over both says little about either.
    """
    out = {}
    for name, variants in (("single_turn_ab", ("a", "b")), ("multi_turn_cd", ("c", "d"))):
        mask = np.array([i.variant in variants for i in te])
        if mask.sum() >= 8 and len(np.unique(y_te[mask])) == 2:
            out[name] = mask
    return out


def run_probe(
    model_key: str,
    items_path: Path,
    target: str = "gap",
    pooling: str = "last",
    out_dir: Path = Path("results"),
    probes: tuple[str, ...] = ("linear", "massmean", "mlp"),
    loto: bool = True,
    exclude_seeds: bool = False,
    tag: str = "",
) -> Path:
    cfg = get_settings()
    spec = cfg.model_spec(model_key)
    cache = ActivationCache(cfg.resolved_cache_dir() / "activations")
    items = approved(load_items(items_path))
    if exclude_seeds:
        # hand-written seeds are not length-matched across variants; the augmented
        # set is, so the cleaner experiment uses augmented items only
        items = [i for i in items if i.source != "seed"]
    # Match on (id, final turn, content hash) so a stale build in the cache is
    # simply absent rather than quietly standing in for the current text.
    items = [
        i
        for i in items
        if cache.has(
            CaptureKey(spec.id, i.id, i.n_user_turns, content_sha(i.prefix(i.n_user_turns)))
        )
    ]
    if not items:
        raise RuntimeError(f"no cached activations for {spec.id}; run `halluscope capture` first")

    behavior = None
    if target != "gap":
        bpath = out_dir / f"behavior_{model_key}.json"
        behavior = load_behavior_labels(bpath)
        items = [i for i in items if i.id in behavior]

    n_layers = cache.n_layers(spec.id)
    layers = list(range(n_layers))
    splits = grouped_split(items, seed=cfg.split.seed, fractions=cfg.split.fractions)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    y_tr, y_va, y_te = (labels_for(s, target, behavior) for s in (tr, va, te))
    source = _source_factory(cache, spec.id, pooling, tr, va, te)

    out: dict = {
        "model": spec.id,
        "model_key": model_key,
        "target": target,
        "pooling": pooling,
        # Which build of the dataset this was run on. Three probe files that all
        # say "gap, last" are three different experiments, and a table that
        # cannot tell them apart invites reading the saturated one as the result.
        "dataset": Path(items_path).stem,
        "tag": tag,
        "exclude_seeds": exclude_seeds,
        "n_items": len(items),
        "n_layers": n_layers,
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "positive_rate_test": float(np.mean(y_te)),
        "probes": {},
        "baselines": {},
        "loto": {},
    }
    for name in probes:
        res = layer_sweep(
            source,
            y_tr,
            y_va,
            y_te,
            layers,
            probe_factory=lambda name=name: PROBES[name](seed=cfg.probe.seed),
            n_bootstrap=cfg.probe.n_bootstrap,
            seed=cfg.probe.seed,
            record_test_curve=(name == "linear"),
        )
        d = res.to_dict()
        d["test_prob"] = res.test_prob
        d["test_ids"] = [i.id for i in te]
        # The single-turn pair (a vs b) and the multi-turn pair (c vs d) are different
        # questions; c vs d is only answerable from context, so report them apart.
        probs = np.array(res.test_prob)
        d["by_pair"] = {
            pair_name: evaluate_scores(
                y_te[mask], probs[mask], prob=probs[mask], n_bootstrap=cfg.probe.n_bootstrap
            ).to_dict()
            for pair_name, mask in _pair_masks(te, y_te).items()
        }
        out["probes"][name] = d

    # Same train set and same test subsets the probes are scored on, so the
    # headline comparison is probe against words on identical footing. Without
    # the per-pair split a strong single-turn baseline hides a weak multi-turn one.
    for bname, final_only in (("tfidf_dialogue", False), ("tfidf_final_turn", True)):
        b = tfidf_baseline(
            tr, y_tr, te, y_te, final_turn_only=final_only, n_bootstrap=cfg.probe.n_bootstrap
        ).to_dict()
        b["by_pair"] = {
            pair_name: tfidf_baseline(
                tr,
                y_tr,
                [i for i, m in zip(te, mask, strict=True) if m],
                y_te[mask],
                final_turn_only=final_only,
                n_bootstrap=cfg.probe.n_bootstrap,
            ).to_dict()
            for pair_name, mask in _pair_masks(te, y_te).items()
        }
        out["baselines"][bname] = b
    out["baselines"]["length"] = length_baseline(
        te, y_te, n_bootstrap=cfg.probe.n_bootstrap
    ).to_dict()
    out["baselines"]["digit_count"] = digit_count_baseline(
        te, y_te, n_bootstrap=cfg.probe.n_bootstrap
    ).to_dict()

    # Head to head on identical test items. The interesting question is not
    # whether the probe clears chance, it is whether reading the hidden state
    # beats reading the words, and only a paired comparison can answer that.
    surface = {
        name: tfidf_scores(tr, y_tr, te, final_turn_only=fo, seed=cfg.probe.seed)
        for name, fo in (("tfidf_dialogue", False), ("tfidf_final_turn", True))
    }
    best_surface = max(surface, key=lambda k: auroc(y_te, surface[k]))
    out["probe_vs_surface"] = {"baseline": best_surface, "overall": {}, "by_pair": {}}
    masks = _pair_masks(te, y_te)
    for name, d in out["probes"].items():
        probs = np.array(d["test_prob"])
        out["probe_vs_surface"]["overall"][name] = paired_bootstrap_delta(
            auroc, y_te, probs, surface[best_surface], n=cfg.probe.n_bootstrap
        )
        out["probe_vs_surface"]["by_pair"][name] = {
            pair: paired_bootstrap_delta(
                auroc,
                y_te[m],
                probs[m],
                surface[best_surface][m],
                n=cfg.probe.n_bootstrap,
            )
            for pair, m in masks.items()
        }

    if loto:
        best_layer = out["probes"]["linear"]["best_layer"]
        for topic, tr_t, te_t in leave_one_topic_out(items):
            inner = grouped_split(tr_t, seed=cfg.split.seed, fractions=(0.8, 0.2, 0.0))
            trr, vaa = inner["train"], inner["val"]
            src = _source_factory(cache, spec.id, pooling, trr, vaa, te_t)
            res = layer_sweep(
                src,
                labels_for(trr, target, behavior),
                labels_for(vaa, target, behavior),
                labels_for(te_t, target, behavior),
                [best_layer],
                probe_factory=lambda: LinearProbe(seed=cfg.probe.seed),
                n_bootstrap=cfg.probe.n_bootstrap,
                seed=cfg.probe.seed,
                record_test_curve=False,
            )
            out["loto"][topic] = res.test.to_dict()

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ("_noseeds" if exclude_seeds else "")
    path = out_dir / f"probe_{model_key}_{target}_{pooling}{suffix}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return path
