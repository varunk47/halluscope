"""Cross-model transfer and activation steering, end to end.

Both start from a finished gap-probe run: its best layer fixes where to read
and, for steering, where to push. Both write one JSON file under ``results``
that the report and the results page pick up by prefix.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from rich.progress import track

from halluscope.capture.cache import ActivationCache, CaptureKey, shas_for
from halluscope.config import get_settings
from halluscope.data.io import approved, load_items
from halluscope.data.schema import Item
from halluscope.data.splits import grouped_split
from halluscope.eval.metrics import auroc
from halluscope.probes.linear import LinearProbe, MassMeanProbe
from halluscope.probes.run import gap_label, probe_result_path
from halluscope.probes.steer import steered_generate
from halluscope.probes.transfer import matched_layer, recipe_transfer, representation_similarity

PAIRS = {"single_turn_ab": ("a", "b"), "multi_turn_cd": ("c", "d")}


def _experiment_items(
    cfg, spec_ids: list[str], items_path: Path, exclude_seeds: bool
) -> list[Item]:
    """The items a probe run used, restricted to those every listed model has cached."""
    cache = ActivationCache(cfg.resolved_cache_dir() / "activations")
    items = approved(load_items(items_path))
    if exclude_seeds:
        items = [i for i in items if i.source != "seed"]
    shas = shas_for(items)
    return [
        i
        for i in items
        if all(cache.has(CaptureKey(m, i.id, i.n_user_turns, shas[i.id])) for m in spec_ids)
    ]


def _labels(items: list[Item]) -> np.ndarray:
    return np.array([gap_label(i) for i in items])


def _tag_suffix(items_path: Path, tag: str) -> str:
    t = tag or Path(items_path).stem.removeprefix("items").lstrip("_")
    return f"_{t}" if t else ""


# ---- transfer -------------------------------------------------------------------------


def run_transfer(
    src_key: str,
    dst_key: str,
    items_path: Path,
    out_dir: Path = Path("results"),
    pooling: str = "last",
    tag: str = "",
) -> Path:
    """Same probe recipe on a second model at the matched relative depth, plus CKA.

    Weights are not shared: hidden sizes differ. What is tested is whether the
    phenomenon appears at the same depth in another model, and how alike the
    two models' geometry is on the same items.
    """
    cfg = get_settings()
    src, dst = cfg.model_spec(src_key), cfg.model_spec(dst_key)
    cache = ActivationCache(cfg.resolved_cache_dir() / "activations")
    with open(
        probe_result_path(out_dir, src_key, "gap", pooling, items_path), encoding="utf-8"
    ) as fh:
        src_res = json.load(fh)
    src_layer = int(src_res["probes"]["linear"]["best_layer"])
    items = _experiment_items(cfg, [src.id, dst.id], items_path, bool(src_res.get("exclude_seeds")))
    if len(items) < 40:
        raise RuntimeError(f"only {len(items)} items cached for both {src.id} and {dst.id}")
    sp = grouped_split(items, seed=cfg.split.seed, fractions=cfg.split.fractions)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    y_tr, y_va, y_te = _labels(tr), _labels(va), _labels(te)
    src_n, dst_n = cache.n_layers(src.id), cache.n_layers(dst.id)
    dst_layer = matched_layer(src_layer, src_n, dst_n)

    # The destination model's own best layer, chosen on validation like the source run.
    t_idx = {i.id: i.n_user_turns for i in items}
    shas = shas_for(items)
    val_auc = {}
    for layer in track(range(dst_n), description=f"sweep {dst_key}"):
        Xtr = cache.matrix(dst.id, [i.id for i in tr], t_idx, layer, pooling, shas=shas)
        Xva = cache.matrix(dst.id, [i.id for i in va], t_idx, layer, pooling, shas=shas)
        val_auc[layer] = auroc(y_va, LinearProbe(seed=cfg.probe.seed).fit(Xtr, y_tr).decision(Xva))
    dst_best = max(val_auc, key=lambda k: (val_auc[k], -k))

    def pairs(model_id: str, layer: int) -> dict:
        out = {}
        for name, variants in PAIRS.items():
            mask = np.array([i.variant in variants for i in te])
            if mask.sum() >= 8 and len(np.unique(y_te[mask])) == 2:
                sub = [i for i, m in zip(te, mask, strict=True) if m]
                out[name] = recipe_transfer(
                    cache,
                    model_id,
                    layer,
                    tr,
                    y_tr,
                    sub,
                    y_te[mask],
                    pooling,
                    cfg.probe.n_bootstrap,
                    cfg.probe.seed,
                ).to_dict()
        return out

    def full(model_id: str, layer: int) -> dict:
        m = recipe_transfer(
            cache,
            model_id,
            layer,
            tr,
            y_tr,
            te,
            y_te,
            pooling,
            cfg.probe.n_bootstrap,
            cfg.probe.seed,
        ).to_dict()
        m["by_pair"] = pairs(model_id, layer)
        return m

    out = {
        "src_model": src.id,
        "src_key": src_key,
        "dst_model": dst.id,
        "dst_key": dst_key,
        "dataset": Path(items_path).stem,
        "tag": tag,
        "pooling": pooling,
        "n_items": len(items),
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "src_layer": src_layer,
        "src_n_layers": src_n,
        "dst_n_layers": dst_n,
        "dst_matched_layer": dst_layer,
        "dst_best_layer": dst_best,
        "dst_per_layer_val_auroc": {str(k): float(v) for k, v in val_auc.items()},
        "src_at_best": full(src.id, src_layer),
        "dst_at_matched": full(dst.id, dst_layer),
        "dst_at_own_best": full(dst.id, dst_best),
        "cka_matched": representation_similarity(
            cache, src.id, dst.id, te, src_layer, dst_layer, pooling
        ),
        "cka_best": representation_similarity(
            cache, src.id, dst.id, te, src_layer, dst_best, pooling
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"transfer_{src_key}_{dst_key}_{pooling}{_tag_suffix(items_path, tag)}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


# ---- steering -------------------------------------------------------------------------


class AsksVerdict(BaseModel):
    asks_clarifying_question: bool = Field(
        description="true if the reply's main move is to ask the user for a missing or conflicting detail before doing the task"
    )


def run_steer(
    model_key: str,
    items_path: Path,
    out_dir: Path = Path("results"),
    pooling: str = "last",
    tag: str = "",
    limit: int = 32,
    alphas: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    max_new_tokens: int = 160,
) -> Path:
    """Push the residual stream along the mass-mean gap direction and count asks.

    ``alphas`` are in units of the training set's spread along the direction,
    so +1 moves a state one standard deviation toward "underspecified". If the
    asking rate rises with alpha on specified requests and falls with negative
    alpha on underspecified ones, the direction is causally relevant and not
    just a correlate. A judge decides whether each steered reply asks.
    """
    from halluscope.judge.client import JudgeClient
    from halluscope.models.loader import load_model

    cfg = get_settings()
    spec = cfg.model_spec(model_key)
    cache = ActivationCache(cfg.resolved_cache_dir() / "activations")
    with open(
        probe_result_path(out_dir, model_key, "gap", pooling, items_path), encoding="utf-8"
    ) as fh:
        res = json.load(fh)
    layer = int(res["probes"]["linear"]["best_layer"])
    items = _experiment_items(cfg, [spec.id], items_path, bool(res.get("exclude_seeds")))
    sp = grouped_split(items, seed=cfg.split.seed, fractions=cfg.split.fractions)
    tr, te = sp["train"], sp["test"]
    t_idx = {i.id: i.n_user_turns for i in items}
    shas = shas_for(items)
    Xtr = cache.matrix(spec.id, [i.id for i in tr], t_idx, layer, pooling, shas=shas)
    mm = MassMeanProbe(seed=cfg.probe.seed).fit(Xtr, _labels(tr))
    direction = mm.raw_direction()
    scale = float(np.std(Xtr @ direction))

    # Half specified, half gap, in split order so the subset is reproducible.
    spec_items = [i for i in te if i.label == "specified"][: limit // 2]
    gap_items = [i for i in te if i.label != "specified"][: limit - len(spec_items)]
    chosen = spec_items + gap_items

    client = JudgeClient(cfg.judge, cost_log=cfg.paths.cost_log)
    loaded = load_model(spec)
    rows = []
    for a in alphas:
        for it in track(chosen, description=f"steer alpha={a:+.1f}"):
            t0 = time.time()
            text = steered_generate(loaded, it.turns, direction, layer, a * scale, max_new_tokens)
            verdict = client.complete(
                "judge_primary",
                [
                    {
                        "role": "system",
                        "content": "You label whether an assistant reply asks a clarifying question.",
                    },
                    {
                        "role": "user",
                        "content": "Request:\n"
                        + "\n".join(f"{t.role}: {t.content}" for t in it.turns)
                        + f"\n\nReply:\n{text}\n\nDoes the reply ask the user for a missing or conflicting detail before doing the task?",
                    },
                ],
                AsksVerdict,
                tag=f"steer:{it.id}:{a:+.1f}",
            )
            rows.append(
                {
                    "item_id": it.id,
                    "label": it.label,
                    "variant": it.variant,
                    "alpha": a,
                    "asks": verdict.asks_clarifying_question,
                    "text": text,
                    "seconds": round(time.time() - t0, 2),
                }
            )
    curve: dict[str, dict[str, float]] = {}
    for label in ("specified", "underspecified", "inconsistent", "all"):
        curve[label] = {}
        for a in alphas:
            sel = [r for r in rows if r["alpha"] == a and (label == "all" or r["label"] == label)]
            if sel:
                curve[label][f"{a:+.1f}"] = float(np.mean([r["asks"] for r in sel]))
    out = {
        "model": spec.id,
        "model_key": model_key,
        "dataset": Path(items_path).stem,
        "tag": tag,
        "pooling": pooling,
        "layer": layer,
        "direction": "mass-mean, train split, standardized then mapped back to model coordinates",
        "alpha_unit": "standard deviations of the train set along the direction",
        "scale": scale,
        "alphas": list(alphas),
        "n_items": len(chosen),
        "n_specified": len(spec_items),
        "n_gap": len(gap_items),
        "asking_rate": curve,
        "rows": rows,
        "cost": client.cost_summary(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"steer_{model_key}_{pooling}{_tag_suffix(items_path, tag)}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path
