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


def _asking_curve(rows: list[dict], alphas) -> dict[str, dict[str, float]]:
    curve: dict[str, dict[str, float]] = {}
    for label in ("specified", "underspecified", "inconsistent", "all"):
        curve[label] = {}
        for a in alphas:
            sel = [r for r in rows if r["alpha"] == a and (label == "all" or r["label"] == label)]
            if sel:
                curve[label][f"{a:+.1f}"] = float(np.mean([r["asks"] for r in sel]))
    return curve


def _reply_length(rows: list[dict], alphas) -> dict[str, float]:
    """Mean reply length per alpha, as a cheap check that the push left the model fluent.

    A push large enough to break generation would drive the asking rate to zero
    for an uninteresting reason, so the length has to be reported next to the rate.
    """
    out = {}
    for a in alphas:
        sel = [len(r["text"].strip()) for r in rows if r["alpha"] == a]
        if sel:
            out[f"{a:+.1f}"] = float(np.mean(sel))
    return out


def random_unit_directions(dim: int, n: int, seed: int) -> list[np.ndarray]:
    """``n`` unit vectors drawn uniformly on the sphere in ``dim`` dimensions.

    In high dimensions two such vectors are almost orthogonal to each other and
    to any fixed direction, so these are a fair "same push, no meaning" control.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = rng.standard_normal(dim)
        out.append(v / (np.linalg.norm(v) + 1e-12))
    return out


def run_steer(
    model_key: str,
    items_path: Path,
    out_dir: Path = Path("results"),
    pooling: str = "last",
    tag: str = "",
    limit: int = 32,
    alphas: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    max_new_tokens: int = 160,
    random_controls: int = 0,
    control_seed: int | None = None,
    resume: bool = False,
) -> Path:
    """Push the residual stream along the mass-mean gap direction and count asks.

    ``alphas`` are in units of the training set's spread along the direction,
    so +1 moves a state one standard deviation toward "underspecified". If the
    asking rate rises with alpha on specified requests and falls with negative
    alpha on underspecified ones, the direction is causally relevant and not
    just a correlate. A judge decides whether each steered reply asks.

    ``random_controls`` repeats the whole sweep along that many random unit
    directions at the identical push norm. Without it, a flat curve has two
    readings that cannot be told apart: the direction carries no causal weight,
    or a push of this size does nothing at all whatever its direction. The
    control separates them, and it is what the 2026 probing critiques ask for.
    Alpha zero is the unsteered model, so it is generated once and shared.
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

    c_seed = cfg.probe.seed if control_seed is None else control_seed
    controls = random_unit_directions(direction.shape[0], random_controls, c_seed)
    # The probe direction is unit norm and ``steer`` normalizes again, so every
    # direction here adds a vector of exactly the same length at the same layer.
    plan: list[tuple[str, np.ndarray, tuple[float, ...]]] = [("probe", direction, tuple(alphas))]
    nonzero = tuple(a for a in alphas if a != 0.0)
    for k, v in enumerate(controls):
        plan.append((f"random_{k}", v, nonzero))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"steer_{model_key}_{pooling}{_tag_suffix(items_path, tag)}.json"

    partial = path.with_name(path.stem + ".partial.json")
    rows: list[dict] = []
    if resume:
        seen: set[tuple] = set()
        for src_path in (path, partial):
            if not src_path.exists():
                continue
            for r in json.loads(src_path.read_text(encoding="utf-8")).get("rows", []):
                r.setdefault("direction", "probe")
                key = (r["direction"], r["item_id"], float(r["alpha"]))
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    done = {(r["direction"], r["item_id"], float(r["alpha"])) for r in rows}
    todo = sum(
        1
        for name, _, als in plan
        for a in als
        for it in chosen
        if (name, it.id, float(a)) not in done
    )
    if todo == 0:
        client = None
        loaded = None
    else:
        client = JudgeClient(cfg.judge, cost_log=cfg.paths.cost_log)
        loaded = load_model(spec)

    for name, vec, als in plan:
        for a in als:
            pending = [it for it in chosen if (name, it.id, float(a)) not in done]
            if not pending:
                continue
            for it in track(pending, description=f"{name} alpha={a:+.1f}"):
                t0 = time.time()
                text = steered_generate(loaded, it.turns, vec, layer, a * scale, max_new_tokens)
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
                    tag=f"steer:{name}:{it.id}:{a:+.1f}",
                )
                rows.append(
                    {
                        "direction": name,
                        "item_id": it.id,
                        "label": it.label,
                        "variant": it.variant,
                        "alpha": a,
                        "asks": verdict.asks_clarifying_question,
                        "text": text,
                        "seconds": round(time.time() - t0, 2),
                    }
                )
            # Written to a sidecar after every alpha so a long control run survives an
            # interruption without clobbering the finished result it resumed from.
            partial.write_text(
                json.dumps({"partial": True, "rows": rows}, indent=2), encoding="utf-8"
            )

    probe_rows = [r for r in rows if r["direction"] == "probe"]
    control_rows = [r for r in rows if r["direction"] != "probe"]
    zero_rows = [r for r in probe_rows if r["alpha"] == 0.0]
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
        "asking_rate": _asking_curve(probe_rows, alphas),
        "reply_chars": _reply_length(probe_rows, alphas),
        "rows": rows,
        "cost": client.cost_summary() if client else {},
    }
    if control_rows:
        names = sorted({r["direction"] for r in control_rows})
        # Alpha zero is the unsteered model, identical for every direction, so the
        # controls borrow the probe run's rows rather than regenerating them.
        per_direction = {
            n: _asking_curve([r for r in control_rows if r["direction"] == n] + zero_rows, alphas)
            for n in names
        }
        pooled = _asking_curve(control_rows + zero_rows, alphas)
        out["random_controls"] = {
            "n": len(names),
            "seed": c_seed,
            "note": (
                "same layer, same items, same push norm, directions drawn uniformly on the unit "
                "sphere; alpha zero is shared with the probe direction because no push is applied"
            ),
            "cosine_with_probe": {
                f"random_{k}": float(np.dot(v, direction)) for k, v in enumerate(controls)
            },
            "per_direction": per_direction,
            "pooled": pooled,
            "pooled_reply_chars": _reply_length(control_rows + zero_rows, alphas),
            "probe_minus_random": _steer_comparison(probe_rows, control_rows, nonzero),
        }
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)
    return path


def _steer_comparison(probe_rows: list[dict], control_rows: list[dict], alphas) -> dict:
    """Paired difference in asking rate between the probe direction and the controls.

    Items are the same on both sides, so the bootstrap resamples items and
    recomputes both rates on the resampled set rather than treating the two
    curves as independent.
    """
    from halluscope.eval.metrics import paired_bootstrap_delta

    mean = lambda _y, s: float(np.mean(s))  # noqa: E731
    out = {}
    for a in alphas:
        by_item = {r["item_id"]: r for r in probe_rows if r["alpha"] == a}
        ctrl: dict[str, list[float]] = {}
        for r in control_rows:
            if r["alpha"] == a:
                ctrl.setdefault(r["item_id"], []).append(float(r["asks"]))
        ids = [i for i in by_item if i in ctrl]
        if not ids:
            continue
        pa = np.array([float(by_item[i]["asks"]) for i in ids])
        ca = np.array([float(np.mean(ctrl[i])) for i in ids])
        d = paired_bootstrap_delta(mean, np.zeros(len(ids)), pa, ca, n=1000)
        d["probe_rate"] = float(pa.mean())
        d["random_rate"] = float(ca.mean())
        out[f"{a:+.1f}"] = d
    return out
