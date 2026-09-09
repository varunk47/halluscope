"""Score every item in a split with every uncertainty baseline, on the same
items the probe is evaluated on, with per-method cost.

Output: ``results/uq_{model}_{split}.json`` with per-item scores per method
and a summary table of AUROC, AUPRC, generations required, and seconds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from rich.progress import track

from halluscope.capture.cache import ActivationCache, CaptureKey, shas_for
from halluscope.config import get_settings
from halluscope.data.io import approved, load_items
from halluscope.data.schema import Item
from halluscope.data.splits import grouped_split
from halluscope.eval.metrics import evaluate_scores
from halluscope.judge.client import JudgeClient
from halluscope.models.chat import generate
from halluscope.models.loader import LoadedModel, load_model
from halluscope.probes.run import gap_label
from halluscope.uq.eigenscore import eigenscore
from halluscope.uq.entropy import logit_lens_entropy, predictive_entropy
from halluscope.uq.prompted import p_true, verbalized_confidence
from halluscope.uq.semantic_entropy import make_llm_equivalence, semantic_entropy
from halluscope.uq.sep import SEProbe


def _sample_embeddings(loaded: LoadedModel, item: Item, texts: list[str], layer: int) -> np.ndarray:
    """Mid-layer last-token embedding of each sampled answer appended to the dialogue."""
    from halluscope.data.schema import Turn

    rows = []
    for t in texts:
        turns = list(item.turns) + [Turn(role="assistant", content=t or " ")]
        # capture_prefix wants to end on a user turn; render manually instead
        cap = _capture_any(loaded, turns)
        rows.append(cap[layer])
    return np.stack(rows)


def _capture_any(loaded: LoadedModel, turns) -> np.ndarray:
    import torch

    from halluscope.models.chat import render_chat

    tok = loaded.tokenizer
    prompt = render_chat(tok, turns, add_generation_prompt=False)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(loaded.device)
    with torch.no_grad():
        out = loaded.model(**enc, output_hidden_states=True, use_cache=False)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu().numpy()


def score_item(
    loaded: LoadedModel,
    item: Item,
    client: JudgeClient | None,
    methods: list[str],
    K: int,
    seed: int,
    max_new_tokens: int,
    mid_layer: int,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    samples = None
    greedy = None

    def need_samples():
        nonlocal samples
        if samples is None:
            t0 = time.time()
            samples = generate(
                loaded, item.turns, max_new_tokens=max_new_tokens, temperature=0.7, n=K, seed=seed
            )
            out["_sampling_seconds"] = {"value": time.time() - t0}
        return samples

    def need_greedy():
        nonlocal greedy
        if greedy is None:
            t0 = time.time()
            greedy = generate(
                loaded, item.turns, max_new_tokens=max_new_tokens, temperature=0.0, n=1
            )[0]
            out["_greedy_seconds"] = {"value": time.time() - t0}
        return greedy

    if "predictive_entropy" in methods:
        g = need_greedy()
        out["predictive_entropy"] = {
            "value": predictive_entropy(g),
            "n_generations": 1,
            "seconds": out["_greedy_seconds"]["value"],
        }

    if "logit_lens_entropy" in methods:
        t0 = time.time()
        ents = logit_lens_entropy(loaded, item.turns)
        out["logit_lens_entropy"] = {
            "value": float(np.nanmean(ents[-8:])),
            "n_generations": 0,
            "seconds": time.time() - t0,
            "per_layer": [float(x) for x in ents],
        }

    if "eigenscore" in methods:
        s = need_samples()
        t0 = time.time()
        emb = _sample_embeddings(loaded, item, [g.text for g in s], mid_layer)
        out["eigenscore"] = {
            "value": eigenscore(emb),
            "n_generations": K,
            "seconds": out["_sampling_seconds"]["value"] + time.time() - t0,
        }

    if "semantic_entropy" in methods and client is not None:
        s = need_samples()
        t0 = time.time()
        ctx = "\n".join(f"{t.role}: {t.content}" for t in item.turns)
        eq = make_llm_equivalence(
            client,
            "judge_secondary" if "judge_secondary" in client.cfg.aliases else "judge_primary",
            ctx,
        )
        out["semantic_entropy"] = {
            "value": semantic_entropy([g.text for g in s], eq),
            "n_generations": K,
            "seconds": out["_sampling_seconds"]["value"] + time.time() - t0,
        }

    if "ptrue" in methods:
        g = need_greedy()
        t0 = time.time()
        out["ptrue"] = {
            "value": p_true(loaded, item.turns, g.text),
            "n_generations": 1,
            "seconds": out["_greedy_seconds"]["value"] + time.time() - t0,
        }

    if "verbalized" in methods:
        t0 = time.time()
        out["verbalized"] = {
            "value": verbalized_confidence(loaded, item.turns, seed=seed),
            "n_generations": 1,
            "seconds": time.time() - t0,
        }

    out["_samples"] = {
        "texts": [g.text for g in (samples or [])],
        "greedy": greedy.text if greedy else None,
    }
    return out


def run_uq(
    model_key: str,
    items_path: Path,
    split: str = "test",
    out_dir: Path = Path("results"),
    methods: list[str] | None = None,
    limit: int | None = None,
    pooling: str = "last",
) -> Path:
    cfg = get_settings()
    spec = cfg.model_spec(model_key)
    methods = methods or cfg.uq.methods
    items_all = approved(load_items(items_path))
    splits = grouped_split(items_all, seed=cfg.split.seed, fractions=cfg.split.fractions)
    items = splits[split]
    if limit:
        items = items[:limit]
    client = (
        JudgeClient(cfg.judge, cost_log=cfg.paths.cost_log)
        if "semantic_entropy" in methods
        else None
    )
    loaded = load_model(spec)
    mid = loaded.n_layers // 2

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"uq_{model_key}_{split}.json"
    rows: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            rows = {r["item_id"]: r for r in json.load(fh)["rows"]}

    sampling_methods = {"semantic_entropy", "eigenscore"}
    todo = [i for i in items if i.id not in rows]
    for idx, it in enumerate(track(todo, description=f"uq {model_key} {split}")):
        # K-sample methods are slow on a laptop GPU: run them on the first sampling_limit items
        m = (
            methods
            if idx < cfg.uq.sampling_limit
            else [x for x in methods if x not in sampling_methods]
        )
        scores = score_item(
            loaded, it, client, m, cfg.uq.K, cfg.uq.seed, cfg.uq.gen.max_new_tokens, mid
        )
        rows[it.id] = {
            "item_id": it.id,
            "label": it.label,
            "topic": it.topic,
            "variant": it.variant,
            "y": gap_label(it),
            "scores": scores,
        }
        _write(path, spec.id, split, rows, methods, cfg.probe.n_bootstrap)

    # SEP: train a ridge probe from hidden states to semantic entropy on train split, score test split
    if "sep" in methods and "semantic_entropy" in methods:
        _add_sep(cfg, spec.id, model_key, items_all, splits, rows, path, split, methods, pooling)
    _write(path, spec.id, split, rows, methods, cfg.probe.n_bootstrap)
    return path


def _add_sep(cfg, model_id, model_key, items_all, splits, rows, path, split, methods, pooling):
    """SEP needs semantic entropy on the train split; compute it if a train uq file exists."""
    train_path = path.parent / f"uq_{model_key}_train.json"
    if not train_path.exists():
        return
    with open(train_path, encoding="utf-8") as fh:
        train_rows = {r["item_id"]: r for r in json.load(fh)["rows"]}
    cache = ActivationCache(cfg.resolved_cache_dir() / "activations")
    t_idx = {i.id: i.n_user_turns for i in items_all}
    shas = shas_for(items_all)
    tr_ids = [i for i, r in train_rows.items() if "semantic_entropy" in r["scores"] and i in shas]
    if len(tr_ids) < 20:
        return
    layer = cache.n_layers(model_id) * 2 // 3
    Xtr = cache.matrix(model_id, tr_ids, t_idx, layer, pooling, shas=shas)
    se = np.array([train_rows[i]["scores"]["semantic_entropy"]["value"] for i in tr_ids])
    probe = SEProbe().fit(Xtr, se)
    te_ids = [
        i for i in rows if i in shas and cache.has(CaptureKey(model_id, i, t_idx[i], shas[i]))
    ]
    Xte = cache.matrix(model_id, te_ids, t_idx, layer, pooling, shas=shas)
    pred = probe.predict(Xte)
    for iid, v in zip(te_ids, pred, strict=True):
        rows[iid]["scores"]["sep"] = {
            "value": float(v),
            "n_generations": 0,
            "seconds": 0.0,
            "layer": layer,
        }


def _write(
    path: Path, model_id: str, split: str, rows: dict[str, dict], methods: list[str], n_boot: int
) -> None:
    summary = {}
    ys = np.array([r["y"] for r in rows.values()])
    for m in methods + ["sep"]:
        vals = [r["scores"].get(m, {}).get("value", np.nan) for r in rows.values()]
        v = np.array(vals, dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() < 5 or len(np.unique(ys[ok])) < 2:
            continue
        met = evaluate_scores(ys[ok], v[ok], n_bootstrap=n_boot)
        secs = [r["scores"].get(m, {}).get("seconds", 0.0) for r in rows.values()]
        gens = [r["scores"].get(m, {}).get("n_generations", 0) for r in rows.values()]
        summary[m] = {
            **met.to_dict(),
            "mean_seconds": float(np.mean(secs)),
            "n_generations": int(np.max(gens)) if gens else 0,
        }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"model": model_id, "split": split, "summary": summary, "rows": list(rows.values())},
            fh,
            indent=2,
        )
