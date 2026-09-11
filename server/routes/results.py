"""Results files, summarized and in full."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from halluscope.config import get_settings

router = APIRouter(tags=["results"])


def _dir() -> Path:
    return Path(get_settings().paths.results_dir)


def _summary(name: str, d: dict) -> dict:
    kind = name.split("_")[0]
    base = {"name": name, "kind": kind, "model": d.get("model")}
    if kind == "probe":
        base.update(
            {
                "dataset": d.get("tag") or d.get("dataset"),
                "headline": _headline(d),
                "target": d.get("target"),
                "pooling": d.get("pooling"),
                "n_layers": d.get("n_layers"),
                "split_sizes": d.get("split_sizes"),
                "probes": {
                    k: {"best_layer": v["best_layer"], "test": v["test"]}
                    for k, v in d.get("probes", {}).items()
                },
                "baselines": d.get("baselines", {}),
                "loto": d.get("loto", {}),
            }
        )
    elif kind in ("transfer", "steer"):
        base.update({k: v for k, v in d.items() if k != "rows"})
    elif kind in ("uq", "behavior", "loop"):
        base.update(
            {
                "summary": d.get("summary", {}),
                "split": d.get("split"),
                "condition": d.get("condition"),
                "seed": d.get("seed"),
                "cost": d.get("cost"),
            }
        )
    return base


def _headline(d: dict) -> list[dict]:
    """Probe against the best surface baseline on identical test items, per pair.

    Flattened here so the page shows the finding without re-deriving it from
    three nested dicts.
    """
    pvs = d.get("probe_vs_surface")
    if not pvs:
        return []
    surface = d.get("baselines", {}).get(pvs["baseline"], {}).get("by_pair", {})
    rows = []
    for probe, pairs in pvs.get("by_pair", {}).items():
        for pair, delta in pairs.items():
            rows.append(
                {
                    "probe": probe,
                    "pair": pair,
                    "probe_auroc": d["probes"][probe].get("by_pair", {}).get(pair, {}).get("auroc"),
                    "words_auroc": surface.get(pair, {}).get("auroc"),
                    "baseline": pvs["baseline"],
                    **delta,
                }
            )
    return rows


@router.get("/results")
def list_results() -> dict:
    out = []
    for p in sorted(_dir().glob("*.json")):
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except json.JSONDecodeError:
            continue
        out.append(_summary(p.stem, d))
    return {"results": out}


@router.get("/results/{name}")
def get_result(name: str) -> dict:
    p = _dir() / f"{name}.json"
    if not p.exists() or not p.resolve().is_relative_to(_dir().resolve()):
        raise HTTPException(404, f"no result {name}")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
