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
