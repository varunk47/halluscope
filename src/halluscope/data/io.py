"""Read and write items.

Seed files are compact family specs (see ``expand.py``); everything else is
JSONL of fully materialized items.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import orjson

from halluscope.data.expand import FamilySpec, expand
from halluscope.data.schema import Family, Item


def load_seed_specs(seeds_dir: Path | str) -> list[FamilySpec]:
    """Topic files hold the nine hand-written fields; ``_updates.json`` holds the
    compatible-update sentence per family (kept separate so it is easy to audit)."""
    seeds_dir = Path(seeds_dir)
    updates: dict[str, str] = {}
    upd_path = seeds_dir / "_updates.json"
    if upd_path.exists():
        with open(upd_path, encoding="utf-8") as fh:
            updates = json.load(fh)
    specs: list[FamilySpec] = []
    for path in sorted(seeds_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        for raw in payload:
            if "update" not in raw:
                raw = {**raw, "update": updates.get(raw["family"], "")}
            specs.append(FamilySpec.model_validate(raw))
    return specs


def load_seed_families(seeds_dir: Path | str) -> list[Family]:
    return [expand(s) for s in load_seed_specs(seeds_dir)]


def load_items(path: Path | str) -> list[Item]:
    """Load items from a seeds directory or a JSONL file of items."""
    path = Path(path)
    if path.is_dir():
        return [it for fam in load_seed_families(path) for it in fam.items]
    if path.suffix != ".jsonl":
        raise ValueError(f"expected a seeds directory or a .jsonl file, got {path}")
    items: list[Item] = []
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(Item.model_validate(orjson.loads(line)))
    return items


def save_items(items: Iterable[Item], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "wb") as fh:
        for it in items:
            fh.write(orjson.dumps(it.model_dump()))
            fh.write(b"\n")
            n += 1
    return n


def approved(items: Iterable[Item], require_human: bool | None = None) -> list[Item]:
    """Items usable for experiments.

    With ``require_human`` (default from ``settings.review.require_human_approval``)
    only human-approved or human-edited items pass. Otherwise anything not
    rejected passes, and the results record how many were human-approved.
    """
    if require_human is None:
        from halluscope.config import get_settings

        require_human = get_settings().review.require_human_approval
    if require_human:
        return [it for it in items if it.review_status in ("approved", "edited")]
    return [it for it in items if it.review_status != "rejected"]


def review_counts(items: Iterable[Item]) -> dict[str, int]:
    out = {"approved": 0, "edited": 0, "pending": 0, "rejected": 0}
    for it in items:
        out[it.review_status] = out.get(it.review_status, 0) + 1
    return out
