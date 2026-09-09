"""Dataset browsing and review."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from halluscope.config import get_settings
from halluscope.data.io import load_items, save_items
from halluscope.data.schema import Item, Turn

router = APIRouter(tags=["items"])
_lock = Lock()


def items_path() -> Path:
    cfg = get_settings()
    p = cfg.paths.data_dir / "augmented" / "items.jsonl"
    return p if p.exists() else cfg.paths.seeds_dir


def _load() -> list[Item]:
    return load_items(items_path())


@router.get("/items")
def list_items(
    status: str | None = Query(None, description="pending | approved | edited | rejected"),
    topic: str | None = None,
    label: str | None = None,
    source: str | None = None,
    q: str | None = Query(None, description="substring search over turns"),
    limit: int = 200,
    offset: int = 0,
) -> dict:
    items = _load()
    if status:
        items = [i for i in items if i.review_status == status]
    if topic:
        items = [i for i in items if i.topic == topic]
    if label:
        items = [i for i in items if i.label == label]
    if source:
        items = [i for i in items if i.source == source]
    if q:
        ql = q.lower()
        items = [i for i in items if any(ql in t.content.lower() for t in i.turns)]
    total = len(items)
    counts = {
        "pending": sum(i.review_status == "pending" for i in _load()),
    }
    return {
        "total": total,
        "counts": counts,
        "items": [i.model_dump() for i in items[offset : offset + limit]],
    }


@router.get("/items/{item_id}")
def get_item(item_id: str) -> dict:
    for i in _load():
        if i.id == item_id:
            return i.model_dump()
    raise HTTPException(404, f"no item {item_id}")


class ReviewPatch(BaseModel):
    review_status: str | None = None
    reviewed_by: str | None = None
    turns: list[Turn] | None = None
    gap: str | None = None
    expected_clarifying_question: str | None = None
    plausible_silent_assumption: str | None = None


@router.patch("/items/{item_id}")
def patch_item(item_id: str, patch: ReviewPatch) -> dict:
    path = items_path()
    if path.is_dir():
        raise HTTPException(
            400,
            "seeds are read-only; run `halluscope build-data` to create data/augmented/items.jsonl",
        )
    with _lock:
        items = load_items(path)
        for idx, it in enumerate(items):
            if it.id != item_id:
                continue
            data = it.model_dump()
            for k, v in patch.model_dump(exclude_none=True).items():
                data[k] = v
            if patch.turns is not None and patch.review_status is None:
                data["review_status"] = "edited"
            new = Item.model_validate(data)
            items[idx] = new
            save_items(items, path)
            return new.model_dump()
    raise HTTPException(404, f"no item {item_id}")
