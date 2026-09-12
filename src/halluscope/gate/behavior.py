"""Behavioral labels: what does the model actually do when it answers?

For every item, generate the model's own reply at temperature 0 and have a
cross-family judge decide whether it asked, flagged, or silently assumed.
These labels become probe targets (``will_assume``, ``will_ask``) and the
measurement of the recognition-action gap.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.progress import track

from halluscope.config import get_settings
from halluscope.data.io import approved, load_items
from halluscope.data.schema import Item
from halluscope.judge.client import JudgeClient
from halluscope.judge.rubrics import AssumptionVerdict, assumption_prompt
from halluscope.models.chat import generate
from halluscope.models.loader import LoadedModel, load_model


def answer_item(loaded: LoadedModel, item: Item, max_new_tokens: int = 256) -> str:
    return generate(loaded, item.turns, max_new_tokens=max_new_tokens, temperature=0.0, n=1)[0].text


def judge_answer(
    client: JudgeClient, item: Item, answer: str, alias: str = "judge_primary"
) -> AssumptionVerdict:
    return client.complete(
        alias, assumption_prompt(item, answer), AssumptionVerdict, tag=f"behavior:{item.id}"
    )


def behavior_filename(model_key: str, items_path: Path, tag: str = "") -> str:
    """``behavior_{model}_{build}.json``; the untagged name is the default build."""
    t = tag or Path(items_path).stem.removeprefix("items").lstrip("_")
    return f"behavior_{model_key}_{t}.json" if t else f"behavior_{model_key}.json"


def run_behavior(
    model_key: str,
    items_path: Path,
    out_dir: Path = Path("results"),
    client: JudgeClient | None = None,
    second_judge: bool = True,
    limit: int | None = None,
    batch_size: int = 8,
    tag: str = "",
) -> Path:
    cfg = get_settings()
    spec = cfg.model_spec(model_key)
    items = approved(load_items(items_path))
    if limit:
        items = items[:limit]
    client = client or JudgeClient(cfg.judge, cost_log=cfg.paths.cost_log)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / behavior_filename(model_key, items_path, tag)

    done: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            done = {r["item_id"]: r for r in json.load(fh)["rows"]}

    todo = [i for i in items if i.id not in done]
    loaded = load_model(spec) if todo else None
    from halluscope.models.chat import generate_batch

    answers: dict[str, str] = {}
    for start in track(range(0, len(todo), batch_size), description=f"generate {model_key}"):
        chunk = todo[start : start + batch_size]
        gens = generate_batch(
            loaded,
            [i.turns for i in chunk],
            max_new_tokens=cfg.uq.gen.max_new_tokens,
            temperature=0.0,
            batch_size=batch_size,
        )
        for it, g in zip(chunk, gens, strict=True):
            answers[it.id] = g.text
    for it in track(todo, description=f"judge {model_key}"):
        answer = answers[it.id]
        v1 = judge_answer(client, it, answer, "judge_primary")
        row = {
            "item_id": it.id,
            "label": it.label,
            "topic": it.topic,
            "variant": it.variant,
            "answer": answer,
            "asked": v1.asked,
            "flagged": v1.flagged,
            "silently_assumed": v1.silently_assumed,
            "assumed_value": v1.assumed_value,
            "rationale": v1.rationale,
        }
        if second_judge and "judge_secondary" in cfg.judge.aliases:
            try:
                second = judge_answer(client, it, answer, "judge_secondary")
                row["judge2"] = {
                    "asked": second.asked,
                    "flagged": second.flagged,
                    "silently_assumed": second.silently_assumed,
                }
            except Exception as e:  # noqa: BLE001
                row["judge2_error"] = str(e)[:200]
        done[it.id] = row
        _write(path, spec.id, list(done.values()), client)
    _write(path, spec.id, list(done.values()), client)
    return path


def _write(path: Path, model_id: str, rows: list[dict], client: JudgeClient) -> None:
    summary = summarize(rows)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"model": model_id, "summary": summary, "cost": client.cost_summary(), "rows": rows},
            fh,
            indent=2,
        )


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for label in ("specified", "underspecified", "inconsistent"):
        sub = [r for r in rows if r["label"] == label]
        if not sub:
            continue
        n = len(sub)
        out[label] = {
            "n": n,
            "asked": sum(r["asked"] for r in sub) / n,
            "flagged": sum(r["flagged"] for r in sub) / n,
            "silently_assumed": sum(r["silently_assumed"] for r in sub) / n,
        }
    with_j2 = [r for r in rows if "judge2" in r]
    if with_j2:
        from halluscope.judge.agreement import cohens_kappa

        a = [r["silently_assumed"] for r in with_j2]
        b = [r["judge2"]["silently_assumed"] for r in with_j2]
        out["judge_agreement"] = {"n": len(with_j2), "kappa_silently_assumed": cohens_kappa(a, b)}
    return out
