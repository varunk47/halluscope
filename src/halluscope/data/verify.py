"""LLM verification pass over augmented families.

Augmentation can drift: the underspecified variant may still contain the gap
detail, the "compatible update" may actually conflict, or the contradiction
may not contradict. A judge from a different model than the augmenter checks
each family against its own annotations and families that fail are marked
``rejected`` so they never enter an experiment. Humans review on top of this.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from halluscope.data.io import load_items, save_items
from halluscope.data.schema import Item
from halluscope.judge.client import JudgeClient, JudgeError

log = logging.getLogger(__name__)


class FamilyVerdict(BaseModel):
    b_omits_gap: bool = Field(description="Variant b does not state the detail named in gap")
    b_is_answerable_without_guessing: bool = Field(
        description="Variant b could be completed without guessing the gap; should be false for a good item"
    )
    c_is_consistent: bool = Field(
        description="In variant c the final turn does not conflict with the earlier turn"
    )
    d_contradicts: bool = Field(
        description="In variant d the final turn conflicts with the earlier turn"
    )
    off_domain: bool = Field(
        description="Any variant mentions physics, mechanics, materials, or simulation"
    )
    notes: str


def _dlg(item: Item) -> str:
    return "\n".join(f"{t.role.upper()}: {t.content}" for t in item.turns)


def verify_prompt(fam: dict[str, Item]) -> list[dict[str, str]]:
    b = fam["b"]
    return [
        {
            "role": "system",
            "content": (
                "You audit a dataset of AI-engineering requests. Each family has four dialogues. "
                "Check the annotations strictly and answer with booleans."
            ),
        },
        {
            "role": "user",
            "content": (
                f"GAP annotation: {b.gap}\n\n"
                f"Variant a (specified):\n{_dlg(fam['a'])}\n\n"
                f"Variant b (should be underspecified, missing exactly the gap):\n{_dlg(b)}\n\n"
                f"Variant c (multi-turn, should be fully consistent):\n{_dlg(fam['c'])}\n\n"
                f"Variant d (multi-turn, final turn should contradict the earlier turn):\n{_dlg(fam['d'])}\n"
            ),
        },
    ]


def verify_families(
    items: list[Item], client: JudgeClient, alias: str = "judge_primary", workers: int = 4
) -> dict[str, FamilyVerdict | None]:
    fams: dict[str, dict[str, Item]] = defaultdict(dict)
    for it in items:
        if it.source == "augmented":
            fams[it.family][it.variant] = it
    complete = {f: v for f, v in fams.items() if set(v) == {"a", "b", "c", "d"}}

    def one(f: str):
        try:
            return f, client.complete(
                alias, verify_prompt(complete[f]), FamilyVerdict, tag=f"verify:{f}"
            )
        except JudgeError as e:
            log.warning("verify %s failed: %s", f, e)
            return f, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, sorted(complete)))


def passes(v: FamilyVerdict) -> bool:
    """Structural checks only.

    ``b_is_answerable_without_guessing`` is recorded but not used: judges read it
    as "could one implement something reasonable by choosing values freely",
    which is true of every underspecified request and is the very behavior the
    dataset exists to detect.
    """
    return v.b_omits_gap and v.c_is_consistent and v.d_contradicts and not v.off_domain


def reapply(items_path: Path, report_path: Path) -> dict:
    """Recompute rejections from a saved report with the current ``passes`` rule.
    Families previously rejected by the LLM pass and now passing return to pending."""
    import json

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    verdicts = {
        f: (FamilyVerdict.model_validate(v) if v else None) for f, v in report["verdicts"].items()
    }
    rejected = {f for f, v in verdicts.items() if v is not None and not passes(v)}
    items = load_items(items_path)
    changed = 0
    for it in items:
        if it.family not in verdicts:
            continue
        should = "rejected" if it.family in rejected else "pending"
        if (
            it.reviewed_by in (None, "llm-verify")
            and it.review_status in ("pending", "rejected")
            and it.review_status != should
        ):
            it.review_status = should
            it.reviewed_by = "llm-verify" if should == "rejected" else None
            changed += 1
    save_items(items, items_path)
    report["families_rejected"] = len(rejected)
    report["rejected"] = sorted(rejected)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"families_rejected": len(rejected), "items_changed": changed}


def run_verify(
    items_path: Path, client: JudgeClient | None = None, out_report: Path | None = None
) -> dict:
    from halluscope.config import get_settings

    cfg = get_settings()
    client = client or JudgeClient(cfg.judge, cost_log=cfg.paths.cost_log)
    items = load_items(items_path)
    verdicts = verify_families(items, client)
    rejected = {f for f, v in verdicts.items() if v is not None and not passes(v)}
    for it in items:
        if it.family in rejected and it.review_status == "pending":
            it.review_status = "rejected"
            it.reviewed_by = "llm-verify"
    save_items(items, items_path)
    report = {
        "families_checked": len(verdicts),
        "families_rejected": len(rejected),
        "rejected": sorted(rejected),
        "verdicts": {f: (v.model_dump() if v else None) for f, v in verdicts.items()},
        "cost": client.cost_summary(),
    }
    if out_report:
        import json

        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
