import json
from types import SimpleNamespace

from halluscope.config import AliasCfg, JudgeCfg
from halluscope.data.expand import FamilySpec, expand
from halluscope.data.io import load_items, save_items
from halluscope.data.verify import FamilyVerdict, passes, run_verify
from halluscope.judge.client import JudgeClient


def _resp(content):
    r = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    r._hidden_params = {"response_cost": 0.0}
    return r


def _fam(name):
    spec = FamilySpec(
        family=name,
        topic="rag",
        setup="s",
        specified="spec",
        underspecified="under",
        gap="g",
        question="q?",
        assumption="a",
        fill="fill",
        final="final",
        contradiction="contra",
        update="upd",
    )
    items = expand(spec).items
    for it in items:
        it.source = "augmented"
        it.review_status = "pending"
    return items


def test_passes_requires_all_checks():
    good = FamilyVerdict(
        b_omits_gap=True,
        b_is_answerable_without_guessing=False,
        c_is_consistent=True,
        d_contradicts=True,
        off_domain=False,
        notes="",
    )
    assert passes(good)
    assert not passes(good.model_copy(update={"d_contradicts": False}))
    assert not passes(good.model_copy(update={"off_domain": True}))


def test_run_verify_marks_failing_family_rejected(tmp_path):
    items = _fam("rag-0001-p1") + _fam("rag-0002-p1")
    path = tmp_path / "items.jsonl"
    save_items(items, path)

    def fake(**kw):
        verdict = {
            "b_omits_gap": True,
            "b_is_answerable_without_guessing": False,
            "c_is_consistent": True,
            "d_contradicts": True,
            "off_domain": False,
            "notes": "",
        }
        return _resp(json.dumps(verdict))

    cfg = JudgeCfg(aliases={"judge_primary": AliasCfg(models=["fake/j"])}, max_retries=1)
    client = JudgeClient(cfg, cost_log=tmp_path / "c.jsonl", completion_fn=fake)
    r = run_verify(path, client=client)
    assert r["families_checked"] == 2
    reloaded = load_items(path)
    assert all(it.review_status == "pending" for it in reloaded)  # everything passed


def test_run_verify_rejects_when_judge_says_no_contradiction(tmp_path):
    items = _fam("rag-0003-p1")
    path = tmp_path / "items.jsonl"
    save_items(items, path)

    def fake(**kw):
        return _resp(
            json.dumps(
                {
                    "b_omits_gap": True,
                    "b_is_answerable_without_guessing": False,
                    "c_is_consistent": True,
                    "d_contradicts": False,
                    "off_domain": False,
                    "notes": "d does not conflict",
                }
            )
        )

    cfg = JudgeCfg(aliases={"judge_primary": AliasCfg(models=["fake/j"])}, max_retries=1)
    client = JudgeClient(cfg, cost_log=tmp_path / "c.jsonl", completion_fn=fake)
    r = run_verify(path, client=client)
    assert r["families_rejected"] == 1
    assert all(it.review_status == "rejected" for it in load_items(path))
