import json
from types import SimpleNamespace

import pytest

from halluscope.config import AliasCfg, JudgeCfg
from halluscope.judge.agreement import agreement_rate, cohens_kappa
from halluscope.judge.client import JudgeClient, JudgeError, extract_json
from halluscope.judge.pairwise import pairwise_swapped
from halluscope.judge.rubrics import AssumptionVerdict, PairwiseVerdict


def _resp(content, cost=0.001, pt=10, ct=5):
    r = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
    )
    r._hidden_params = {"response_cost": cost}
    return r


def _cfg():
    return JudgeCfg(
        aliases={
            "judge_primary": AliasCfg(models=["fake/a", "fake/b"]),
        },
        max_retries=2,
    )


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(JudgeError):
        extract_json("no json here")


def test_complete_repairs_fenced_output_and_logs_cost(tmp_path):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return _resp(
            '```json\n{"asked": false, "flagged": false, "silently_assumed": true, '
            '"assumed_value": "k=5", "rationale": "picked k"}\n```'
        )

    c = JudgeClient(_cfg(), cost_log=tmp_path / "cost.jsonl", completion_fn=fake)
    v = c.complete("judge_primary", [{"role": "user", "content": "judge"}], AssumptionVerdict)
    assert v.silently_assumed and v.behavior == "assumed" and v.assumed_value == "k=5"
    rows = [json.loads(line) for line in (tmp_path / "cost.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["ok"] and rows[0]["cost_usd"] == 0.001
    assert c.cost_summary()["calls"] == 1
    assert calls[0]["model"] == "fake/a"


def test_fallback_to_second_model_on_error(tmp_path):
    def fake(**kw):
        if kw["model"] == "fake/a":
            raise RuntimeError("rate limited")
        return _resp('{"winner": "A", "rationale": "x"}')

    c = JudgeClient(_cfg(), cost_log=tmp_path / "c.jsonl", completion_fn=fake)
    c.cfg.max_retries = 1
    v = c.complete("judge_primary", [{"role": "user", "content": "q"}], PairwiseVerdict)
    assert v.winner == "A"
    rows = [json.loads(line) for line in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert [r["ok"] for r in rows] == [False, True]


def test_complete_retries_on_invalid_shape_then_raises(tmp_path):
    def fake(**kw):
        return _resp('{"winner": "maybe", "rationale": "x"}')

    c = JudgeClient(_cfg(), cost_log=tmp_path / "c.jsonl", completion_fn=fake)
    with pytest.raises(JudgeError):
        c.complete("judge_primary", [{"role": "user", "content": "q"}], PairwiseVerdict)


def test_pairwise_swapped_detects_position_bias(tmp_path):
    def always_first(**kw):
        return _resp('{"winner": "A", "rationale": "first"}')

    c = JudgeClient(_cfg(), cost_log=tmp_path / "c.jsonl", completion_fn=always_first)
    build = lambda a, b: [{"role": "user", "content": f"{a} vs {b}"}]  # noqa: E731
    assert pairwise_swapped(c, "judge_primary", build, "x", "y") == "inconsistent"

    def prefers_y(**kw):
        content = kw["messages"][-1]["content"]
        winner = "A" if content.startswith("y") else "B"
        return _resp(json.dumps({"winner": winner, "rationale": "y"}))

    c2 = JudgeClient(_cfg(), cost_log=tmp_path / "c2.jsonl", completion_fn=prefers_y)
    assert pairwise_swapped(c2, "judge_primary", build, "x", "y") == "B"


def test_kappa():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9
    assert agreement_rate([1, 1, 0], [1, 0, 0]) == pytest.approx(2 / 3)
