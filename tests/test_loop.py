import json
from types import SimpleNamespace

from halluscope.config import AliasCfg, JudgeCfg
from halluscope.data.schema import Item, Turn
from halluscope.gate.loop import LoopRecord, reference_request, run_dialogue, summarize
from halluscope.judge.client import JudgeClient


class FakeLoaded:
    """Stands in for LoadedModel; generate() is monkeypatched to use it."""

    tokenizer = None


def _resp(content):
    r = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    r._hidden_params = {"response_cost": 0.0}
    return r


def _client(tmp_path):
    def fake(**kw):
        content = kw["messages"][-1]["content"]
        if "Your reply:" in content:
            return _resp("Use bge-m3 with cosine and k=8.")
        return _resp(json.dumps({"grade": "correct", "assumption_made": False, "rationale": "ok"}))

    cfg = JudgeCfg(
        aliases={
            "judge_primary": AliasCfg(models=["fake/j"]),
            "simulated_user": AliasCfg(models=["fake/u"]),
        },
        max_retries=1,
    )
    return JudgeClient(cfg, cost_log=tmp_path / "c.jsonl", completion_fn=fake)


def _items():
    a = Item(
        id="rag-1-a",
        family="rag-1",
        topic="rag",
        variant="a",
        label="specified",
        turns=[Turn(role="user", content="Chunk at 512, bge-m3, cosine, k=8.")],
    )
    b = Item(
        id="rag-1-b",
        family="rag-1",
        topic="rag",
        variant="b",
        label="underspecified",
        turns=[Turn(role="user", content="Chunk and embed the docs.")],
        gap="model, metric, k",
        expected_clarifying_question="Which model, metric, k?",
        reference_specified_variant="rag-1-a",
    )
    return a, b


def test_reference_request_uses_specified_variant():
    a, b = _items()
    by = {a.id: a, b.id: b}
    assert "bge-m3" in reference_request(b, by)
    assert reference_request(a, by).endswith("k=8.")


def test_gate_condition_asks_at_most_max_rounds(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_generate(loaded, turns, **kw):
        calls["n"] += 1
        text = (
            "Which embedding model?"
            if turns[-1].content.startswith("The request may")
            else "Final plan."
        )
        return [SimpleNamespace(text=text)]

    monkeypatch.setattr("halluscope.gate.loop.generate", fake_generate)
    a, b = _items()
    client = _client(tmp_path)
    rec = run_dialogue(
        FakeLoaded(), client, b, "ref", "gate", gate=lambda it, t: True, max_rounds=2
    )
    assert rec.asked == 2 and len(rec.user_replies) == 2
    assert rec.final_answer == "Final plan." and rec.grade == "correct"

    rec_off = run_dialogue(FakeLoaded(), client, b, "ref", "off", gate=None, max_rounds=2)
    assert rec_off.asked == 0


def test_summarize_groups_by_label():
    recs = [
        LoopRecord("x", "specified", "off", 0, grade="correct", assumption_made=False),
        LoopRecord("y", "underspecified", "off", 0, grade="wrong", assumption_made=True),
    ]
    s = summarize(recs)
    assert s["all"]["n"] == 2 and s["underspecified"]["assumption_rate"] == 1.0
    assert s["specified"]["correct"] == 1.0
