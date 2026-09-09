import json
from types import SimpleNamespace

from halluscope.config import AliasCfg, JudgeCfg
from halluscope.data.augment import FIELDS, augment_family, build_dataset
from halluscope.data.expand import FamilySpec
from halluscope.data.io import load_items
from halluscope.judge.client import JudgeClient


def _spec():
    return FamilySpec(
        family="rag-0001",
        topic="rag",
        setup="We index docs.",
        specified="Chunk at 512 with bge-m3 and cosine, top 8.",
        underspecified="Chunk and embed the docs.",
        gap="chunk size, model, metric, k missing",
        question="Which chunk size, model, metric, and k?",
        assumption="picks defaults",
        fill="We use bge-m3, cosine, 512 chunks, top 8.",
        final="Write the code.",
        contradiction="We cannot run embeddings anymore.",
        update="Also, log every query to a file.",
    )


def _resp(content):
    r = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    r._hidden_params = {"response_cost": 0.0}
    return r


def _client(tmp_path, fn):
    cfg = JudgeCfg(aliases={"augmenter": AliasCfg(models=["fake/x"])}, max_retries=2)
    return JudgeClient(cfg, cost_log=tmp_path / "c.jsonl", completion_fn=fn)


def test_augment_family_preserves_structure(tmp_path):
    def fake(**kw):
        k = kw["messages"][-1]["content"].split("Paraphrase variant number: ")[1][0]
        payload = {f: f"{f} paraphrase {k}" for f in FIELDS}
        return _resp(json.dumps(payload))

    items = augment_family(_client(tmp_path, fake), _spec(), n_paraphrases=2)
    assert len(items) == 8
    fams = {it.family for it in items}
    assert fams == {"rag-0001-p1", "rag-0001-p2"}
    assert all(it.source == "augmented" and it.review_status == "pending" for it in items)
    by = {it.id: it for it in items}
    assert by["rag-0001-p1-b"].label == "underspecified"
    assert by["rag-0001-p1-b"].gap == "gap paraphrase 1"
    assert by["rag-0001-p1-d"].label == "inconsistent" and len(by["rag-0001-p1-d"].turns) == 3


def test_augment_skips_after_repeated_bad_output(tmp_path):
    def bad(**kw):
        return _resp('{"setup": "only one field"}')

    items = augment_family(_client(tmp_path, bad), _spec(), n_paraphrases=1)
    assert items == []


def test_build_dataset_writes_seeds_and_augmented(tmp_path, monkeypatch):
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "rag.json").write_text(json.dumps([_spec().model_dump()]), encoding="utf-8")

    def fake(**kw):
        return _resp(json.dumps({f: f"{f} new" for f in FIELDS}))

    out = tmp_path / "items.jsonl"
    n = build_dataset(seeds, out, n_paraphrases=1, client=_client(tmp_path, fake))
    assert n == 8
    items = load_items(out)
    assert sum(it.source == "seed" for it in items) == 4
    assert sum(it.source == "augmented" for it in items) == 4
