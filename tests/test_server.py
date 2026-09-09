import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HALLUSCOPE_PATHS__RESULTS_DIR", str(tmp_path / "results"))
    # Pin the data dir at an empty tmp tree so the item routes fall back to the
    # read-only seeds. Without this the tests read whatever dataset the developer
    # last built, and the PATCH case below would write to it.
    monkeypatch.setenv("HALLUSCOPE_PATHS__DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "probe_qwen_gap_last.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3.5-4B",
                "model_key": "qwen",
                "target": "gap",
                "pooling": "last",
                "n_layers": 3,
                "split_sizes": {"train": 1, "val": 1, "test": 1},
                "probes": {"linear": {"best_layer": 2, "test": {"auroc": 0.9}}},
                "baselines": {},
                "loto": {},
            }
        ),
        encoding="utf-8",
    )
    from halluscope.config import get_settings

    get_settings.cache_clear()
    from server.app import app

    yield TestClient(app)
    get_settings.cache_clear()


def test_health_and_models(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"]
    r = client.get("/api/models")
    assert "qwen" in r.json()["models"]


def test_items_from_seeds(client):
    r = client.get("/api/items", params={"topic": "rag", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 32 and len(body["items"]) == 5
    one = client.get(f"/api/items/{body['items'][0]['id']}")
    assert one.status_code == 200
    r = client.patch("/api/items/rag-0001-a", json={"review_status": "approved"})
    assert r.status_code == 400  # seeds are read-only


def test_results_listing(client):
    r = client.get("/api/results")
    assert r.status_code == 200
    names = [x["name"] for x in r.json()["results"]]
    assert "probe_qwen_gap_last" in names
    full = client.get("/api/results/probe_qwen_gap_last").json()
    assert full["probes"]["linear"]["best_layer"] == 2
    assert client.get("/api/results/nope").status_code == 404


def test_provenance(client):
    r = client.get("/api/provenance")
    assert r.status_code == 200 and "transformers" in r.json()


def test_score_rejects_dialogue_not_ending_on_user(client):
    r = client.post(
        "/api/score",
        json={"turns": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]},
    )
    assert r.status_code == 400
