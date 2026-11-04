"""Contract tests of the HTTP API (model replaced by a fake: fast and deterministic)."""

import pytest


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"] == "fake-model"


def test_predict_single(client):
    response = client.post("/predict", json={"text": "I love this product"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert 0.0 <= body["score"] <= 1.0
    assert set(body["scores"]) == {"negative", "neutral", "positive"}


def test_predict_batch_preserves_order(client):
    texts = ["I hate Mondays", "The meeting is at 5pm", "Great job team"]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake-model"
    assert [p["text"] for p in body["predictions"]] == texts
    assert [p["label"] for p in body["predictions"]] == ["negative", "neutral", "positive"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": ""}, {"text": "   "}, {"text": 123}, {"text": "x" * 201}],
)
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [{"texts": []}, {"texts": ["ok", " "]}, {"texts": ["a"] * 5}],
)
def test_batch_rejects_invalid_input(client, payload):
    assert client.post("/predict/batch", json=payload).status_code == 422


def test_degraded_when_model_fails_to_load(broken_client):
    health = broken_client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["model_loaded"] is False
    assert broken_client.post("/predict", json={"text": "hello"}).status_code == 503
