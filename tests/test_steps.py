"""End-to-end run of the DAG steps with a fake model and an in-memory dataset."""

import json
from pathlib import Path

import pandas as pd
import pytest

from sentiment_service.model import Prediction
from sentiment_service.pipeline import data, steps, training
from tests.fakes import FakePredictor

TEXTS = {
    "i love this brand": "positive",
    "great update": "positive",
    "i hate waiting": "negative",
    "awful service": "negative",
    "store opens at 9": "neutral",
    "it is tuesday": "neutral",
    "love the hate": "negative",  # the fake model gets this one wrong
}


class OraclePredictor:
    """Stands in for a retrained candidate that fixed the errors of the current model."""

    def __init__(self, name):
        self.model_name = name

    def predict(self, texts):
        return [Prediction(TEXTS[t], 0.95, {TEXTS[t]: 0.95}) for t in texts]


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PIPELINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("PIPELINE_REPORTS_DIR", str(tmp_path / "reports"))

    frame = pd.DataFrame({"text": list(TEXTS) * 10, "label": list(TEXTS.values()) * 10})
    monkeypatch.setattr(data, "load_tweet_eval", lambda *a, **k: frame.copy())
    monkeypatch.setattr(
        steps,
        "_load",
        lambda model: OraclePredictor(model) if "candidates" in model else FakePredictor(),
    )

    def fake_fine_tune(base_model, texts, labels, output_dir, **_):
        output_dir.mkdir(parents=True)
        return output_dir

    monkeypatch.setattr(training, "fine_tune", fake_fine_tune)
    return tmp_path


def test_full_loop_promotes_better_candidate(pipeline_env):
    run_id = "20260105"
    batch = steps.validate(steps.collect(run_id, sample=70))
    evaluated = steps.evaluate_current(batch, run_id)
    assert 0 < evaluated["macro_f1"] < 1

    drift = steps.check_drift(evaluated["predictions_path"])
    assert drift["drift_detected"] is False  # same distribution as the reference

    candidate = steps.retrain(batch, run_id)
    result = steps.promote_or_reject(candidate, run_id)
    assert result["promoted"] is True
    assert result["candidate_f1"] == pytest.approx(1.0)

    registry = json.loads((pipeline_env / "models/registry/current.json").read_text())
    assert registry["model"] == candidate
    assert steps.current_model() == candidate

    report = steps.write_report(run_id, evaluation=evaluated, drift=drift, promotion=result)
    assert json.loads(Path(report).read_text())["promotion"]["promoted"] is True


def test_decide_uses_reference_f1(pipeline_env):
    reference = steps.ensure_reference()
    healthy = {"macro_f1": reference["macro_f1"]}
    assert steps.decide(healthy, {"drift_detected": False})["retrain"] is False
    assert steps.decide(healthy, {"drift_detected": True})["retrain"] is True
    assert steps.decide({"macro_f1": 0.3}, {"drift_detected": False})["retrain"] is True


def test_validate_rejects_bad_batch(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"text": ["", "x"], "label": ["neutral", "angry"]}).to_parquet(path)
    with pytest.raises(ValueError, match="Data validation failed"):
        steps.validate(str(path))
