"""Unit tests of the evaluation, retraining decision and data validation logic."""

import pandas as pd
import pytest

from sentiment_service.pipeline.data import validate_batch
from sentiment_service.pipeline.evaluation import Policy, evaluate, needs_retraining, should_promote
from tests.fakes import FakePredictor


def test_evaluate_with_fake_predictor():
    texts = ["love it", "hate it", "it is tuesday", "great"]
    labels = ["positive", "negative", "neutral", "negative"]
    result = evaluate(FakePredictor(), texts, labels)
    assert result["accuracy"] == pytest.approx(0.75)
    assert len(result["confusion_matrix"]) == 3


@pytest.mark.parametrize(
    ("f1", "ref", "drift", "expected"),
    [
        (0.72, 0.72, False, False),
        (0.60, 0.62, False, True),
        (0.68, 0.75, False, True),
        (0.72, 0.72, True, True),
    ],
)
def test_needs_retraining(f1, ref, drift, expected):
    assert needs_retraining(f1, ref, drift)[0] is expected


def test_should_promote_requires_margin():
    policy = Policy(min_promotion_gain=0.01)
    assert should_promote(0.74, 0.72, policy)
    assert not should_promote(0.725, 0.72, policy)


def test_validate_batch():
    good = pd.DataFrame({"text": ["a"] * 60, "label": ["neutral"] * 60})
    assert validate_batch(good) == []
    bad = pd.DataFrame({"text": ["", "b"], "label": ["neutral", "angry"]})
    problems = validate_batch(bad)
    assert any("too few rows" in p for p in problems)
    assert any("empty texts" in p for p in problems)
    assert any("unknown labels" in p for p in problems)
