"""Unit tests of the label-free drift signals (PSI, social markers, confidence)."""

import numpy as np
import pytest

from sentiment_service.pipeline.drift import (
    MARKER_SHIFT_ALERT,
    PSI_ALERT,
    drift_report,
    marker_shares,
    numeric_psi,
    psi,
)


def test_psi_is_zero_for_identical_distributions():
    assert psi(np.array([10, 20, 30]), np.array([10, 20, 30])) == pytest.approx(0.0)


def test_psi_detects_large_shift():
    assert psi(np.array([80, 10, 10]), np.array([10, 10, 80])) > PSI_ALERT


def test_numeric_psi_stable_vs_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 20, 5000)
    assert numeric_psi(ref, rng.normal(100, 20, 5000)) < 0.05
    assert numeric_psi(ref, rng.normal(160, 20, 5000)) > PSI_ALERT


def test_drift_report_flags_shift_and_not_stability():
    texts = ["good morning everyone"] * 50 + ["i hate the traffic today"] * 50
    preds = ["positive"] * 50 + ["negative"] * 50
    conf = [0.9] * 100
    stable = drift_report(texts, preds, conf, texts, preds, conf)
    assert stable["drift_detected"] is False

    new_texts = ["quarterly earnings guidance revised upward amid volatility " * 4] * 100
    shifted = drift_report(texts, preds, conf, new_texts, ["neutral"] * 100, [0.6] * 100)
    assert shifted["data_drift"] is True
    assert shifted["prediction_drift"] is True


def test_marker_shift_detects_non_social_text():
    tweets = ["@user loving the #launch", "so tired today", "@user help pls"] * 20
    reviews = ["A slow, beautifully shot film about grief."] * 60
    conf = [0.9] * 60
    preds = ["neutral"] * 60
    assert marker_shares(tweets)["mention"] == pytest.approx(2 / 3)
    report = drift_report(tweets, preds, conf, reviews, preds, conf)
    assert report["marker_shift"] > MARKER_SHIFT_ALERT
    assert report["data_drift"] is True
