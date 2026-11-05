"""Model evaluation and the retrain / promote decision rules."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sentiment_service.model import LABELS, SentimentPredictor


@dataclass(frozen=True)
class Policy:
    """Thresholds of the retraining loop (kept in one place, versioned with the code)."""

    # Absolute floor on macro-F1 on freshly labelled data.
    min_macro_f1: float = 0.65
    # Tolerated drop with respect to the reference macro-F1 measured at release time.
    max_f1_drop: float = 0.05
    # A retrained candidate must beat the current model by at least this margin.
    min_promotion_gain: float = 0.01


def evaluate(predictor: SentimentPredictor, texts: list[str], labels: list[str]) -> dict:
    preds = [p.label for p in predictor.predict(list(texts))]
    per_class = f1_score(labels, preds, labels=list(LABELS), average=None, zero_division=0)
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=list(LABELS), average="macro")),
        "f1_per_class": dict(zip(LABELS, map(float, per_class), strict=True)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(LABELS)).tolist(),
        "predictions": preds,
    }


def needs_retraining(
    macro_f1: float, reference_f1: float, drift_detected: bool, policy: Policy = Policy()
) -> tuple[bool, list[str]]:
    reasons = []
    if macro_f1 < policy.min_macro_f1:
        reasons.append(f"macro_f1 {macro_f1:.3f} < floor {policy.min_macro_f1}")
    if reference_f1 - macro_f1 > policy.max_f1_drop:
        reasons.append(f"macro_f1 dropped {reference_f1 - macro_f1:.3f} > {policy.max_f1_drop}")
    if drift_detected:
        reasons.append("input/prediction drift detected")
    return bool(reasons), reasons


def should_promote(candidate_f1: float, current_f1: float, policy: Policy = Policy()) -> bool:
    return candidate_f1 >= current_f1 + policy.min_promotion_gain
