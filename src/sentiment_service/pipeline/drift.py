"""Label-free drift signals comparing a reference window with the current one.

* data drift   -> distribution of the inputs changes (text length, share of
                  mentions / links / hashtags, i.e. "is this still social media?");
* prediction drift -> distribution of predicted labels / confidence changes, an
                  early proxy for concept drift when ground truth is not yet available.
Concept drift proper (same text, different correct label) is only confirmed by
the macro-F1 on freshly labelled samples, see ``evaluation.py``.

Note: a word-level out-of-vocabulary rate was tried and dropped. Against a
1000-tweet reference ~30% of tokens are unseen in *any* window (tweets have a very
long-tailed vocabulary), so it does not separate stable from drifted traffic.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from sentiment_service.model import LABELS

PSI_WARN = 0.1  # moderate shift
PSI_ALERT = 0.2  # significant shift (common industry rule of thumb)
CONFIDENCE_DROP_ALERT = 0.05  # absolute drop of the mean top-class probability
MARKER_SHIFT_ALERT = 0.15  # absolute change in the share of texts with @/URL/#

MARKERS = {
    "mention": re.compile(r"(?<!\w)@\w+"),
    "url": re.compile(r"https?://|www\.|\bhttp\b"),
    "hashtag": re.compile(r"(?<!\w)#\w+"),
}


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Population Stability Index between two discrete distributions (same bins)."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_psi(reference, current, bins: int = 10) -> float:
    """PSI of a continuous feature, with quantile bins fitted on the reference."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi(np.histogram(reference, edges)[0], np.histogram(current, edges)[0])


def label_distribution(labels) -> np.ndarray:
    counts = Counter(labels)
    return np.array([counts.get(label, 0) for label in LABELS], dtype=float)


def marker_shares(texts) -> dict[str, float]:
    texts = list(texts)
    return {
        name: float(np.mean([bool(rx.search(t)) for t in texts])) if texts else 0.0
        for name, rx in MARKERS.items()
    }


def drift_report(
    ref_texts, ref_preds, ref_conf, cur_texts, cur_preds, cur_conf
) -> dict[str, float | bool]:
    ref_markers, cur_markers = marker_shares(ref_texts), marker_shares(cur_texts)
    report = {
        "text_length_psi": numeric_psi([len(t) for t in ref_texts], [len(t) for t in cur_texts]),
        "marker_shift": max(abs(cur_markers[k] - ref_markers[k]) for k in MARKERS),
        "label_psi": psi(label_distribution(ref_preds), label_distribution(cur_preds)),
        "mean_confidence_ref": float(np.mean(ref_conf)),
        "mean_confidence_cur": float(np.mean(cur_conf)),
    }
    report["confidence_drop"] = report["mean_confidence_ref"] - report["mean_confidence_cur"]
    report["data_drift"] = bool(
        report["text_length_psi"] > PSI_ALERT or report["marker_shift"] > MARKER_SHIFT_ALERT
    )
    report["prediction_drift"] = bool(
        report["label_psi"] > PSI_ALERT or report["confidence_drop"] > CONFIDENCE_DROP_ALERT
    )
    report["drift_detected"] = report["data_drift"] or report["prediction_drift"]
    return report
