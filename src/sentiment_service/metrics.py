"""Prometheus metrics exposed on /metrics.

Two families:
* service health (traffic, errors, latency) -> classic RED metrics;
* model / data signals (label mix, confidence, text length, social markers) -> drift proxies,
  since in production we do not get ground-truth labels in real time.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from sentiment_service.pipeline.drift import MARKERS

REQUESTS = Counter(
    "sentiment_http_requests_total",
    "HTTP requests by endpoint, method and status code.",
    ["endpoint", "method", "status"],
)
LATENCY = Histogram(
    "sentiment_http_request_duration_seconds",
    "HTTP request latency.",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
PREDICTIONS = Counter(
    "sentiment_predictions_total",
    "Predicted labels (one increment per text).",
    ["label"],
)
CONFIDENCE = Histogram(
    "sentiment_prediction_confidence",
    "Probability of the predicted (top) class.",
    buckets=(0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
LOW_CONFIDENCE = Counter(
    "sentiment_low_confidence_predictions_total",
    "Predictions whose top-class probability is below LOW_CONFIDENCE_THRESHOLD.",
)
TEXT_LENGTH = Histogram(
    "sentiment_input_text_length_chars",
    "Length of incoming texts in characters.",
    buckets=(10, 25, 50, 100, 140, 200, 280, 500, 1000, 2000),
)
INPUT_MARKERS = Counter(
    "sentiment_input_markers_total",
    "Texts containing a social-media marker (mention, url, hashtag).",
    ["marker"],
)
MODEL_LOADED = Gauge("sentiment_model_loaded", "1 if the model is loaded and ready.")
MODEL_INFO = Gauge("sentiment_model_info", "Model currently served.", ["model_name"])


def record_predictions(texts, predictions, low_confidence_threshold: float) -> None:
    for text, pred in zip(texts, predictions, strict=True):
        PREDICTIONS.labels(label=pred.label).inc()
        CONFIDENCE.observe(pred.score)
        TEXT_LENGTH.observe(len(text))
        if pred.score < low_confidence_threshold:
            LOW_CONFIDENCE.inc()
        for name, pattern in MARKERS.items():
            if pattern.search(text):
                INPUT_MARKERS.labels(marker=name).inc()
