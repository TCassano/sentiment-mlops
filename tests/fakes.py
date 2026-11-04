from __future__ import annotations

from sentiment_service.model import Prediction


class FakePredictor:
    """Deterministic keyword-based stand-in for the real model (fast API tests)."""

    model_name = "fake-model"

    def predict(self, texts: list[str]) -> list[Prediction]:
        out = []
        for text in texts:
            lowered = text.lower()
            if "love" in lowered or "great" in lowered:
                label = "positive"
            elif "hate" in lowered or "awful" in lowered:
                label = "negative"
            else:
                label = "neutral"
            scores = {k: 0.05 for k in ("negative", "neutral", "positive")}
            scores[label] = 0.9
            out.append(Prediction(label=label, score=0.9, scores=scores))
        return out
