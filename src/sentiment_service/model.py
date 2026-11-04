"""Thin wrapper around the HuggingFace sentiment model.

The API depends only on the ``SentimentPredictor`` protocol, so tests can swap in a
fake predictor and the service never needs to know where the weights come from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

LABELS = ("negative", "neutral", "positive")

_USER_RE = re.compile(r"(?<!\w)@\w+")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")


def preprocess(text: str) -> str:
    """Normalise a tweet the same way the model was trained (see the model card).

    User mentions become ``@user`` and links become ``http``; whitespace is collapsed.
    """
    text = _USER_RE.sub("@user", text)
    text = _URL_RE.sub("http", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    scores: dict[str, float]


class SentimentPredictor(Protocol):
    model_name: str

    def predict(self, texts: list[str]) -> list[Prediction]: ...


class HFSentimentModel:
    """Loads tokenizer + model once and runs batched inference on CPU (or GPU if present)."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        max_seq_length: int = 128,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision
        ).to(self.device)
        self.model.eval()
        id2label = self.model.config.id2label
        self.labels = [id2label[i].lower() for i in range(len(id2label))]

    def predict(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        results: list[Prediction] = []
        for start in range(0, len(texts), batch_size):
            batch = [preprocess(t) for t in texts[start : start + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.no_grad():
                probs = self._torch.softmax(self.model(**encoded).logits, dim=-1).cpu()
            for row in probs.tolist():
                scores = dict(zip(self.labels, row, strict=True))
                best = max(scores, key=scores.__getitem__)
                results.append(Prediction(label=best, score=scores[best], scores=scores))
        return results
