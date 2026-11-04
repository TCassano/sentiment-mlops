"""Runtime configuration, read from environment variables (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # Either a HuggingFace Hub id or a local directory (e.g. a promoted retrained model).
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", DEFAULT_MODEL))
    # Pin the Hub revision so the same image always serves the same weights.
    model_revision: str | None = field(default_factory=lambda: os.getenv("MODEL_REVISION") or None)
    max_text_chars: int = field(default_factory=lambda: _env_int("MAX_TEXT_CHARS", 2000))
    max_batch_size: int = field(default_factory=lambda: _env_int("MAX_BATCH_SIZE", 64))
    max_seq_length: int = field(default_factory=lambda: _env_int("MAX_SEQ_LENGTH", 128))
    # Predictions with top-class probability below this are counted as "low confidence".
    low_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.5"))
    )


def get_settings() -> Settings:
    return Settings()
