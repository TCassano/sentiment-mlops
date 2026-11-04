"""Request / response models of the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., examples=["I love the new update, great job!"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace")
        return value


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, examples=[["Great service", "Worst day ever"]])

    @field_validator("texts")
    @classmethod
    def no_blank_items(cls, values: list[str]) -> list[str]:
        if any(not v.strip() for v in values):
            raise ValueError("texts must not contain empty or whitespace-only items")
        return values


class PredictionOut(BaseModel):
    text: str
    label: str = Field(..., examples=["positive"])
    score: float = Field(..., ge=0.0, le=1.0)
    scores: dict[str, float]


class BatchPredictionOut(BaseModel):
    model: str
    predictions: list[PredictionOut]


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    model: str | None
    version: str
