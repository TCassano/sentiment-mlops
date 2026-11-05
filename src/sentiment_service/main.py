"""FastAPI application: /health, /predict, /predict/batch, /metrics."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import make_asgi_app

from sentiment_service import __version__, metrics
from sentiment_service.config import Settings, get_settings
from sentiment_service.model import HFSentimentModel, SentimentPredictor
from sentiment_service.schemas import (
    BatchPredictionOut,
    BatchPredictRequest,
    HealthOut,
    PredictionOut,
    PredictRequest,
)

logger = logging.getLogger("sentiment_service")


def _default_loader(settings: Settings) -> SentimentPredictor:
    return HFSentimentModel(
        settings.model_name,
        revision=settings.model_revision,
        max_seq_length=settings.max_seq_length,
    )


def create_app(
    settings: Settings | None = None,
    loader: Callable[[Settings], SentimentPredictor] = _default_loader,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = None
        try:
            started = time.perf_counter()
            app.state.predictor = loader(settings)
            logger.info(
                "Model %s loaded in %.1fs", settings.model_name, time.perf_counter() - started
            )
            metrics.MODEL_LOADED.set(1)
            metrics.MODEL_INFO.labels(model_name=settings.model_name).set(1)
        except Exception:
            # Keep the process up so /health can report the failure instead of crash-looping.
            logger.exception("Model loading failed")
            metrics.MODEL_LOADED.set(0)
        yield
        metrics.MODEL_LOADED.set(0)

    app = FastAPI(
        title="Sentiment Monitoring API",
        description="Classifies social media texts as negative / neutral / positive.",
        version=__version__,
        lifespan=lifespan,
    )
    app.mount("/metrics", make_asgi_app())

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            # Use the route template (not the raw URL) to keep label cardinality bounded.
            endpoint = getattr(route, "path", "unmatched")
            if endpoint != "/metrics":
                metrics.REQUESTS.labels(endpoint, request.method, str(status)).inc()
                metrics.LATENCY.labels(endpoint).observe(time.perf_counter() - started)

    def _predictor(request: Request) -> SentimentPredictor:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return predictor

    def _run(request: Request, texts: list[str]) -> list[PredictionOut]:
        too_long = [i for i, t in enumerate(texts) if len(t) > settings.max_text_chars]
        if too_long:
            raise HTTPException(
                status_code=422,
                detail=f"texts at positions {too_long} exceed {settings.max_text_chars} chars",
            )
        predictions = _predictor(request).predict(texts)
        metrics.record_predictions(texts, predictions, settings.low_confidence_threshold)
        return [
            PredictionOut(text=t, label=p.label, score=p.score, scores=p.scores)
            for t, p in zip(texts, predictions, strict=True)
        ]

    @app.get("/health", response_model=HealthOut)
    def health(request: Request) -> HealthOut:
        predictor = request.app.state.predictor
        loaded = predictor is not None
        return HealthOut(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model=predictor.model_name if loaded else None,
            version=__version__,
        )

    @app.post("/predict", response_model=PredictionOut)
    def predict(body: PredictRequest, request: Request) -> PredictionOut:
        return _run(request, [body.text])[0]

    @app.post("/predict/batch", response_model=BatchPredictionOut)
    def predict_batch(body: BatchPredictRequest, request: Request) -> BatchPredictionOut:
        if len(body.texts) > settings.max_batch_size:
            raise HTTPException(
                status_code=422, detail=f"batch size exceeds {settings.max_batch_size}"
            )
        predictions = _run(request, body.texts)
        return BatchPredictionOut(model=_predictor(request).model_name, predictions=predictions)

    return app


app = create_app()
