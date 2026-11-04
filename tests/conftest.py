from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentiment_service.config import DEFAULT_MODEL, Settings
from sentiment_service.main import create_app
from tests.fakes import FakePredictor


@pytest.fixture
def settings() -> Settings:
    return Settings(model_name="fake-model", max_text_chars=200, max_batch_size=4)


@pytest.fixture
def client(settings):
    app = create_app(settings=settings, loader=lambda _: FakePredictor())
    with TestClient(app) as c:  # context manager runs the lifespan (model loading)
        yield c


@pytest.fixture
def broken_client(settings):
    def failing_loader(_):
        raise RuntimeError("weights not found")

    app = create_app(settings=settings, loader=failing_loader)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def real_model():
    from sentiment_service.model import HFSentimentModel

    return HFSentimentModel(DEFAULT_MODEL)
