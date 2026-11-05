PY ?= python
IMAGE ?= sentiment-api:local

.PHONY: install lint format test test-model docker-build docker-run up down

install:  ## create the reproducible dev environment (inside an activated venv)
	$(PY) -m pip install -r requirements-torch-cpu.txt
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

test:  ## fast suite (fake model)
	pytest -m "not model" --cov=sentiment_service --cov-report=term-missing

test-model:  ## behavioural tests on the real model
	pytest -m model

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

up:  ## API + Prometheus + Grafana + traffic generator
	docker compose up -d --build

down:
	docker compose down
