# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# 1) Dependencies first: this layer is cached as long as the requirements don't change.
COPY requirements-torch-cpu.txt requirements.txt ./
RUN pip install -r requirements-torch-cpu.txt && pip install -r requirements.txt

# 2) Application code.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# 3) Run as an unprivileged user (uid 1000 is also what HuggingFace Spaces expects).
RUN useradd --create-home --uid 1000 app
USER app
ENV HF_HOME=/home/app/.cache/huggingface

# 4) Bake the weights into the image: no download at startup, same weights on every run.
ARG MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest
ARG PRELOAD_MODEL=1
ENV MODEL_NAME=${MODEL_NAME}
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
      python -c "import os; from sentiment_service.model import HFSentimentModel as M; M(os.environ['MODEL_NAME'])"; \
    fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn sentiment_service.main:app --host 0.0.0.0 --port ${PORT}"]
