---
title: Sentiment Monitoring API
emoji: ðŸ“Š
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

FastAPI service classifying social posts as negative / neutral / positive with
`cardiffnlp/twitter-roberta-base-sentiment-latest`.

Endpoints: `GET /health`, `POST /predict`, `POST /predict/batch`, docs at `/docs`.

Deployed automatically by the GitHub Actions `deploy-hf-space` job.
