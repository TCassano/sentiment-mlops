# Sentiment Monitoring — pipeline MLOps

Servizio di classificazione del sentiment (negativo / neutro / positivo) per post social, basato sul modello
pre-addestrato
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

## Setup dell'ambiente

Richiede Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements-torch-cpu.txt  # PyTorch CPU (~200 MB invece di ~2.5 GB con CUDA)
pip install -r requirements-dev.txt
pip install -e . --no-deps                 # installa il pacchetto sentiment_service (src/)
pre-commit install                         # ruff gira automaticamente a ogni commit
```

## Avvio del servizio

```bash
uvicorn sentiment_service.main:app --reload   # documentazione interattiva: http://localhost:8000/docs
```

| Metodo | Path | Descrizione |
|---|---|---|
| GET | `/health` | stato del servizio e del modello (`ok` / `degraded`) |
| POST | `/predict` | `{"text": str}` → etichetta, confidenza, probabilità per classe |
| POST | `/predict/batch` | `{"texts": [str, ...]}` (max 64) |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"text": "@acme the new update is fantastic!"}'
```

Configurazione tramite variabili d'ambiente: `MODEL_NAME`, `MODEL_REVISION`, `MAX_TEXT_CHARS`,
`MAX_BATCH_SIZE`, `MAX_SEQ_LENGTH`.

| File | Contenuto |
|---|---|
| `requirements.txt` | dipendenze del servizio, con versioni bloccate |
| `requirements-torch-cpu.txt` | PyTorch CPU, installato a parte dal suo indice |
| `requirements-dev.txt` | test, lint, dataset e strumenti di sviluppo |
| `pyproject.toml` | metadati del pacchetto, configurazione di ruff e pytest |
