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
pre-commit install                         # ruff gira automaticamente a ogni commit
```

| File | Contenuto |
|---|---|
| `requirements.txt` | dipendenze del servizio, con versioni bloccate |
| `requirements-torch-cpu.txt` | PyTorch CPU, installato a parte dal suo indice |
| `requirements-dev.txt` | test, lint, dataset e strumenti di sviluppo |
| `pyproject.toml` | metadati del pacchetto, configurazione di ruff e pytest |
