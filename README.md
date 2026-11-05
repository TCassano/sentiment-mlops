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

| File | Contenuto |
|---|---|
| `requirements.txt` | dipendenze del servizio, con versioni bloccate |
| `requirements-torch-cpu.txt` | PyTorch CPU, installato a parte dal suo indice |
| `requirements-dev.txt` | test, lint, dataset e strumenti di sviluppo |
| `pyproject.toml` | metadati del pacchetto, configurazione di ruff e pytest |

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

## Docker

```bash
docker build -t sentiment-api:local .          # ~5 min la prima volta, poi pochi secondi se cambia solo il codice
docker run --rm -p 8000:8000 sentiment-api:local
```

L'immagine usa `python:3.11-slim` con PyTorch CPU, include i pesi del modello (nessun download all'avvio), gira
come utente non-root e ha un `HEALTHCHECK` su `/health`. Per un'immagine senza pesi: `--build-arg PRELOAD_MODEL=0`.

## Monitoraggio (Prometheus + Grafana)

```bash
docker compose up -d --build     # API + Prometheus + Grafana + generatore di traffico
docker compose down              # spegne tutto (aggiungere -v per cancellare anche i dati)
```

| Servizio | Indirizzo |
|---|---|
| API | <http://localhost:8000/docs>, metriche su `/metrics/` |
| Prometheus | <http://localhost:9090/alerts> |
| Grafana | <http://localhost:3000> (admin / admin) → cartella **Sentiment** → dashboard **Sentiment API** |

**Segnali monitorati.** In produzione le etichette vere non arrivano in tempo reale, quindi il servizio espone due
famiglie di metriche:

- **salute del servizio**: richieste per stato HTTP, latenza (istogramma), modello caricato;
- **segnali di drift senza etichette**: distribuzione delle etichette predette, confidenza della classe vincente,
  quota di predizioni a bassa confidenza (< 0.5), lunghezza dei testi, quota di testi con menzioni, link e hashtag.

Il **data drift** (cambia il tipo di testo in ingresso) si vede dalla quota di marcatori social e dalla lunghezza
dei testi. Il **prediction drift** (cambia la distribuzione delle predizioni o cala la confidenza) è un primo
indizio di **concept drift**, che si conferma solo misurando la macro-F1 su campioni etichettati di recente.

**Alert principale.** La quota di predizioni con confidenza < 0.5 supera il **15%** per 30 minuti. Sui dati di
riferimento questa quota è circa il 4%: la soglia è quasi 4 volte il valore normale, abbastanza alta da non
scattare per il rumore e abbastanza bassa da accorgersi di testi fuori dominio. Gli altri alert (servizio giù, errori
5xx > 5%, latenza p95 > 1 s, spostamento dei marcatori social, picco di negativi) sono in
[`monitoring/prometheus/alerts.yml`](monitoring/prometheus/alerts.yml).

Per vedere reagire i pannelli di drift, mandare testi fuori dominio dopo 200 richieste:
`python scripts/load_generator.py --drift-after 200`.

## CI/CD (GitHub Actions)

Il workflow [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml) gira a ogni push e pull request.

| Job | Fase | Cosa fa |
|---|---|---|
| `lint` | CI | `ruff check` e `ruff format --check` |
| `test` | CI | test veloci con modello finto e copertura |
| `test-model` | CI | test comportamentali sul modello reale (pesi in cache) |
| `docker` | build + CD | build dell'immagine, smoke test del container; su `main` e sui tag `v*` pubblica su GHCR |
| `deploy-hf-space` | CD (opzionale) | carica il servizio su uno Space Docker di HuggingFace |

Il job `docker` parte solo se i tre job di CI passano (`needs`). Il deploy su HuggingFace è opzionale: si attiva
creando il secret `HF_TOKEN` (token con permesso di scrittura) e la variabile `HF_SPACE_ID` (es. `utente/sentiment-api`)
in *Settings → Secrets and variables → Actions*. Senza questi valori il job termina senza errori e salta il deploy.

## Test e qualità del codice

```bash
pytest -m "not model"          # suite veloce: API con modello finto (~1 s)
pytest -m model                # test comportamentali sul modello reale (scarica i pesi la prima volta)
ruff check . && ruff format --check .
```
