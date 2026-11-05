# Sentiment Monitoring — pipeline MLOps

[![CI/CD](https://github.com/TCassano/sentiment-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/TCassano/sentiment-mlops/actions/workflows/ci-cd.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TCassano/sentiment-mlops/blob/main/notebooks/MLOps_Sentiment_Pipeline.ipynb)

Servizio di classificazione del sentiment (negativo / neutro / positivo) per post social, basato sul modello
pre-addestrato
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
con test, container, CI/CD, monitoraggio (Prometheus + Grafana) e un ciclo settimanale di valutazione e retraining
(Airflow).

Il notebook [`notebooks/MLOps_Sentiment_Pipeline.ipynb`](notebooks/MLOps_Sentiment_Pipeline.ipynb) documenta
architettura, scelte, risultati e limiti, ed esegue sul codice reale test, valutazione, analisi di drift e un giro
completo degli step del DAG.

```
src/sentiment_service/   servizio FastAPI, modello, metriche, configurazione
  └── pipeline/          raccolta, validazione, valutazione, drift, retraining (usati dal DAG)
tests/                   test di API, modello, drift, pipeline, step e DAG
airflow/dags/            DAG settimanale
monitoring/              Prometheus (scrape + alert) e Grafana (datasource, dashboard, alert)
scripts/                 valutazione da CLI e generatore di traffico
notebooks/               notebook di documentazione e risultati
```

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

## Pipeline di valutazione e retraining (Airflow)

Il DAG [`sentiment_monitoring_retraining`](airflow/dags/sentiment_retraining_dag.py) gira ogni settimana:

```
collect_data → validate_data → evaluate_current_model → check_drift → decide
    decide ──(retrain)──→ retrain_model → evaluate_and_promote ──┐
    decide ──(ok)───────→ skip_retraining ───────────────────────┴─→ publish_report
```

| Passo | Cosa fa |
|---|---|
| `collect_data` | raccoglie un lotto di post etichettati (simulato con `tweet_eval` train) |
| `validate_data` | controlli di qualità: righe minime, colonne, testi vuoti, etichette ammesse; se fallisce, il run si ferma |
| `evaluate_current_model` | macro-F1 del modello in produzione sul lotto |
| `check_drift` | PSI della lunghezza e delle etichette, marcatori social, calo di confidenza rispetto al riferimento |
| `decide` | retraining se macro-F1 < 0.65, se cala di oltre 0.05 rispetto al riferimento, o se c'è drift |
| `retrain_model` | fine-tuning di un'epoca sul lotto nuovo → modello candidato |
| `evaluate_and_promote` | il candidato va in produzione solo se batte il modello attuale di almeno 0.01 di macro-F1 su un *golden set* fisso |
| `publish_report` | report JSON del run, qualunque ramo sia stato eseguito |

Tutta la logica sta in [`sentiment_service.pipeline`](src/sentiment_service/pipeline/) ed è testata senza Airflow
(`tests/test_steps.py`, con un modello finto). Il file del DAG si limita a collegare i passi; i task si passano
solo valori piccoli (percorsi dei file e dizionari di metriche) tramite XCom, mai i dati.

Riferimento misurato sul modello attuale (`tweet_eval` validation, 1000 esempi stratificati): macro-F1 **0.753**,
predizioni con confidenza < 0.5 **4.0%**.

```bash
python scripts/evaluate.py --sample 2000        # valutazione singola, report in reports/eval.json

# Airflow (Linux/WSL, Airflow 2.10)
pip install "apache-airflow==2.10.4" --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.10.4/constraints-3.11.txt
pip install -r requirements-torch-cpu.txt -r requirements-dev.txt && pip install .
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/airflow/dags
airflow standalone
```

Nel CI il job `dag-check` installa Airflow e verifica che il DAG si carichi senza errori e abbia la struttura attesa.

## CI/CD (GitHub Actions)

Il workflow [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml) gira a ogni push e pull request.

| Job | Fase | Cosa fa |
|---|---|---|
| `lint` | CI | `ruff check` e `ruff format --check` |
| `test` | CI | test veloci con modello finto e copertura |
| `test-model` | CI | test comportamentali sul modello reale (pesi in cache) |
| `dag-check` | CI | installa Airflow e verifica che il DAG si carichi con la struttura attesa |
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

## Comandi rapidi (Makefile)

Su Linux/macOS (o Windows con `make` installato), con il venv attivo:

| Comando | Cosa fa |
|---|---|
| `make install` | installa le dipendenze bloccate, il pacchetto e gli hook di pre-commit |
| `make lint` / `make format` | controlla / sistema stile e formattazione con ruff |
| `make test` / `make test-model` | suite veloce con copertura / test sul modello reale |
| `make docker-build` / `make docker-run` | costruisce / avvia l'immagine del servizio |
| `make up` / `make down` | avvia / spegne lo stack con Prometheus e Grafana |
