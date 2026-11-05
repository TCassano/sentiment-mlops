"""Steps of the periodic loop: collect -> validate -> evaluate -> drift -> retrain -> promote.

Each step reads/writes files under PIPELINE_DATA_DIR / PIPELINE_MODELS_DIR and returns
small JSON-serialisable values, so Airflow can pass them between tasks via XCom.

Model "registry": ``models/registry/current.json`` points to the model in production
(Hub id or local dir). Promotion = updating that pointer; the CD pipeline then builds
a new image with ``MODEL_NAME`` set to the promoted model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from sentiment_service.config import DEFAULT_MODEL
from sentiment_service.model import LABELS
from sentiment_service.pipeline import data, drift, evaluation

GOLDEN_SAMPLE = 1000  # fixed, stratified slice of tweet_eval test used to compare models
REFERENCE_SAMPLE = 1000  # tweet_eval validation slice describing the "release-time" traffic


def _data_dir() -> Path:
    return Path(os.getenv("PIPELINE_DATA_DIR", "data"))


def _models_dir() -> Path:
    return Path(os.getenv("PIPELINE_MODELS_DIR", "models"))


def _registry() -> Path:
    return _models_dir() / "registry" / "current.json"


def current_model() -> str:
    if _registry().exists():
        return json.loads(_registry().read_text())["model"]
    return DEFAULT_MODEL


def _load(model: str):
    from sentiment_service.model import HFSentimentModel

    return HFSentimentModel(model)


def _predict_frame(predictor, texts: list[str]) -> pd.DataFrame:
    preds = predictor.predict(texts)
    return pd.DataFrame(
        {"text": texts, "pred": [p.label for p in preds], "conf": [p.score for p in preds]}
    )


# --------------------------------------------------------------------------- steps
def collect(run_id: str, sample: int = 500) -> str:
    seed = int(run_id[-6:]) if run_id[-6:].isdigit() else 0
    return str(data.collect_batch(_data_dir() / "raw", run_id, sample=sample, seed=seed))


def validate(batch_path: str) -> str:
    problems = data.validate_batch(pd.read_parquet(batch_path))
    if problems:
        # Failing the task stops the run before a bad batch reaches evaluation/training.
        raise ValueError(f"Data validation failed for {batch_path}: {problems}")
    return batch_path


def ensure_reference(model: str | None = None) -> dict:
    """Reference window (texts, predictions, confidence) + reference macro-F1."""
    ref_dir = _data_dir() / "reference"
    meta_path = ref_dir / "reference.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    model = model or current_model()
    df = data.load_tweet_eval("validation", sample=REFERENCE_SAMPLE)
    frame = _predict_frame(_load(model), df["text"].tolist())
    ref_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(ref_dir / "reference.parquet", index=False)
    macro_f1 = f1_score(df["label"], frame["pred"], labels=list(LABELS), average="macro")
    meta = {"model": model, "macro_f1": float(macro_f1), "n": len(df)}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def evaluate_current(batch_path: str, run_id: str) -> dict:
    model = current_model()
    predictor = _load(model)
    df = pd.read_parquet(batch_path)
    frame = _predict_frame(predictor, df["text"].tolist())
    out = _data_dir() / "predictions" / f"pred_{run_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return {
        "model": model,
        "n": len(df),
        "accuracy": float(accuracy_score(df["label"], frame["pred"])),
        "macro_f1": float(
            f1_score(df["label"], frame["pred"], labels=list(LABELS), average="macro")
        ),
        "predictions_path": str(out),
    }


def check_drift(predictions_path: str) -> dict:
    ensure_reference()
    ref = pd.read_parquet(_data_dir() / "reference" / "reference.parquet")
    cur = pd.read_parquet(predictions_path)
    return drift.drift_report(
        ref["text"], ref["pred"], ref["conf"], cur["text"], cur["pred"], cur["conf"]
    )


def decide(eval_result: dict, drift_result: dict) -> dict:
    reference = ensure_reference()
    retrain, reasons = evaluation.needs_retraining(
        eval_result["macro_f1"], reference["macro_f1"], drift_result["drift_detected"]
    )
    return {"retrain": retrain, "reasons": reasons}


def retrain(batch_path: str, run_id: str, epochs: int = 1) -> str:
    from sentiment_service.pipeline.training import fine_tune

    df = pd.read_parquet(batch_path)
    out = _models_dir() / "candidates" / run_id
    fine_tune(current_model(), df["text"].tolist(), df["label"].tolist(), out, epochs=epochs)
    return str(out)


def golden_f1(model: str) -> float:
    df = data.load_tweet_eval("test", sample=GOLDEN_SAMPLE, seed=7)
    return evaluation.evaluate(_load(model), df["text"].tolist(), df["label"].tolist())["macro_f1"]


def promote_or_reject(candidate: str, run_id: str) -> dict:
    current = current_model()
    candidate_f1, current_f1 = golden_f1(candidate), golden_f1(current)
    promoted = evaluation.should_promote(candidate_f1, current_f1)
    if promoted:
        _registry().parent.mkdir(parents=True, exist_ok=True)
        _registry().write_text(
            json.dumps(
                {
                    "model": candidate,
                    "run_id": run_id,
                    "golden_macro_f1": candidate_f1,
                    "previous": current,
                    "promoted_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )
    return {
        "candidate": candidate,
        "candidate_f1": candidate_f1,
        "current_f1": current_f1,
        "promoted": promoted,
    }


def write_report(run_id: str, **sections) -> str:
    out = Path(os.getenv("PIPELINE_REPORTS_DIR", "reports")) / f"run_{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": run_id, **sections}, indent=2, default=str))
    return str(out)
