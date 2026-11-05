"""Data access for evaluation and retraining.

Public test data: ``tweet_eval/sentiment`` (SemEval-2017 Task 4), the same label
space as the model: 0 = negative, 1 = neutral, 2 = positive.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment_service.model import LABELS

DATASET = "cardiffnlp/tweet_eval"
SUBSET = "sentiment"


def load_tweet_eval(split: str = "test", sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with columns ``text`` and ``label`` (string label)."""
    from datasets import load_dataset

    df = load_dataset(DATASET, SUBSET, split=split).to_pandas()
    df["label"] = df["label"].map(dict(enumerate(LABELS)))
    if sample is not None and sample < len(df):
        # Stratified sample: keeps the class balance of the full split.
        df = df.groupby("label", group_keys=False).sample(frac=sample / len(df), random_state=seed)
    return df.reset_index(drop=True)[["text", "label"]]


def collect_batch(output_dir: str | Path, run_id: str, sample: int = 500, seed: int = 0) -> Path:
    """Simulate the periodic collection of freshly labelled social posts.

    In production this step would pull the posts of the period from the social
    APIs / data lake, joined with the labels produced by human annotators on a
    random sample. Here we draw from the tweet_eval train split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_tweet_eval("train", sample=sample, seed=seed)
    path = output_dir / f"batch_{run_id}.parquet"
    df.to_parquet(path, index=False)
    return path


def validate_batch(df: pd.DataFrame, min_rows: int = 50) -> list[str]:
    """Basic data-quality checks; returns the list of problems (empty = OK)."""
    problems = []
    if len(df) < min_rows:
        problems.append(f"too few rows: {len(df)} < {min_rows}")
    if missing := {"text", "label"} - set(df.columns):
        problems.append(f"missing columns: {sorted(missing)}")
        return problems
    if df["text"].isna().any() or (df["text"].str.strip() == "").any():
        problems.append("empty texts present")
    if unknown := set(df["label"].dropna()) - set(LABELS):
        problems.append(f"unknown labels: {sorted(unknown)}")
    return problems
