"""Evaluate a model on tweet_eval/sentiment and write a JSON report.

Usage: python scripts/evaluate.py --sample 2000 --out reports/eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentiment_service.config import DEFAULT_MODEL
from sentiment_service.model import HFSentimentModel
from sentiment_service.pipeline.data import load_tweet_eval
from sentiment_service.pipeline.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--out", default="reports/eval.json")
    args = parser.parse_args()

    df = load_tweet_eval(args.split, sample=args.sample)
    result = evaluate(HFSentimentModel(args.model), df["text"].tolist(), df["label"].tolist())
    result.pop("predictions")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "split": args.split, **result}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
