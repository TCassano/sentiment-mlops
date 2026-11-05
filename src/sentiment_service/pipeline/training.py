"""Fine-tuning of the current model on newly labelled data.

A deliberately small, dependency-free training loop (no Trainer/accelerate): the goal
is a reproducible retraining step the DAG can call, not state-of-the-art training.
"""

from __future__ import annotations

from pathlib import Path

from sentiment_service.model import LABELS, preprocess


def fine_tune(
    base_model: str,
    texts: list[str],
    labels: list[str],
    output_dir: str | Path,
    epochs: int = 1,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model).to(device)
    label2id = {label: i for i, label in enumerate(LABELS)}
    y = torch.tensor([label2id[label] for label in labels])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(texts))
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size].tolist()
            enc = tokenizer(
                [preprocess(texts[i]) for i in idx],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            loss = model(**enc, labels=y[idx].to(device)).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
