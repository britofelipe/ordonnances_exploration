"""
Fine-tuning CamemBERT-bio for Named Entity Recognition
Medical prescription NER: Drug, Strength, Frequency, etc.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


# ── 1. CONFIG ──────────────────────────────────────────────────────────────────

@dataclass
class NERConfig:
    model_name: str = "almanach/camembert-bio-base"
    data_path: str = "data.jsonl"           # your .jsonl file
    output_dir: str = "./camembert-ner-output"
    max_length: int = 128

    # Training
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Data split
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42

    # Misc
    fp16: bool = True                       # safe to enable on modern GPUs
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"


cfg = NERConfig()


# ── 2. LOAD & SPLIT DATA ───────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_label_mappings(records: list[dict]) -> tuple[dict, dict]:
    """Collect all unique NER tags and build id <-> label maps."""
    labels = sorted({tag for r in records for tag in r["ner_tags"]})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def split_dataset(records: list[dict], cfg: NERConfig) -> DatasetDict:
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(records, test_size=cfg.test_size, random_state=cfg.seed)
    # val_size is relative to the remaining data
    relative_val = cfg.val_size / (1 - cfg.test_size)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=cfg.seed)

    def to_dataset(recs):
        return Dataset.from_dict({
            "tokens":   [r["tokens"]   for r in recs],
            "ner_tags": [r["ner_tags"] for r in recs],
        })

    return DatasetDict({"train": to_dataset(train),
                        "validation": to_dataset(val),
                        "test": to_dataset(test)})


# ── 3. TOKENIZATION & LABEL ALIGNMENT ─────────────────────────────────────────

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length):
    """
    Tokenize word-level tokens and align BIO labels to subword tokens.

    Strategy:
      - First subword of a word  → inherits the original label
      - Subsequent subwords      → labeled -100 (ignored by CrossEntropyLoss)
      - Special tokens ([CLS], [SEP], [PAD]) → -100
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,   # crucial: input is already split into words
    )

    aligned_labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                # Special token
                label_ids.append(-100)
            elif word_id != prev_word_id:
                # First subword of a new word → assign real label
                label_ids.append(label2id[label_seq[word_id]])
            else:
                # Continuation subword → ignore
                label_ids.append(-100)
            prev_word_id = word_id
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


# ── 4. METRICS ─────────────────────────────────────────────────────────────────

def build_compute_metrics(id2label):
    """Returns a compute_metrics fn that uses seqeval (entity-level F1)."""

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            true_label_row, true_pred_row = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:           # skip special / continuation tokens
                    continue
                true_label_row.append(id2label[l])
                true_pred_row.append(id2label[p])
            true_labels.append(true_label_row)
            true_preds.append(true_pred_row)

        return {
            "f1":        f1_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall":    recall_score(true_labels, true_preds),
        }

    return compute_metrics


# ── 5. MAIN ────────────────────────────────────────────────────────────────────

def main():
    # --- Load data
    print("Loading data...")
    records = load_jsonl(cfg.data_path)
    label2id, id2label = build_label_mappings(records)
    print(f"Labels ({len(label2id)}): {list(label2id.keys())}")

    dataset = split_dataset(records, cfg)
    print(f"Splits → train: {len(dataset['train'])}  "
          f"val: {len(dataset['validation'])}  "
          f"test: {len(dataset['test'])}")

    # --- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    tokenize_fn = lambda examples: tokenize_and_align_labels(
        examples, tokenizer, label2id, cfg.max_length
    )

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )

    # --- Model
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # classifier head is re-initialized
    )

    # --- Training args
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        save_total_limit=cfg.save_total_limit,

        report_to="none",               # swap for "wandb" / "tensorboard" if needed
        seed=cfg.seed,
    )

    # --- Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(id2label),
    )

    # --- Train
    print("\nStarting training...")
    trainer.train()

    # --- Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.predict(tokenized_dataset["test"])
    predictions = np.argmax(test_results.predictions, axis=-1)
    labels_array = test_results.label_ids

    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, labels_array):
        tl, tp = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            tl.append(id2label[l])
            tp.append(id2label[p])
        true_labels.append(tl)
        true_preds.append(tp)

    print("\n── Test Set Report ──────────────────────────────────")
    print(classification_report(true_labels, true_preds))

    # --- Save best model
    best_path = Path(cfg.output_dir) / "best_model"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    print(f"\nBest model saved to: {best_path}")


if __name__ == "__main__":
    main()