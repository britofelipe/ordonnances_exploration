#!/usr/bin/env python3
# train_flan_t5_small.py
"""
Fine-tune google/flan-t5-small on OCR -> FHIR JSON pairs.
Usage: python train_flan_t5_small.py
You can pass optional args (see argparse defaults).
"""
import json
from pathlib import Path
import argparse
import numpy as np
import logging
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Helpers: load one pair
# -------------------------
def load_one_pair(txt_path: Path):
    """
    Given path/to/ordo_0001.txt, expects corresponding ordo_0001.fhir.json
    Returns dict with 'input_text' and 'target_text' (canonical JSON string).
    """
    json_path = txt_path.with_suffix(".fhir.json")
    with open(txt_path, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    with open(json_path, "r", encoding="utf-8") as f:
        fhir = json.load(f)

    # canonical JSON serialization to give model consistent targets
    target_text = json.dumps(
        fhir, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {"input_text": input_text, "target_text": target_text}


# -------------------------
# Preprocessing function
# -------------------------
def preprocess_batch(batch, tokenizer, max_input_len, max_target_len):
    # tokenizer for inputs (encoder)
    inputs = tokenizer(
        batch["input_text"],
        max_length=max_input_len,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )
    # tokenizer for targets (decoder)
    # for T5, using tokenizer as target tokenizer is fine; HF supports it
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            max_length=max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    label_ids = labels["input_ids"]

    # replace pad token id's in labels by -100 so they are ignored by loss
    label_ids = [
        [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
        for seq in label_ids
    ]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}


# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    # if preds are logits, Trainer returns generated ids when predict_with_generate=True,
    # so here preds should be IDs already (but robust against both)
    if isinstance(preds, tuple) or (not isinstance(preds, np.ndarray) and hasattr(preds, "shape")):
        # unlikely; keep for safety
        preds = preds[0]

    # replace -100 in labels back to pad_token_id to decode
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    pred_strs = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # exact match
    exact = [int(p == t) for p, t in zip(pred_strs, label_strs)]
    exact_match = float(np.mean(exact))

    # json validity rate
    valid = []
    for p in pred_strs:
        try:
            json.loads(p)
            valid.append(1)
        except Exception:
            valid.append(0)
    json_valid_rate = float(np.mean(valid))

    return {"exact_match": exact_match, "json_valid": json_valid_rate}


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Flan-T5-Small on OCR->FHIR pairs")
    parser.add_argument("--data_dir", type=str, default="output_mimic_fhir_ocr",
                        help="Directory containing .txt and .fhir.json pairs")
    parser.add_argument("--output_dir", type=str, default="./flan_t5_small_out",
                        help="Where to save trained model and tokenizer")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 (mixed precision) if supported")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 (mixed precision) if supported")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"{data_dir} does not exist"

    # collect pairs
    all_pairs = []
    for txt_file in sorted(data_dir.glob("*.txt")):
        # require that corresponding .fhir.json exists
        if not txt_file.with_suffix(".fhir.json").exists():
            logger.warning("Skipping %s because %s not found", txt_file.name, txt_file.with_suffix(".fhir.json").name)
            continue
        all_pairs.append(load_one_pair(txt_file))

    logger.info("Loaded %d (input,target) pairs from %s", len(all_pairs), data_dir)

    if len(all_pairs) == 0:
        raise SystemExit("No data found. Exiting.")

    # HF Dataset
    dataset = Dataset.from_list(all_pairs)
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]
    logger.info("Train size: %d, Eval size: %d", len(train_ds), len(eval_ds))

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # important T5 settings
    # decoder_start_token_id: T5 expects the model config to have this; default often exists but ensure:
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id or model.config.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Preprocess datasets
    preprocess_fn = lambda batch: preprocess_batch(batch, tokenizer, args.max_input_len, args.max_target_len)
    tokenized_train = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names)

    # Data collator (handles padding dynamically for seq2seq)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_dir=f"{args.output_dir}/logs",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="json_valid",
        greater_is_better=True,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    # Train
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model and tokenizer saved to %s", args.output_dir)

    # quick inference demo on a few eval examples
    logger.info("Running quick inference on 3 eval examples")
    for i in range(min(3, len(eval_ds))):
        sample = eval_ds[i]
        inputs = tokenizer(sample["input_text"], return_tensors="pt", truncation=True, max_length=args.max_input_len).to(trainer.model.device)
        with torch.no_grad():
            gen_ids = trainer.model.generate(**inputs, max_length=args.max_target_len, num_beams=4)
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        logger.info("=== EXAMPLE %d ===", i)
        logger.info("INPUT (truncated): %s", sample["input_text"][:200].replace("\n", " "))
        logger.info("PREDICT: %s", gen_text[:400])
        try:
            parsed = json.loads(gen_text)
            logger.info("PREDICT JSON valid: True")
        except Exception:
            logger.info("PREDICT JSON valid: False")

if __name__ == "__main__":
    main()
