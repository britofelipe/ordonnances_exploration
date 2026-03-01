#!/usr/bin/env python3

import json
import argparse
import logging
from pathlib import Path
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from generate_ordo_mimic import Posology, LineItem, OrdoDoc, to_fhir_bundle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# DSL helpers
# =============================

def ordo_to_linear_text(doc: OrdoDoc) -> str:
    lines = [
        "ORDO",
        f"PATIENT: {doc.patient_name}",
        f"PRESCRIPTEUR: {doc.prescriber_name}",
        f"DATE: {doc.date_str}",
        ""
    ]

    for li in doc.lines:
        lines.append("MED_START")
        lines.append(f"DRUG: {li.drug_name}")
        if li.strength: lines.append(f"STRENGTH: {li.strength}")
        if li.posology.form: lines.append(f"FORM: {li.posology.form}")
        if li.posology.route: lines.append(f"ROUTE: {li.posology.route}")
        if li.posology.dose: lines.append(f"DOSE: {li.posology.dose}")
        if li.posology.frequency: lines.append(f"FREQ: {li.posology.frequency}")
        if li.posology.duration: lines.append(f"DURATION: {li.posology.duration}")
        if li.refills is not None: lines.append(f"REFILLS: {li.refills}")
        lines.append("MED_END")
        lines.append("")

    lines.append("END")
    return "\n".join(lines)


def fhir_bundle_to_ordo(bundle: dict) -> OrdoDoc:
    entries = bundle.get("entry", [])
    patient_name, prescriber_name, date_str = "UNKNOWN", "UNKNOWN", "1900-01-01"
    lines = []

    for e in entries:
        mr = e.get("resource", {})
        if mr.get("resourceType") != "MedicationRequest":
            continue

        if patient_name == "UNKNOWN":
            patient_name = mr.get("subject", {}).get("display", "UNKNOWN")

        if prescriber_name == "UNKNOWN":
            prescriber_name = mr.get("requester", {}).get("display", "UNKNOWN")

        if date_str == "1900-01-01":
            date_str = mr.get("authoredOn", "1900-01-01")

        med_text = mr.get("medicationCodeableConcept", {}).get("text", "")
        drug_name, strength = med_text, ""

        if "(" in med_text and med_text.endswith(")"):
            i = med_text.rfind("(")
            drug_name = med_text[:i].strip()
            strength = med_text[i+1:-1].strip()

        di = (mr.get("dosageInstruction") or [{}])[0]

        poso = Posology(
            dose="",
            frequency="",
            duration="",
            route=di.get("route", {}).get("text", ""),
            form="",
            as_needed=False,
            as_needed_for=""
        )

        d0 = (di.get("doseAndRate") or [{}])[0]
        poso.dose = d0.get("doseString", "")

        rep = di.get("timing", {}).get("repeat", {})
        if "frequency" in rep:
            poso.frequency = f"{rep['frequency']}/j"

        if "boundsDuration" in rep:
            bd = rep["boundsDuration"]
            val = bd.get("value")
            if val is not None:
                poso.duration = f"{int(val)} jours"

        refills = None
        disp = mr.get("dispenseRequest", {})
        if "numberOfRepeatsAllowed" in disp:
            try:
                refills = int(disp["numberOfRepeatsAllowed"])
            except Exception:
                pass

        lines.append(LineItem(
            drug_name=drug_name,
            strength=strength,
            posology=poso,
            refills=refills
        ))

    return OrdoDoc(
        patient_name=patient_name,
        prescriber_name=prescriber_name,
        date_str=date_str,
        lines=lines
    )


def load_one_pair(txt_path: Path):
    json_path = txt_path.with_suffix(".fhir.json")

    with open(txt_path, encoding="utf-8") as f:
        input_text = f.read().strip()

    with open(json_path, encoding="utf-8") as f:
        fhir = json.load(f)

    doc = fhir_bundle_to_ordo(fhir)
    target_text = ordo_to_linear_text(doc)

    return {"input_text": input_text, "target_text": target_text}

# =============================
# Preprocessing
# =============================

def preprocess_batch(batch, tokenizer, max_input_len=512, max_target_len=512):
    inputs = tokenizer(
        batch["input_text"],
        max_length=max_input_len,
        padding="max_length",
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            max_length=max_target_len,
            padding="max_length",
            truncation=True,
        )

    label_ids = [
        [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
        for seq in labels["input_ids"]
    ]

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": label_ids,
    }

# =============================
# Metrics
# =============================

def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    pred_strs = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact = [int(p == t) for p, t in zip(pred_strs, label_strs)]
    return {"exact_match": float(np.mean(exact))}

# =============================
# Main
# =============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    pairs = [load_one_pair(p) for p in sorted(data_dir.glob("*.txt"))]

    dataset = Dataset.from_list(pairs)

    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    test_ds = dataset["test"]

    train_val = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
    train_ds = train_val["train"]
    val_ds = train_val["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    preprocess_fn = lambda batch: preprocess_batch(batch, tokenizer)

    tokenized_train = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(preprocess_fn, batched=True, remove_columns=val_ds.column_names)
    tokenized_test = test_ds.map(preprocess_fn, batched=True, remove_columns=test_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        predict_with_generate=True,
        generation_max_length=512,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        save_total_limit=3,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # =============================
    # Test evaluation + save results
    # =============================

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_test,
        metric_key_prefix="test"
    )

    with open(output_path / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    predictions = trainer.predict(tokenized_test)
    pred_ids = predictions.predictions
    label_ids = predictions.label_ids

    label_ids = np.where(label_ids != -100, tokenizer.pad_token_id, label_ids)

    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    with open(output_path / "test_predictions.jsonl", "w", encoding="utf-8") as f:
        for p, r in zip(pred_texts, label_texts):
            f.write(json.dumps({"prediction": p, "reference": r}, ensure_ascii=False) + "\n")

    with open(output_path / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()