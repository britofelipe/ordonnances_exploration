"""
Evaluate a fine-tuned CamemBERT-bio NER model on Posos/MedNERF (HuggingFace).

Training labels (from generate_encoder_dataset.py) are an exact match with
MedNERF's label set:
  O, B-Drug, I-Drug, B-Dosage, I-Dosage, B-Duration, I-Duration,
  B-Form, I-Form, B-Frequency, I-Frequency, B-Strength, I-Strength

Usage
-----
  # Minimal
  python evaluate_on_mednerf.py --model_path ./camembertBIO-ner-output/best_model

  # Save full results to a folder
  python evaluate_on_mednerf.py --model_path ./camembertBIO-ner-output/best_model \\
      --output_dir ./mednerf_results

  # Also print mis-classified sentences
  python evaluate_on_mednerf.py --model_path ./camembertBIO-ner-output/best_model \\
      --show_errors
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# ── 1. ARGUMENT PARSING ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned CamemBERT-bio NER model on Posos/MedNERF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the fine-tuned model directory (or a HuggingFace model id)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on. MedNERF only exposes a single split.",
    )
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help="Max subword sequence length (should match the value used during training)",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Inference batch size",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="If set, write the seqeval report, summary JSON, and per-sample JSONL here",
    )
    parser.add_argument(
        "--show_errors",
        action="store_true",
        help="Print sentences where at least one token was mis-classified",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override: 'cpu', 'cuda', 'cuda:1', etc. Auto-detected if not set.",
    )
    parser.add_argument(
        "--dataset_name",
        default="Posos/MedNERF",
        help="HuggingFace dataset identifier",
    )
    return parser.parse_args()


# ── 2. LABEL COMPATIBILITY CHECK ───────────────────────────────────────────────

# Entity types shared by MedNERF and generate_encoder_dataset.py
EXPECTED_ENTITY_TYPES = {"Drug", "Strength", "Form", "Dosage", "Duration", "Frequency"}

EXPECTED_LABELS = (
    ["O"]
    + [
        f"{prefix}-{etype}"
        for etype in sorted(EXPECTED_ENTITY_TYPES)
        for prefix in ("B", "I")
    ]
)


def check_label_compatibility(model_labels: list) -> None:
    """
    Verify that the loaded model covers all MedNERF entity types.
    Because generate_encoder_dataset.py uses the same 13-label BIO scheme as
    MedNERF this check should always pass with a clean fine-tuned model.
    """
    model_label_set = set(model_labels)
    missing = [lbl for lbl in EXPECTED_LABELS if lbl not in model_label_set]

    if missing:
        warnings.warn(
            f"\n  The following MedNERF labels are MISSING from the model label set:\n"
            f"    {missing}\n"
            f"  Scores for those entity types will be 0.\n"
            f"  Model labels: {sorted(model_label_set)}",
            stacklevel=2,
        )
    else:
        print(
            f"  ✓ Label compatibility check passed "
            f"({len(EXPECTED_LABELS)} labels, all present in the model)."
        )


# ── 3. DATA LOADING ────────────────────────────────────────────────────────────

def load_mednerf(dataset_name: str, split: str):
    """
    Load MedNERF from HuggingFace and return (tokens_list, ner_tags_list).

    MedNERF ships with a single split that HuggingFace may expose as 'train'
    regardless of the name requested — we fall back gracefully.
    """
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception:
        print(f"  Split '{split}' not found — falling back to 'train' split.")
        ds = load_dataset(dataset_name, split="train")

    print(f"  Loaded {len(ds)} examples from '{dataset_name}'.")
    print(f"  Columns: {list(ds[0].keys())}")

    first_tags = ds[0]["ner_tags"]
    if isinstance(first_tags[0], int):
        # Tags stored as integer IDs — resolve using dataset feature metadata
        tag_names = ds.features["ner_tags"].feature.names
        tokens_list = [ex["tokens"] for ex in ds]
        tags_list = [[tag_names[t] for t in ex["ner_tags"]] for ex in ds]
    else:
        tokens_list = [ex["tokens"] for ex in ds]
        tags_list = [ex["ner_tags"] for ex in ds]

    unique_tags = sorted({tag for seq in tags_list for tag in seq})
    print(f"  Unique NER tags in dataset : {unique_tags}")
    print(f"  Example tokens  : {tokens_list[0][:8]}")
    print(f"  Example ner_tags: {tags_list[0][:8]}")
    return tokens_list, tags_list


# ── 4. INFERENCE ───────────────────────────────────────────────────────────────

def predict_ner(
    model,
    tokenizer,
    tokens_list: list,
    id2label: dict,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list:
    """
    Run token classification on word-level token lists.

    Strategy: for each word, the label predicted for its *first* subword token
    is used (identical to the alignment used during training).
    Returns one predicted label string per word.
    """
    model.eval()
    all_preds = []

    for batch_start in range(0, len(tokens_list), batch_size):
        batch_tokens = tokens_list[batch_start : batch_start + batch_size]

        # Batch encode
        encoding = tokenizer(
            batch_tokens,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            is_split_into_words=True,
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)

        pred_ids = torch.argmax(outputs.logits, dim=-1).cpu().numpy()  # (B, seq_len)

        for j, word_tokens in enumerate(batch_tokens):
            # Re-encode individually to recover word_ids alignment
            single_enc = tokenizer(
                word_tokens,
                truncation=True,
                max_length=max_length,
                is_split_into_words=True,
            )
            word_ids_map = single_enc.word_ids()   # list[int | None]
            preds_for_sample = pred_ids[j]         # (seq_len,)

            word_pred: dict = {}
            for subword_idx, word_id in enumerate(word_ids_map):
                if word_id is None:
                    continue
                if word_id not in word_pred:       # first subword wins
                    word_pred[word_id] = id2label[preds_for_sample[subword_idx]]

            # One label per original word; default to "O" for any gap
            sentence_preds = [
                word_pred.get(k, "O") for k in range(len(word_tokens))
            ]
            all_preds.append(sentence_preds)

        n_done = min(batch_start + batch_size, len(tokens_list))
        if (batch_start // batch_size) % 5 == 0:
            print(f"  Processed {n_done}/{len(tokens_list)} examples...")

    return all_preds


# ── 5. ERROR ANALYSIS ──────────────────────────────────────────────────────────

def print_errors(
    tokens_list: list,
    gold_list: list,
    pred_list: list,
    max_show: int = 20,
) -> None:
    """Print sentences with at least one mis-classified token."""
    n_shown = 0
    for tokens, gold, pred in zip(tokens_list, gold_list, pred_list):
        if gold == pred:
            continue
        print(f"\n{'─' * 65}")
        for tok, g, p in zip(tokens, gold, pred):
            marker = "  ✗" if g != p else "   "
            print(f"{marker}  {tok:<25}  gold={g:<15}  pred={p}")
        n_shown += 1
        if n_shown >= max_show:
            print(f"\n  (showing first {max_show} error sentences — use --show_errors to see all)")
            break


# ── 6. MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  Device  : {device}")
    print(f"  Model   : {args.model_path}")
    print(f"  Dataset : {args.dataset_name}  (split='{args.split}')")
    print(f"{'='*65}\n")

    # ── Load model & tokenizer ──────────────────────────────────────────────────
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.to(device)

    id2label: dict = model.config.id2label
    print(f"  Model label set ({len(id2label)}): {sorted(id2label.values())}")

    # ── Label compatibility ──────────────────────────────────────────────────────
    print("\nChecking label compatibility with MedNERF...")
    check_label_compatibility(list(id2label.values()))

    # ── Load MedNERF ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.dataset_name}...")
    tokens_list, gold_tags_list = load_mednerf(args.dataset_name, args.split)

    # ── Inference ───────────────────────────────────────────────────────────────
    print(f"\nRunning inference (batch_size={args.batch_size}, max_length={args.max_length})...")
    predictions = predict_ner(
        model=model,
        tokenizer=tokenizer,
        tokens_list=tokens_list,
        id2label=id2label,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )

    # ── Evaluation ──────────────────────────────────────────────────────────────
    print("\n── seqeval Entity-Level Results ─────────────────────────────────\n")
    report = classification_report(gold_tags_list, predictions, digits=4)
    print(report)

    overall_f1        = f1_score(gold_tags_list, predictions)
    overall_precision = precision_score(gold_tags_list, predictions)
    overall_recall    = recall_score(gold_tags_list, predictions)

    print(f"  Precision (micro) : {overall_precision:.4f}")
    print(f"  Recall    (micro) : {overall_recall:.4f}")
    print(f"  F1        (micro) : {overall_f1:.4f}")

    # ── Per-token accuracy ──────────────────────────────────────────────────────
    correct = total = 0
    for gold_seq, pred_seq in zip(gold_tags_list, predictions):
        for g, p in zip(gold_seq, pred_seq):
            total += 1
            correct += int(g == p)
    token_acc = correct / total if total else 0.0
    print(f"  Token accuracy    : {token_acc:.4f}  ({correct}/{total})")

    # ── Error analysis ──────────────────────────────────────────────────────────
    if args.show_errors:
        print("\n── Mis-classified sentences ──────────────────────────────────────")
        print_errors(tokens_list, gold_tags_list, predictions)

    # ── Save results ────────────────────────────────────────────────────────────
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. seqeval report (plain text)
        report_path = out / "seqeval_report.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"\n  seqeval report → {report_path}")

        # 2. Summary metrics (JSON)
        summary = {
            "model":              args.model_path,
            "dataset":            args.dataset_name,
            "split":              args.split,
            "overall_precision":  overall_precision,
            "overall_recall":     overall_recall,
            "overall_f1":         overall_f1,
            "token_accuracy":     token_acc,
            "num_examples":       len(tokens_list),
        }
        summary_path = out / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Summary JSON    → {summary_path}")

        # 3. Per-sample predictions (JSONL)
        predictions_path = out / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as f:
            for i, (toks, gold, pred) in enumerate(
                zip(tokens_list, gold_tags_list, predictions)
            ):
                record = {
                    "id":        i,
                    "tokens":    toks,
                    "gold_tags": gold,
                    "pred_tags": pred,
                    "correct":   gold == pred,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Predictions     → {predictions_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
