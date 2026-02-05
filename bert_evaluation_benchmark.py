import json
import re
import time
import os
from pathlib import Path
from typing import Optional, List
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# =========================================================
# TOGGLE CPU IF GPU FAILS (Uncomment below if you get CUDA errors)
# =========================================================
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Fix for parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================
# IMPORTS FROM YOUR GENERATOR
# =============================
from generate_ordo_mimic import Posology, LineItem, OrdoDoc, to_fhir_bundle

# =============================
# 1) Helpers: FHIR <-> DSL
# =============================

def ordo_to_linear_text(doc: OrdoDoc) -> str:
    lines = []
    lines.append("ORDO")
    lines.append(f"PATIENT: {doc.patient_name}")
    lines.append(f"PRESCRIPTEUR: {doc.prescriber_name}")
    lines.append(f"DATE: {doc.date_str}")
    lines.append("")
    for li in doc.lines:
        lines.append("MED_START")
        lines.append(f"DRUG: {li.drug_name}")
        if li.strength:
            lines.append(f"STRENGTH: {li.strength}")
        if li.posology.form:
            lines.append(f"FORM: {li.posology.form}")
        if li.posology.route:
            lines.append(f"ROUTE: {li.posology.route}")
        if li.posology.dose:
            lines.append(f"DOSE: {li.posology.dose}")
        if li.posology.frequency:
            lines.append(f"FREQ: {li.posology.frequency}")
        if li.posology.duration:
            lines.append(f"DURATION: {li.posology.duration}")
        if li.refills is not None:
            lines.append(f"REFILLS: {li.refills}")
        lines.append("MED_END")
        lines.append("")
    lines.append("END")
    return "\n".join(lines)

def fhir_bundle_to_ordo(bundle: dict) -> OrdoDoc:
    entries = bundle.get("entry", [])
    patient_name = "UNKNOWN"
    prescriber_name = "UNKNOWN"
    date_str = "1900-01-01"
    lines: List[LineItem] = []

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
        drug_name = med_text
        strength = ""
        if "(" in med_text and med_text.endswith(")"):
            i = med_text.rfind("(")
            drug_name = med_text[:i].strip()
            strength = med_text[i+1:-1].strip()

        di = (mr.get("dosageInstruction") or [{}])[0]
        route = di.get("route", {}).get("text", "")

        poso = Posology(dose="", frequency="", duration="", route=route, form="")

        if "doseAndRate" in di and di["doseAndRate"]:
            d0 = di["doseAndRate"][0]
            ds = d0.get("doseString", "")
            if ds:
                poso.dose = ds

        rep = di.get("timing", {}).get("repeat", {})
        if "frequency" in rep:
            poso.frequency = f"{int(rep['frequency'])}/j"
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
                refills = None

        lines.append(LineItem(drug_name=drug_name, strength=strength, posology=poso, refills=refills))

    return OrdoDoc(patient_name=patient_name, prescriber_name=prescriber_name, date_str=date_str, lines=lines)

# =============================
# 2) ROBUSTO DSL Parser
# =============================

FIELD_KEYS = ["DRUG", "STRENGTH", "FORM", "ROUTE", "DOSE", "FREQ", "DURATION", "REFILLS"]
FIELD_RE = r"(?:%s):" % "|".join(FIELD_KEYS)

def _extract_one(text: str, key: str) -> str:
    pattern = rf"{key}:\s*(.*?)(?=\s+(?:{FIELD_RE}|\bMED_START\b|\bMED_END\b|\bEND\b)|\s*$)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

def normalize_dsl(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    # 1) REPAIR tokens broken by the model
    t = re.sub(r"MED_\s*END", "MED_END", t, flags=re.IGNORECASE)

    # 2) Place markers on new lines
    # FIX: Removed trailing \b from keys ending in : because "KEY: Value" 
    # has no word boundary between : and space.
    markers = [
        r"\bORDO\b",
        r"\bMED_START\b",
        r"\bMED_END\b",
        r"\bEND\b",
        r"\bPATIENT:",       # Changed from \bPATIENT:\b
        r"\bPRESCRIPTEUR:",  # Changed from \bPRESCRIPTEUR:\b
        r"\bDATE:",          # Changed from \bDATE:\b
        r"\bDRUG:",          # Changed from \bDRUG:\b
        r"\bSTRENGTH:",      # Changed from \bSTRENGTH:\b
        r"\bFORM:",          # ...
        r"\bROUTE:",
        r"\bDOSE:",
        r"\bFREQ:",
        r"\bDURATION:",
        r"\bREFILLS:",
    ]

    for pat in markers:
        # We add a space after the marker in the replacement just in case
        t = re.sub(rf"\s*({pat})\s*", r"\n\1 ", t, flags=re.IGNORECASE)

    # 3) Clean duplicate spaces and excess empty lines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()

def _clean_header_value(v: str) -> str:
    if not v:
        return ""
    v = re.split(r"\bPRESCRIPTEUR:\b|\bPATIENT:\b|\bDATE:\b|\bMED_START\b|\bEND\b", v, maxsplit=1)[0]
    return v.strip()

def linear_text_to_ordo_robust(text: str) -> OrdoDoc:
    dsl = normalize_dsl(text)
    patient = "UNKNOWN"
    prescriber = "UNKNOWN"
    date_str = "1900-01-01"

    lines = [ln.strip() for ln in dsl.splitlines() if ln.strip()]
    idx = 0
    while idx < len(lines):
        ln = lines[idx]
        if ln.upper().startswith("PATIENT:"):
            patient = _clean_header_value(ln.split(":", 1)[1])
        elif ln.upper().startswith("PRESCRIPTEUR:"):
            prescriber = _clean_header_value(ln.split(":", 1)[1])
        elif ln.upper().startswith("DATE:"):
            date_str = _clean_header_value(ln.split(":", 1)[1])
        if ln.upper().startswith("MED_START"):
            break
        idx += 1

    t = " ".join(dsl.replace("\n", " ").split())
    meds = []
    for block in re.findall(r"\bMED_START\b\s*(.*?)\s*(?:\bMED_END\b|MED_\s*END)", t, flags=re.IGNORECASE | re.DOTALL):
        drug = _extract_one(block, "DRUG")
        if not drug: continue
        strength = _extract_one(block, "STRENGTH")
        form = _extract_one(block, "FORM")
        route = _extract_one(block, "ROUTE")
        dose = _extract_one(block, "DOSE")
        freq = _extract_one(block, "FREQ")
        dur = _extract_one(block, "DURATION")
        refills = None
        ref = _extract_one(block, "REFILLS")
        if ref:
            try: refills = int(float(ref))
            except: refills = None
        poso = Posology(dose=dose, frequency=freq, duration=dur, route=route, form=form)
        meds.append(LineItem(drug_name=drug, strength=strength, posology=poso, refills=refills))

    return OrdoDoc(patient_name=patient, prescriber_name=prescriber, date_str=date_str, lines=meds)

def canon_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def safe_dsl_to_fhir(dsl_text: str, bundle_id="eval") -> dict:
    dsl_text = normalize_dsl(dsl_text)
    doc = linear_text_to_ordo_robust(dsl_text)
    return to_fhir_bundle(doc, bundle_id=bundle_id)

# =============================
# 3) Load pairs
# =============================

DATA_DIR = Path("output_mimic_fhir_ocr_template_demo_simplified") 

def load_one_pair(txt_path: Path):
    json_path = txt_path.with_suffix(".fhir.json")
    with open(txt_path, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    with open(json_path, "r", encoding="utf-8") as f:
        fhir = json.load(f)
    doc_gt = fhir_bundle_to_ordo(fhir)
    target_dsl = ordo_to_linear_text(doc_gt)
    return {"input_text": input_text, "target_text": target_dsl}

all_pairs = [load_one_pair(p) for p in DATA_DIR.glob("*.txt")]
print(len(all_pairs), "total examples")

dataset = Dataset.from_list(all_pairs)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
test_ds = dataset["test"]
# Replicate train split to ensure we have same test distribution
train_val = dataset["train"]
train_val = train_val.train_test_split(test_size=0.1, seed=42)
train_ds = train_val["train"]
val_ds = train_val["test"]
print("train:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))

# =============================
# 4) Load checkpoint
# =============================

output_dir = "./toobib-ordo-bert2bert-template-simplified"
last_checkpoint = get_last_checkpoint(output_dir)
last_checkpoint = "./toobib-ordo-bert2bert-prod"
if last_checkpoint is None:
    raise ValueError(f"No checkpoint found in {output_dir}")

print("Using checkpoint:", last_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
model = EncoderDecoderModel.from_pretrained(last_checkpoint)
model.eval()

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 512

def preprocess_batch(batch):
    inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    )
    # FIX: Use text_target instead of context manager
    labels = tokenizer(
        text_target=batch["target_text"], 
        max_length=MAX_TARGET_LEN,
        padding="max_length",
        truncation=True,
    )

    label_ids = labels["input_ids"]
    label_ids = [
        [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
        for seq in label_ids
    ]
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": label_ids,
    }

tokenized_test = test_ds.map(preprocess_batch, batched=True, remove_columns=test_ds.column_names)

# =============================
# 5) Metrics
# =============================

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    vocab_size = tokenizer.vocab_size
    preds = np.where((preds < 0) | (preds >= vocab_size), tokenizer.pad_token_id, preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    pred_raw = tokenizer.batch_decode(preds, skip_special_tokens=True)
    gt_raw   = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact_dsl = []
    exact_fhir = []

    for p_txt, t_txt in zip(pred_raw, gt_raw):
        p_norm = normalize_dsl(p_txt)
        t_norm = normalize_dsl(t_txt)
        exact_dsl.append(int(p_norm.strip() == t_norm.strip()))
        try:
            fhir_p = safe_dsl_to_fhir(p_norm, bundle_id="pred")
            fhir_t = safe_dsl_to_fhir(t_norm, bundle_id="gt")
            exact_fhir.append(int(canon_no_ids(fhir_p) == canon_no_ids(fhir_t)))
        except Exception:
            exact_fhir.append(0)

    return {
        "dsl_exact_match": float(np.mean(exact_dsl)),
        "fhir_exact_match": float(np.mean(exact_fhir)),
    }

def strip_ids_from_bundle(bundle: dict) -> dict:
    b = json.loads(json.dumps(bundle))
    b.pop("id", None)
    for e in b.get("entry", []):
        r = e.get("resource", {})
        if isinstance(r, dict):
            r.pop("id", None)
    return b

def canon_no_ids(bundle: dict) -> str:
    return json.dumps(strip_ids_from_bundle(bundle), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    report_to="none", # Disables wandb/mlflow reporting to keep it clean
)

# FIX: Use processing_class instead of tokenizer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    processing_class=tokenizer, 
    compute_metrics=compute_metrics,
)

print("== Final Evaluation on TEST set ==")
#test_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
#print(test_metrics)

# =============================
# 6) Debug: Examples
# =============================
num_examples = 5
rng = np.random.default_rng(0)
indices = rng.choice(len(test_ds), size=min(num_examples, len(test_ds)), replace=False)
device = model.device

print("\n== Examples ==")
for k, idx in enumerate(indices, start=1):
    sample = test_ds[int(idx)]
    input_text = sample["input_text"]
    target_dsl = sample["target_text"]

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
    inputs = {kk: vv.to(device) for kk, vv in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_length=MAX_TARGET_LEN, num_beams=4)

    pred_dsl_raw = tokenizer.decode(gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Normalize
    pred_dsl = normalize_dsl(pred_dsl_raw)
    gt_dsl   = normalize_dsl(target_dsl)
    
    # Convert to FHIR
    pred_fhir = safe_dsl_to_fhir(pred_dsl, bundle_id="pred")
    gt_fhir   = safe_dsl_to_fhir(gt_dsl, bundle_id="gt")

    print(f"\n----- Example {k} (idx={idx}) -----")
    print("INPUT (OCR):")
    print(input_text)
    print("\nGT DSL:")
    print(target_dsl[:500]) # Print first 500 chars
    print("\nPRED DSL:")
    print(pred_dsl[:500])

    # Print JSONs to see why they don't match
    print("\nGT FHIR (canon head):")
    print(json.dumps(gt_fhir, ensure_ascii=False, indent=2)[:500] + "...") 
    print("\nPRED FHIR (canon head):")
    print(json.dumps(pred_fhir, ensure_ascii=False, indent=2)[:500] + "...")

    is_match = int(canon_no_ids(pred_fhir) == canon_no_ids(gt_fhir))
    print(f"\nFHIR EXACT MATCH: {is_match}")
    print("-" * 80)

# =============================
# 7) Benchmark: Latency & Memory
# =============================
print("\n== Starting Benchmark (Time & Memory) ==")
BENCHMARK_SAMPLES = min(len(test_ds), 100)
print(f"Benchmarking on {BENCHMARK_SAMPLES} samples (Batch Size = 1)...")

benchmark_indices = range(BENCHMARK_SAMPLES)
benchmark_data = test_ds.select(benchmark_indices)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

# Warmup
warmup_input = tokenizer(benchmark_data[0]["input_text"], return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
warmup_input = {k: v.to(device) for k, v in warmup_input.items()}
with torch.no_grad():
    _ = model.generate(**warmup_input, max_length=MAX_TARGET_LEN, num_beams=4)

latencies = []
start_time_total = time.perf_counter()

for i, item in enumerate(benchmark_data):
    inputs = tokenizer(item["input_text"], return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**inputs, max_length=MAX_TARGET_LEN, num_beams=4)
    t1 = time.perf_counter()
    latencies.append(t1 - t0)

end_time_total = time.perf_counter()
total_duration = end_time_total - start_time_total

avg_latency = np.mean(latencies)
std_latency = np.std(latencies)
throughput = BENCHMARK_SAMPLES / total_duration

print(f"\nBenchmark Results ({BENCHMARK_SAMPLES} samples):")
print(f"Avg Latency:    {avg_latency:.4f} s (+/- {std_latency:.4f})")
print(f"Throughput:     {throughput:.2f} seq/s")

if torch.cuda.is_available():
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak GPU Mem:   {max_memory:.2f} GB")
else:
    print("Peak GPU Mem:   N/A (CPU)")