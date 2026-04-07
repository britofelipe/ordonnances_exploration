import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from datasets import Dataset

from datasets_mimic.generate_ordo_mimic import LineItem, OrdoDoc, Posology


MED_FIELD_KEYS = ["DRUG", "STRENGTH", "FORM", "ROUTE", "DOSE", "FREQ", "DURATION", "REFILLS"]
HEADER_KEYS = ["PATIENT", "PRESCRIPTEUR", "DATE"]
FIELD_RE = r"(?:%s):" % "|".join(MED_FIELD_KEYS)


@dataclass
class ParsedMed:
    drug: str = ""
    strength: str = ""
    form: str = ""
    route: str = ""
    dose: str = ""
    freq: str = ""
    duration: str = ""
    refills: str = ""


@dataclass
class ParsedDSL:
    patient: str = "UNKNOWN"
    prescripteur: str = "UNKNOWN"
    date: str = "1900-01-01"
    meds: List[ParsedMed] = None

    def __post_init__(self):
        if self.meds is None:
            self.meds = []


def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_value(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    t = _strip_accents(t).lower()
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\bunknown\b", "unknown", t)
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\s*-\s*", " - ", t)
    t = re.sub(r"\s*\(\s*", " (", t)
    t = re.sub(r"\s*\)\s*", ") ", t)
    t = re.sub(r"\s*:\s*", ": ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def normalize_dsl_layout(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"MED_\s*END", "MED_END", t, flags=re.IGNORECASE)
    markers = [
        r"\bORDO\b",
        r"\bMED_START\b",
        r"\bMED_END\b",
        r"\bEND\b",
        r"\bPATIENT:",
        r"\bPRESCRIPTEUR:",
        r"\bDATE:",
        r"\bDRUG:",
        r"\bSTRENGTH:",
        r"\bFORM:",
        r"\bROUTE:",
        r"\bDOSE:",
        r"\bFREQ:",
        r"\bDURATION:",
        r"\bREFILLS:",
    ]
    for pat in markers:
        t = re.sub(rf"\s*({pat})\s*", r"\n\1 ", t, flags=re.IGNORECASE)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _extract_one(text: str, key: str) -> str:
    pattern = rf"{key}:\s*(.*?)(?=\s+(?:{FIELD_RE}|\bMED_START\b|\bMED_END\b|\bEND\b)|\s*$)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_header(lines: List[str], key: str, default: str) -> str:
    prefix = f"{key}:"
    for line in lines:
        if line.upper().startswith(prefix):
            return normalize_value(line.split(":", 1)[1]) or default
    return default


def parse_dsl_robust(text: str) -> ParsedDSL:
    dsl = normalize_dsl_layout(text)
    lines = [ln.strip() for ln in dsl.splitlines() if ln.strip()]

    parsed = ParsedDSL(
        patient=_extract_header(lines, "PATIENT", "unknown"),
        prescripteur=_extract_header(lines, "PRESCRIPTEUR", "unknown"),
        date=_extract_header(lines, "DATE", "1900-01-01"),
        meds=[],
    )

    flat = " ".join(dsl.replace("\n", " ").split())
    blocks = re.findall(r"\bMED_START\b\s*(.*?)\s*(?:\bMED_END\b|MED_\s*END)", flat, flags=re.IGNORECASE | re.DOTALL)

    for block in blocks:
        med = ParsedMed(
            drug=normalize_value(_extract_one(block, "DRUG")),
            strength=normalize_value(_extract_one(block, "STRENGTH")),
            form=normalize_value(_extract_one(block, "FORM")),
            route=normalize_value(_extract_one(block, "ROUTE")),
            dose=normalize_value(_extract_one(block, "DOSE")),
            freq=normalize_value(_extract_one(block, "FREQ")),
            duration=normalize_value(_extract_one(block, "DURATION")),
            refills=normalize_value(_extract_one(block, "REFILLS")),
        )
        if med.drug:
            parsed.meds.append(med)

    return parsed


def canonicalize_parsed_dsl(parsed: ParsedDSL) -> str:
    lines = [
        "ORDO",
        f"PATIENT: {parsed.patient}",
        f"PRESCRIPTEUR: {parsed.prescripteur}",
        f"DATE: {parsed.date}",
        "",
    ]
    for med in parsed.meds:
        lines.append("MED_START")
        if med.drug:
            lines.append(f"DRUG: {med.drug}")
        if med.strength:
            lines.append(f"STRENGTH: {med.strength}")
        if med.form:
            lines.append(f"FORM: {med.form}")
        if med.route:
            lines.append(f"ROUTE: {med.route}")
        if med.dose:
            lines.append(f"DOSE: {med.dose}")
        if med.freq:
            lines.append(f"FREQ: {med.freq}")
        if med.duration:
            lines.append(f"DURATION: {med.duration}")
        if med.refills:
            lines.append(f"REFILLS: {med.refills}")
        lines.append("MED_END")
        lines.append("")
    lines.append("END")
    return "\n".join(lines).strip()


def compare_strings(a: str, b: str) -> float:
    a_n = normalize_value(a)
    b_n = normalize_value(b)
    if not a_n and not b_n:
        return 1.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def compare_parsed_dsl(gt: ParsedDSL, pred: ParsedDSL) -> Dict[str, float]:
    header_scores = [
        compare_strings(gt.patient, pred.patient),
        compare_strings(gt.prescripteur, pred.prescripteur),
        compare_strings(gt.date, pred.date),
    ]

    max_len = max(len(gt.meds), len(pred.meds))
    med_field_scores: List[float] = []
    med_exact_flags: List[float] = []

    empty = ParsedMed()
    med_keys = ["drug", "strength", "form", "route", "dose", "freq", "duration", "refills"]

    for i in range(max_len):
        gt_med = gt.meds[i] if i < len(gt.meds) else empty
        pred_med = pred.meds[i] if i < len(pred.meds) else empty
        field_scores = [compare_strings(getattr(gt_med, k), getattr(pred_med, k)) for k in med_keys]
        med_field_scores.extend(field_scores)
        med_exact_flags.append(float(all(score == 1.0 for score in field_scores)))

    canonical_gt = canonicalize_parsed_dsl(gt)
    canonical_pred = canonicalize_parsed_dsl(pred)

    all_scores = header_scores + med_field_scores
    return {
        "dsl_normalized_exact_match": float(canonical_gt == canonical_pred),
        "dsl_sequence_similarity": compare_strings(canonical_gt, canonical_pred),
        "dsl_field_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "dsl_header_score": float(np.mean(header_scores)) if header_scores else 0.0,
        "dsl_med_score": float(np.mean(med_field_scores)) if med_field_scores else 0.0,
        "dsl_med_exact_match": float(np.mean(med_exact_flags)) if med_exact_flags else 1.0,
        "gt_med_count": float(len(gt.meds)),
        "pred_med_count": float(len(pred.meds)),
        "canonical_gt": canonical_gt,
        "canonical_pred": canonical_pred,
    }


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


def _clean_json_response(response_text: str) -> str:
    text = response_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def ollama_extract_to_doc(raw_text: str, ollama_url: str, model_name: str, timeout_s: int) -> Tuple[OrdoDoc, dict]:
    schema = {
        "type": "object",
        "properties": {
            "patient": {
                "type": "object",
                "properties": {
                    "nom": {"type": "string"},
                    "date_naissance": {"type": "string"},
                },
                "required": ["nom", "date_naissance"],
            },
            "prescriptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "medicament": {"type": "string"},
                        "unite": {"type": "string"},
                        "posologie": {"type": "string"},
                        "duree": {"type": "string"},
                    },
                    "required": ["medicament", "unite", "posologie", "duree"],
                },
            },
            "date_document": {"type": "string"},
        },
        "required": ["patient", "prescriptions", "date_document"],
    }

    payload = {
        "model": model_name,
        "prompt": (
            "You receive the OCR of a prescription, and you reply the conversion in a strict json format. "
            "You have to fix OCR mistakes if you notice any.\n\n"
            f"{raw_text}"
        ),
        "format": schema,
        "stream": False,
    }

    response = requests.post(ollama_url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    result = response.json()
    response_text = result.get("response", "{}")
    parsed = json.loads(_clean_json_response(response_text))

    patient = parsed.get("patient", {}).get("nom", "UNKNOWN")
    date_str = parsed.get("date_document", "1900-01-01")
    prescriber = "UNKNOWN"
    lines: List[LineItem] = []

    for presc in parsed.get("prescriptions", []):
        drug_name = presc.get("medicament", "")
        strength = presc.get("unite", "")
        duration = presc.get("duree", "")
        dosage = presc.get("posologie", "")
        poso = Posology(dose=dosage, duration=duration, frequency="", route="", form="", as_needed=False, as_needed_for="")
        lines.append(LineItem(drug_name=drug_name, strength=strength, posology=poso, refills=None))

    return OrdoDoc(patient_name=patient, prescriber_name=prescriber, date_str=date_str, lines=lines), result


def load_one_pair(txt_path: Path):
    json_path = txt_path.with_suffix(".fhir.json")
    with open(txt_path, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    with open(json_path, "r", encoding="utf-8") as f:
        fhir = json.load(f)

    entries = fhir.get("entry", [])
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
            strength = med_text[i + 1 : -1].strip()

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

    doc_gt = OrdoDoc(patient_name=patient_name, prescriber_name=prescriber_name, date_str=date_str, lines=lines)
    target_dsl = ordo_to_linear_text(doc_gt)
    return {"input_text": input_text, "target_text": target_dsl}


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "datasets_mimic/output_mimic_fhir_ocr_template_prod"))
    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    timeout_s = int(os.getenv("OLLAMA_TIMEOUT", "600"))
    test_samples = int(os.getenv("TEST_SAMPLES", "0"))

    all_pairs = [load_one_pair(p) for p in data_dir.glob("*.txt")]
    if not all_pairs:
        raise ValueError(f"No .txt/.fhir.json pairs found in {data_dir}")

    print(f"{len(all_pairs)} total examples")
    dataset = Dataset.from_list(all_pairs)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    test_ds = dataset["test"]
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_ds = train_val["train"]
    val_ds = train_val["test"]
    print("train:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))

    if test_samples > 0:
        capped = min(test_samples, len(test_ds))
        test_ds = test_ds.shuffle(seed=845).select(range(capped))
        print(f"Using capped test set: {len(test_ds)} samples (TEST_SAMPLES={test_samples})")

    print("\n== Benchmark DSL-only (full outputs) ==")
    print(f"Running on {len(test_ds)} samples (Batch Size = 1)...")

    latencies = []
    failures = 0
    prompt_eval_tokens = []
    eval_tokens = []

    normalized_exact_scores = []
    sequence_scores = []
    field_scores = []
    header_scores = []
    med_scores = []
    med_exact_scores = []
    gt_med_counts = []
    pred_med_counts = []

    start_time_total = time.perf_counter()

    for i, sample in enumerate(test_ds):
        input_text = sample["input_text"]
        target_dsl_raw = sample["target_text"]
        gt_parsed = parse_dsl_robust(target_dsl_raw)
        canonical_gt = canonicalize_parsed_dsl(gt_parsed)

        t0 = time.perf_counter()
        error_msg = None
        raw_response_json = None
        raw_response_text = None
        pred_dsl_raw = ""
        canonical_pred = ""

        try:
            pred_doc, raw_ollama = ollama_extract_to_doc(
                raw_text=input_text,
                ollama_url=ollama_url,
                model_name=model_name,
                timeout_s=timeout_s,
            )
            raw_response_json = raw_ollama
            raw_response_text = raw_ollama.get("response", "")

            prompt_eval = raw_ollama.get("prompt_eval_count")
            eval_count = raw_ollama.get("eval_count")
            if isinstance(prompt_eval, int):
                prompt_eval_tokens.append(prompt_eval)
            if isinstance(eval_count, int):
                eval_tokens.append(eval_count)

            pred_dsl_raw = ordo_to_linear_text(pred_doc)
            pred_parsed = parse_dsl_robust(pred_dsl_raw)
            scores = compare_parsed_dsl(gt_parsed, pred_parsed)
            canonical_pred = scores.pop("canonical_pred")
            _ = scores.pop("canonical_gt")
        except Exception as e:
            failures += 1
            error_msg = str(e)
            scores = {
                "dsl_normalized_exact_match": 0.0,
                "dsl_sequence_similarity": 0.0,
                "dsl_field_score": 0.0,
                "dsl_header_score": 0.0,
                "dsl_med_score": 0.0,
                "dsl_med_exact_match": 0.0,
                "gt_med_count": float(len(gt_parsed.meds)),
                "pred_med_count": 0.0,
            }

        t1 = time.perf_counter()
        latency = t1 - t0
        latencies.append(latency)

        normalized_exact_scores.append(scores["dsl_normalized_exact_match"])
        sequence_scores.append(scores["dsl_sequence_similarity"])
        field_scores.append(scores["dsl_field_score"])
        header_scores.append(scores["dsl_header_score"])
        med_scores.append(scores["dsl_med_score"])
        med_exact_scores.append(scores["dsl_med_exact_match"])
        gt_med_counts.append(scores["gt_med_count"])
        pred_med_counts.append(scores["pred_med_count"])

        print(f"\n{'=' * 100}")
        print(f"SAMPLE {i + 1}/{len(test_ds)}")
        print(f"LATENCY: {latency:.4f} s")
        print(f"DSL NORMALIZED EXACT MATCH: {scores['dsl_normalized_exact_match']:.4f}")
        print(f"DSL SEQUENCE SIMILARITY:    {scores['dsl_sequence_similarity']:.4f}")
        print(f"DSL FIELD SCORE:            {scores['dsl_field_score']:.4f}")
        print(f"DSL HEADER SCORE:           {scores['dsl_header_score']:.4f}")
        print(f"DSL MED SCORE:              {scores['dsl_med_score']:.4f}")
        print(f"DSL MED EXACT MATCH:        {scores['dsl_med_exact_match']:.4f}")
        print(f"GT MED COUNT:               {int(scores['gt_med_count'])}")
        print(f"PRED MED COUNT:             {int(scores['pred_med_count'])}")
        if error_msg:
            print(f"ERROR: {error_msg}")

        print("\nINPUT (OCR):")
        print(input_text)

        print("\nGROUND TRUTH DSL (RAW):")
        print(target_dsl_raw)

        print("\nGROUND TRUTH DSL (CANONICAL):")
        print(canonical_gt)

        print("\nMODEL RAW JSON/TEXT OUTPUT:")
        if raw_response_text is not None:
            print(raw_response_text)
        elif raw_response_json is not None:
            print(json.dumps(raw_response_json, ensure_ascii=False, indent=2))
        else:
            print("<NO OUTPUT>")

        print("\nPREDICTED DSL (RAW):")
        print(pred_dsl_raw if pred_dsl_raw else "<NO DSL>")

        print("\nPREDICTED DSL (CANONICAL):")
        print(canonical_pred if canonical_pred else "<NO CANONICAL DSL>")

    end_time_total = time.perf_counter()
    total_duration = end_time_total - start_time_total

    print("\n" + "#" * 100)
    print("BENCHMARK SUMMARY")
    print("#" * 100)
    print(f"Samples:                    {len(test_ds)}")
    print(f"Avg Latency:                {float(np.mean(latencies)):.4f} s (+/- {float(np.std(latencies)):.4f})")
    print(f"Throughput:                 {len(test_ds) / total_duration if total_duration > 0 else 0.0:.4f} seq/s")
    print(f"Failures:                   {failures}")
    print(f"DSL Normalized Exact Match: {float(np.mean(normalized_exact_scores)):.4f}")
    print(f"DSL Sequence Similarity:    {float(np.mean(sequence_scores)):.4f}")
    print(f"DSL Field Score:            {float(np.mean(field_scores)):.4f}")
    print(f"DSL Header Score:           {float(np.mean(header_scores)):.4f}")
    print(f"DSL Med Score:              {float(np.mean(med_scores)):.4f}")
    print(f"DSL Med Exact Match:        {float(np.mean(med_exact_scores)):.4f}")
    print(f"Avg GT Med Count:           {float(np.mean(gt_med_counts)):.2f}")
    print(f"Avg Pred Med Count:         {float(np.mean(pred_med_counts)):.2f}")
    if prompt_eval_tokens:
        print(f"Avg Prompt Tok:             {float(np.mean(prompt_eval_tokens)):.1f}")
    if eval_tokens:
        print(f"Avg Output Tok:             {float(np.mean(eval_tokens)):.1f}")

    print("\n== Runtime Config ==")
    print(f"DATA_DIR:                   {data_dir}")
    print(f"OLLAMA_URL:                 {ollama_url}")
    print(f"OLLAMA_MODEL:               {model_name}")
    print(f"OLLAMA_TIMEOUT:             {timeout_s}")
    print(f"TEST_SAMPLES:               {test_samples if test_samples > 0 else 'ALL'}")


if __name__ == "__main__":
    main()
