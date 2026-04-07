import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from datasets import Dataset

from datasets_mimic.generate_ordo_mimic import Posology, LineItem, OrdoDoc, to_fhir_bundle

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

    return OrdoDoc(patient_name=patient_name, prescriber_name=prescriber_name, date_str=date_str, lines=lines)


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
        if not drug:
            continue
        strength = _extract_one(block, "STRENGTH")
        form = _extract_one(block, "FORM")
        route = _extract_one(block, "ROUTE")
        dose = _extract_one(block, "DOSE")
        freq = _extract_one(block, "FREQ")
        dur = _extract_one(block, "DURATION")
        refills = None
        ref = _extract_one(block, "REFILLS")
        if ref:
            try:
                refills = int(float(ref))
            except Exception:
                refills = None
        poso = Posology(dose=dose, frequency=freq, duration=dur, route=route, form=form)
        meds.append(LineItem(drug_name=drug, strength=strength, posology=poso, refills=refills))

    return OrdoDoc(patient_name=patient, prescriber_name=prescriber, date_str=date_str, lines=meds)


def safe_dsl_to_fhir(dsl_text: str, bundle_id: str = "eval") -> dict:
    dsl_text = normalize_dsl(dsl_text)
    doc = linear_text_to_ordo_robust(dsl_text)
    return to_fhir_bundle(doc, bundle_id=bundle_id)


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


def ollama_extract_to_doc(
    raw_text: str,
    ollama_url: str,
    model_name: str,
    timeout_s: int,
) -> tuple[OrdoDoc, dict]:
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
    doc_gt = fhir_bundle_to_ordo(fhir)
    target_dsl = ordo_to_linear_text(doc_gt)
    return {"input_text": input_text, "target_text": target_dsl}


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "datasets_mimic/output_mimic_fhir_ocr_template_prod"))
    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    timeout_s = int(os.getenv("OLLAMA_TIMEOUT", "600"))
    benchmark_samples = int(os.getenv("BENCHMARK_SAMPLES", "100"))
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
        test_ds = test_ds.select(range(capped))
        print(f"Using capped test set: {len(test_ds)} samples (TEST_SAMPLES={test_samples})")

    print("\n== Final Evaluation on TEST set (Ollama) ==")
    exact_dsl = []
    exact_fhir = []
    latencies = []
    failures = 0
    prompt_eval_tokens = []
    eval_tokens = []

    for i, sample in enumerate(test_ds):
        input_text = sample["input_text"]
        target_dsl = sample["target_text"]

        t0 = time.perf_counter()
        try:
            pred_doc, raw_ollama = ollama_extract_to_doc(
                raw_text=input_text,
                ollama_url=ollama_url,
                model_name=model_name,
                timeout_s=timeout_s,
            )
            prompt_eval = raw_ollama.get("prompt_eval_count")
            eval_count = raw_ollama.get("eval_count")
            if isinstance(prompt_eval, int):
                prompt_eval_tokens.append(prompt_eval)
            if isinstance(eval_count, int):
                eval_tokens.append(eval_count)
            pred_dsl = normalize_dsl(ordo_to_linear_text(pred_doc))
            gt_dsl = normalize_dsl(target_dsl)
            pred_fhir = to_fhir_bundle(pred_doc, bundle_id="pred")
            gt_fhir = safe_dsl_to_fhir(gt_dsl, bundle_id="gt")

            exact_dsl.append(int(pred_dsl.strip() == gt_dsl.strip()))
            exact_fhir.append(int(canon_no_ids(pred_fhir) == canon_no_ids(gt_fhir)))
        except Exception as e:
            failures += 1
            exact_dsl.append(0)
            exact_fhir.append(0)
            if i < 3:
                print(f"[ERROR] idx={i}: {e}")
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    test_metrics = {
        "test_dsl_exact_match": float(np.mean(exact_dsl)),
        "test_fhir_exact_match": float(np.mean(exact_fhir)),
        "test_failures": int(failures),
    }
    print(test_metrics)

    print("\n== Examples ==")
    num_examples = 5
    rng = np.random.default_rng(0)
    indices = rng.choice(len(test_ds), size=min(num_examples, len(test_ds)), replace=False)
    for k, idx in enumerate(indices, start=1):
        sample = test_ds[int(idx)]
        input_text = sample["input_text"]
        target_dsl = sample["target_text"]
        try:
            pred_doc, _ = ollama_extract_to_doc(
                raw_text=input_text,
                ollama_url=ollama_url,
                model_name=model_name,
                timeout_s=timeout_s,
            )
            pred_dsl = normalize_dsl(ordo_to_linear_text(pred_doc))
            gt_dsl = normalize_dsl(target_dsl)
            pred_fhir = to_fhir_bundle(pred_doc, bundle_id="pred")
            gt_fhir = safe_dsl_to_fhir(gt_dsl, bundle_id="gt")
            is_match = int(canon_no_ids(pred_fhir) == canon_no_ids(gt_fhir))
        except Exception as e:
            pred_dsl = f"<ERROR: {e}>"
            gt_dsl = normalize_dsl(target_dsl)
            pred_fhir = {}
            gt_fhir = safe_dsl_to_fhir(gt_dsl, bundle_id="gt")
            is_match = 0

        print(f"\n----- Example {k} (idx={idx}) -----")
        print("INPUT (OCR):")
        print(input_text)
        print("\nGT DSL:")
        print(target_dsl[:500])
        print("\nPRED DSL:")
        print(pred_dsl[:500])
        print("\nGT FHIR (canon head):")
        print(json.dumps(gt_fhir, ensure_ascii=False, indent=2)[:500] + "...")
        print("\nPRED FHIR (canon head):")
        print(json.dumps(pred_fhir, ensure_ascii=False, indent=2)[:500] + "...")
        print(f"\nFHIR EXACT MATCH: {is_match}")
        print("-" * 80)

    print("\n== Starting Benchmark (Time & Throughput) ==")
    benchmark_n = min(len(test_ds), benchmark_samples)
    print(f"Benchmarking on {benchmark_n} samples (Batch Size = 1)...")
    benchmark_data = test_ds.select(range(benchmark_n))

    # Warmup
    _ = ollama_extract_to_doc(
        raw_text=benchmark_data[0]["input_text"],
        ollama_url=ollama_url,
        model_name=model_name,
        timeout_s=timeout_s,
    )

    bench_latencies = []
    start_time_total = time.perf_counter()
    for item in benchmark_data:
        t0 = time.perf_counter()
        try:
            _ = ollama_extract_to_doc(
                raw_text=item["input_text"],
                ollama_url=ollama_url,
                model_name=model_name,
                timeout_s=timeout_s,
            )
        except Exception:
            pass
        t1 = time.perf_counter()
        bench_latencies.append(t1 - t0)
    end_time_total = time.perf_counter()

    total_duration = end_time_total - start_time_total
    avg_latency = float(np.mean(bench_latencies))
    std_latency = float(np.std(bench_latencies))
    throughput = benchmark_n / total_duration if total_duration > 0 else 0.0

    print(f"\nBenchmark Results ({benchmark_n} samples):")
    print(f"Avg Latency:    {avg_latency:.4f} s (+/- {std_latency:.4f})")
    print(f"Throughput:     {throughput:.2f} seq/s")
    print(f"Failures:       {failures}")
    if prompt_eval_tokens:
        print(f"Avg Prompt Tok: {float(np.mean(prompt_eval_tokens)):.1f}")
    if eval_tokens:
        print(f"Avg Output Tok: {float(np.mean(eval_tokens)):.1f}")

    print("\n== Runtime Config ==")
    print(f"DATA_DIR:       {data_dir}")
    print(f"OLLAMA_URL:     {ollama_url}")
    print(f"OLLAMA_MODEL:   {model_name}")
    print(f"OLLAMA_TIMEOUT: {timeout_s}")
    print(f"TEST_SAMPLES:   {test_samples if test_samples > 0 else 'ALL'}")


if __name__ == "__main__":
    main()
