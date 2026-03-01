import os
import json
import random
import argparse
import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# --- LABEL CONFIGURATION ---
# Define the labels used for the BIO scheme
LABEL_LIST = [
    "O", 
    "B-DRUG", "I-DRUG",
    "B-STRENGTH", "I-STRENGTH",
    "B-FORM", "I-FORM",
    "B-DOSAGE", "I-DOSAGE",
    "B-DURATION", "I-DURATION",
    "B-FREQUENCY", "I-FREQUENCY"    
]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}

# --- DATA CLASSES & MAPPINGS ---
ROUTE_MAP_FR = {
    "PO": "orale",
    "ORAL": "orale",
    "P.O.": "orale",
    "PER OS": "orale",
    "IV": "intraveineuse",
    "INTRAVENOUS": "intraveineuse",
    "IM": "intramusculaire",
    "INTRAMUSCULAR": "intramusculaire",
    "SC": "sous-cutanée",
    "SUBCUTANEOUS": "sous-cutanée",
    "SUB-Q": "sous-cutanée",
    "PR": "rectale",
    "RECTAL": "rectale",
    "SL": "sublinguale",
    "SUBLINGUAL": "sublinguale",
    "INHALATION": "inhalée",
    "INH": "inhalée",
    "TOPICAL": "cutanée",
    "TP": "cutanée",
    "BUCCAL": "buccale",
    "NASAL": "nasale",
    "ID": "intradermique",
    "INTRADERMAL": "intradermique",
    "OPHTHALMIC": "oculaire",
    "OPHTH": "oculaire",
    "OTIC": "auriculaire",
    "NEB": "nébulisée",
    "EPIDURAL": "épidurale",
    "INTRA-ARTICULAR": "intra-articulaire",
    "INTRATHECAL": "intrathécale",
    "PERCUTANEOUS": "percutanée",
    "FEEDING TUBE": "par sonde",
    "TRANSDERMAL": "transdermique",
    "INTRATRACHEAL": "intratrachéale",
}


FORM_MAP_FR = {
    "VIAL": "flacon",
    "SYRINGE": "seringue",
    "BAG": "sac",
}

FORM_ABBREV = {
    "comprimé": "cp",
    "gélule": "gél",
    "solution": "sol",
    "pommade": "pomm",
    "sirop": "sir",
}

ROUTE_ABBREV = {
    "orale": "po",
    "intraveineuse": "iv",
    "intramusculaire": "im",
    "sous-cutanée": "sc",
    "cutanée": "top",
    "inhalée": "inh",
}

@dataclass
class Posology:
    dose: str
    frequency: str
    duration: str
    route: str
    form: str

@dataclass
class LineItem:
    drug_name: str
    strength: str
    posology: Posology

def map_route_to_french(route_str: str) -> str:
    if not route_str: return ""
    r = route_str.strip().upper()
    return ROUTE_MAP_FR.get(r, route_str.lower())

def infer_form(strength_str: str) -> str:
    if not strength_str: return ""
    s = strength_str.lower()
    for keyword, french_form in FORM_MAP_FR.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", s):
            return french_form
    return ""

def posology_from_mimic(row) -> Posology:
    if row.get("dose_val_rx") and row.get("dose_unit_rx"):
        dose = f"{row['dose_val_rx']} {row['dose_unit_rx']}"
    elif row.get("prod_strength"):
        dose = row["prod_strength"]
    else:
        dose = ""

    route_raw = row.get("route", "")
    route = map_route_to_french(route_raw)

    d24 = row.get("doses_per_24_hrs")
    if pd.notna(d24) and d24 not in ["", None]:
        try:
            frequency = f"{int(float(d24))}/j"
        except Exception:
            frequency = ""
    else:
        frequency = ""

    duration = ""
    try:
        if row.get("starttime") and row.get("stoptime"):
            start = pd.to_datetime(row["starttime"])
            stop = pd.to_datetime(row["stoptime"])
            delta = (stop - start).days
            if delta > 0:
                duration = f"{delta} jours"
    except Exception:
        duration = ""

    if row.get("form_rx"):
        form_raw = str(row["form_rx"]).strip()
        key = form_raw.upper()
        form = FORM_MAP_FR.get(key, form_raw)
    else:
        form = infer_form(row.get("prod_strength", ""))

    as_needed = False
    prn_val = row.get("prn", "")
    if prn_val not in ["", None]:
        s = str(prn_val).strip().upper()
        if s in {"1", "Y", "YES", "TRUE", "T"}:
            as_needed = True

    return Posology(
        dose=dose,
        frequency=frequency,
        duration=duration,
        route=route,
        form=form,
        as_needed=as_needed,
        as_needed_for="",
    )

def format_realistic_line(line: LineItem) -> List[Tuple[str, Optional[str]]]:
    """
    Returns ordered list of (text, label_base).
    label_base = None → tag O
    """

    p = line.posology

    # Always start with drug name
    output = [(line.drug_name, "DRUG")]

    # Strength typically comes early in prescriptions
    priority_fields = []
    if line.strength:
        priority_fields.append((line.strength, "STRENGTH"))

    # Other fields can vary in order
    other_fields = []

    if p.form:
        form = FORM_ABBREV.get(p.form, p.form)
        other_fields.append((form, "FORM"))

    if p.route:
        route = ROUTE_ABBREV.get(p.route, p.route)
        other_fields.append((route, None))  # no label

    if p.dose:
        other_fields.append((p.dose, "DOSAGE"))

    if p.frequency:
        other_fields.append((p.frequency, "FREQUENCY"))

    if p.duration:
        other_fields.append((p.duration, "DURATION"))

    # ---- PARTIAL SHUFFLING ----
    random.shuffle(other_fields)

    output.extend(priority_fields)
    output.extend(other_fields)

    return output

# --- BIO GENERATOR ENGINE ---
class BIODatasetGenerator:

    def tokenize_with_bio(self, text: str, label_base: Optional[str]):
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        if not tokens:
            return [], []

        token_list = []
        tag_list = []

        if label_base is None:
            for tok in tokens:
                token_list.append(tok)
                tag_list.append(LABEL_TO_ID["O"])
            return token_list, tag_list

        token_list.append(tokens[0])
        tag_list.append(LABEL_TO_ID[f"B-{label_base}"])

        for tok in tokens[1:]:
            token_list.append(tok)
            tag_list.append(LABEL_TO_ID[f"I-{label_base}"])

        return token_list, tag_list


    def create_example(self, lines: List[LineItem]) -> Dict:
        all_tokens = []
        all_tag_ids = []

        for line in lines:
            formatted_fields = format_realistic_line(line)

            for text, label_base in formatted_fields:
                tokens, tags = self.tokenize_with_bio(text, label_base)
                all_tokens.extend(tokens)
                all_tag_ids.extend(tags)

        return {
            "tokens": all_tokens,
            "ner_tags": all_tag_ids,
            "full_text": " ".join(all_tokens)
        }    

# --- MAIN EXECUTION ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to MIMIC prescriptions CSV")
    parser.add_argument("--out", default="prescriptions_bio.jsonl")
    parser.add_argument("--count", type=int, default=100, help="Number of prescriptions to generate")
    args = parser.parse_args()

    # Load MIMIC data
    df = pd.read_csv(args.csv).fillna("")
    catalog = [row for _, row in df.iterrows() if row.get("drug")]
    
    generator = BIODatasetGenerator()
    dataset = []

    print(f"Generating {args.count} examples...")

    for _ in range(args.count):
        # Pick 1-4 random meds per prescription
        num_meds = random.randint(1, 4)
        sample_rows = random.sample(catalog, min(num_meds, len(catalog)))
        
        doc_lines = []
        for row in sample_rows:
            doc_lines.append(LineItem(
                drug_name=row["drug"],
                strength=str(row.get("prod_strength", "")),
                posology=posology_from_mimic(row)
            ))
        
        example = generator.create_example(doc_lines)
        dataset.append(example)

    # Save as JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Success! Dataset saved to {args.out}")
    print(f"Labels used: {LABEL_LIST}")

if __name__ == "__main__":
    main()
