#!/usr/bin/env python3

import json
import random
import argparse
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path

# --- LABEL CONFIGURATION ---
LABEL_LIST = [
    "O",
    "B-Drug", "I-Drug",
    "B-Strength", "I-Strength",
    "B-Form", "I-Form",
    "B-Dosage", "I-Dosage",
    "B-Duration", "I-Duration",
    "B-Frequency", "I-Frequency"
]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}

# --- MIMIC DATA CLEANING ---

_CONTAINER_DRUG_NAMES: Set[str] = {
    "bag", "syringe", "vial", "bottle", "ampule", "amp", "cassette",
    "soln", "flush", "sw", "d5w", "premix", "ns",
    "sterile water", "sodium chloride", "dextrose", "iso-osmotic dextrose",
    "iso-osmotic sodium chloride",
    "lactated ringers", "lactated ringer", "ringers", "ringer",
    "normal saline", "half normal saline",
    "blue cassette", "yellow cassette", "cadd cassette",
    # Glucose-only entries are IV/oral gels — not real prescription drugs in French context
    "glucose gel",
}

_IV_FLUID_PREFIX = re.compile(
    r"^(d\s*\d|0[\.\d]*\s*%|%\s*sodium|lactated|ringers?|normal\s*saline|"
    r"potassium\s+chl\s+\d|iso-osmotic)",
    re.IGNORECASE
)

# Qualifiers to strip from drug names
_DRUG_NAME_QUALIFIERS = re.compile(
    r"\b(IV|IM|SC|PO|SL|PR|oral|inj|inhalation|extended-release|ER|XR|XL|SR|"
    r"disintegrating|ophth|ophthalmic|otic|flush|premix|frozen|floor\s+stock|"
    r"ud|cadd|catheter\s+clearance|"
    r"MDI|Neb|ODT|EC|dwell|immediate\s+release|extended\s+release|"
    r"EXTended\s+release|DELayed\s+release|sustained\s+release|"
    r"Inhaler|Rectal|rectal|Topical|topical|Nasal|nasal|"
    r"Critical\s+Care|Oncology|Replacement|Liquid|Cream|Gel|Patch|"
    r"Polyvalent|Vac|valent|polysaccharide|vaccine|Virus|"
    r"Oral\s+Solution|Loading\s+Dose|Self\s+Administering\s+Medication|"
    r"Hemodialysis|CRRT\s+Machine\s+Priming|Alcohol\s+Withdrawal|"
    r"Dose\s+Taper|Buffered|Mineral\s+Oil|Powder|"
    r"Rinse|Wash|Swab|Wipe)\b",
    re.IGNORECASE
)

# Parentheticals that are unit/institutional descriptors, not brand names
_PAREN_JUNK = re.compile(
    r"\(\s*("
    r"units?/m[lL]|UI/mL|mcg/mL|mg/mL|"
    r"units?|UI|Critical|Oncology|Replacement|"
    r"Rectal|Topical|Liquid|Cream|Gel|Patch|"
    r"Self\s+Administering|CRRT|Hemodialysis|"
    r"Loading\s+Dose|Alcohol\s+Withdrawal|Dose\s+Taper|"
    r"Mineral\s+Oil|Powder|Buffered"
    r")[^)]*\)",
    re.IGNORECASE
)

# Dose-release / formulation suffixes that leak into prod_strength after truncation
_STRENGTH_SUFFIX_JUNK = re.compile(
    r"\s*(ER|XL|XR|SR|NS|D5|Mini|Bag\s*Plus|Mini\s*Bag)\s*$",
    re.IGNORECASE
)

_STRENGTH_JUNK_PATTERN = re.compile(
    r"^\s*("
    r"bag|vial|syringe|bottle|cassette|packet|cup|tube|premix|"
    r"flush|soln|solution|sterile water|sodium chloride|dextrose|"
    r"floor stock|frozen|ud|cadd"
    r")\b",
    re.IGNORECASE
)

# dose_unit_rx values that are form/container tokens, not units of measure
_JUNK_DOSE_UNITS: Set[str] = {
    "BAG", "TAB", "TABLET", "APPL", "NEB", "PKT", "PACKET",
    "VIAL", "PUFF", "PTCH", "PATCH", "DROP", "CAP", "CAPSULE",
    "SPRY", "SPRAY", "SYR", "SYRINGE", "SUPP", "LOZENGE", "WAFER",
}

FORM_MAP_FR: Dict[str, str] = {
    "VIAL":     "flacon",
    "SYRINGE":  "seringue",
    "BAG":      "sac",
    "TABLET":   "comprimé",
    "TAB":      "comprimé",
    "CAPSULE":  "gélule",
    "CAP":      "gélule",
    "SOLUTION": "solution",
    "PACKET":   "comprimé",
    "BOTTLE":   "flacon",
    "AMP":      "flacon",
    "AMPULE":   "flacon",
}


def _strip_parens(name: str) -> str:
    """
    Remove all parenthetical content from a drug name string.
    Both balanced ( ... ) and unclosed ( ... <end> are stripped entirely.
    This is appropriate because MIMIC parentheticals contain:
      - brand names we don't need (handled by keeping base INN)
      - form/route descriptors (junk)
      - dosage info already captured elsewhere
    """
    # First remove known junk parentheticals
    name = _PAREN_JUNK.sub("", name)
    # Then remove ALL remaining parenthetical groups (balanced or not)
    # Balanced: "( Topamax )" -> ""
    name = re.sub(r"\([^)]*\)", "", name)
    # Unclosed: "( MS Contin" -> ""
    name = re.sub(r"\([^)]*$", "", name)
    return name


def clean_drug_name(raw: str) -> Optional[str]:
    """
    Clean a MIMIC drug name to a usable Drug entity string.
    Returns None if the entry is a container/fluid or nothing meaningful remains.
    """
    if not raw:
        return None

    name = raw.strip()
    lower = name.lower()

    # Blocklist: containers and IV fluids
    for blocked in _CONTAINER_DRUG_NAMES:
        if lower == blocked or lower.startswith(blocked + " ") or lower.startswith(blocked + "("):
            return None
    if _IV_FLUID_PREFIX.match(name):
        return None

    # Strip ALL parenthetical content (brand names, form descriptors, etc.)
    name = _strip_parens(name)

    # Strip preparation/route/form qualifiers
    name = _DRUG_NAME_QUALIFIERS.sub("", name)

    # Strip numeric tokens and bare % / unit suffixes embedded in drug name
    # (e.g. "Lidocaine Jelly 2%" -> "Lidocaine Jelly", "Vitamin D 3" -> "Vitamin D")
    name = re.sub(r"\b\d+(?:[,\.]\d+)*\s*(?:mg|mcg|g|mL|mEq|mmol|%|UI)?\b", "", name)

    # Strip orphan % and special characters
    name = re.sub(r"[%&]", "", name)

    # Strip "- ICU", "- Days 2-7" style trailing descriptors
    name = re.sub(r"\s*-\s*(ICU|Days?|Dose|Loading|Taper|Step)\b.*$", "", name, flags=re.IGNORECASE)

    # Collapse and trim
    name = re.sub(r"\s{2,}", " ", name).strip()
    name = name.strip("()[]-, /")

    if len(name) < 2:
        return None

    # Post-cleaning check: if the cleaned result is itself a blocked term
    # (e.g. "5% Dextrose" -> strips "5%" -> "Dextrose" -> still an IV fluid)
    if name.lower() in _CONTAINER_DRUG_NAMES:
        return None

    return name


def clean_strength(raw: str) -> Optional[str]:
    """
    Clean a MIMIC prod_strength string to a usable Strength entity.
    Returns None if it is a container/form description or has no valid concentration.
    """
    if not raw:
        return None

    if _STRENGTH_JUNK_PATTERN.match(raw):
        return None

    # Fix 4: remove thousands-separator commas from numbers (e.g. "25,000" -> "25000")
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", raw)

    # Normalize gm -> g
    cleaned = re.sub(r"\bgm\b", "g", cleaned, flags=re.IGNORECASE)

    # Normalize UNIT/UNITS -> UI
    cleaned = re.sub(r"\bUNITS?\b", "UI", cleaned, flags=re.IGNORECASE)

    # Drop zero-value strengths (MIMIC placeholder)
    if re.match(r"^\s*0\s+\w", cleaned):
        return None

    # Must contain at least one numeric+unit pair
    has_unit = re.search(
        r"\d+(?:\.\d+)?\s*[-]?\s*(mg|mcg|µg|g|ml|mEq|mmol|UI|%)",
        cleaned,
        re.IGNORECASE
    )
    if not has_unit:
        return None

    # Remove orphan '%' not preceded by a digit
    cleaned = re.sub(r"(?<!\d)\s*%", "", cleaned)

    # Re-check a concentration unit remains
    has_conc_unit = re.search(
        r"\d+(?:\.\d+)?\s*[-]?\s*(mg|mcg|µg|g|mEq|mmol|UI|%)",
        cleaned,
        re.IGNORECASE
    )
    if not has_conc_unit:
        return None

    # Normalize spacing between value and unit
    cleaned = re.sub(
        r"(\d+(?:\.\d+)?)\s*-?\s*(mg|mcg|µg|g|ml|mEq|mmol|UI)",
        r"\1 \2",
        cleaned,
        flags=re.IGNORECASE
    )

    # Truncate at first form/container keyword
    truncated = re.split(
        r"\b(vial|syringe|bag|bottle|tablet|tab|capsule|cap|packet|premix|"
        r"amp|ampule|cup|tube|flush|soln|frozen|lozenge)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0].strip()

    # Strip trailing release-modifier suffixes (ER, XL, SR, NS, D5, Mini Bag)
    truncated = _STRENGTH_SUFFIX_JUNK.sub("", truncated).strip()

    # Strip trailing punctuation / slashes left over
    truncated = truncated.strip("(),- /")

    if not truncated or re.fullmatch(r"[\s%()/-]*", truncated):
        return None

    return truncated


def clean_form(raw: str) -> Optional[str]:
    """Map a MIMIC form_rx string to canonical French form name, or None."""
    if not raw:
        return None
    key = raw.strip().upper()
    if key in FORM_MAP_FR:
        return FORM_MAP_FR[key]
    lower = raw.lower()
    for eng, fr in FORM_MAP_FR.items():
        if eng.lower() in lower:
            return fr
    return None


def is_valid_row(row: dict) -> bool:
    """Return True if the row has a usable drug name after cleaning."""
    drug = str(row.get("drug") or "")
    if not drug.strip():
        return False
    return clean_drug_name(drug) is not None


# --- MAPPINGS ---

ROUTE_MAP_FR = {
    "PO": "orale", "ORAL": "orale", "P.O.": "orale", "PER OS": "orale",
    "PO/NG": "orale", "NG": "orale",
    "IV": "intraveineuse", "INTRAVENOUS": "intraveineuse",
    "IV DRIP": "intraveineuse", "IV BOLUS": "intraveineuse", "IVPCA": "intraveineuse",
    "IM": "intramusculaire", "INTRAMUSCULAR": "intramusculaire",
    "SC": "sous-cutanée", "SUBCUTANEOUS": "sous-cutanée", "SUB-Q": "sous-cutanée",
    "PR": "rectale", "RECTAL": "rectale",
    "SL": "sublinguale", "SUBLINGUAL": "sublinguale",
    "INHALATION": "inhalée", "INH": "inhalée", "IH": "inhalée",
    "TOPICAL": "cutanée", "TP": "cutanée", "TD": "transdermique",
    "BUCCAL": "buccale", "NASAL": "nasale",
    "ID": "intradermique", "INTRADERMAL": "intradermique",
    "OPHTHALMIC": "oculaire", "OPHTH": "oculaire", "BOTH EYES": "oculaire",
    "OTIC": "auriculaire", "NEB": "nébulisée",
    "EPIDURAL": "épidurale", "INTRA-ARTICULAR": "intra-articulaire",
    "INTRATHECAL": "intrathécale", "PERCUTANEOUS": "percutanée",
    "FEEDING TUBE": "par sonde", "TRANSDERMAL": "transdermique",
    "INTRATRACHEAL": "intratrachéale", "DIALYS": "dialyse",
}

FORM_VARIANTS: Dict[str, List[str]] = {
    "comprimé": ["cp", "cpr", "comprimé"],
    "gélule":   ["gél", "gélule"],
    "solution": ["sol", "solution"],
    "pommade":  ["pomm", "pommade"],
    "sirop":    ["sir", "sirop"],
    "flacon":   ["flacon"],
    "seringue": ["seringue"],
    "sac":      ["sac"],
}

FREQUENCY_MAP: Dict[int, List[Tuple[str, int]]] = {
    1:  [("le matin", 3), ("le soir", 3), ("tous les jours", 2), ("1 fois par jour", 1)],
    2:  [("matin et soir", 3), ("le matin et le soir", 2), ("2 fois par jour", 1)],
    3:  [("matin , midi et soir", 2), ("matin midi et soir", 2), ("3 fois par jour", 1)],
    4:  [("4 fois par jour", 1)],
    6:  [("6 fois par jour", 1)],
    8:  [("8 fois par jour", 1)],
    12: [("toutes les 2 heures", 1)],
    24: [("toutes les heures", 1)],
}

ORDERING_TEMPLATES = [
    (35, ["dose_form", "frequency"]),
    (25, ["dose_form", "frequency", "duration"]),
    (15, ["dose_form", "duration", "frequency"]),
    (15, ["frequency", "dose_form"]),
    (10, ["frequency", "dose_form", "duration"]),
]
_TEMPLATE_WEIGHTS  = [w for w, _ in ORDERING_TEMPLATES]
_TEMPLATE_PATTERNS = [p for _, p in ORDERING_TEMPLATES]

MED_COUNT_CHOICES = [1, 2, 3, 4]
MED_COUNT_WEIGHTS = [70, 25, 4, 1]


# --- DATA CLASSES ---

@dataclass
class Posology:
    dose: str
    strength: str
    frequency: str
    duration: str
    route: str
    form: str
    as_needed: bool = False
    as_needed_for: str = ""

@dataclass
class LineItem:
    drug_name: str
    strength: str
    posology: Posology


# --- HELPERS ---

def detokenize(tokens: List[str]) -> str:
    text = ""
    for tok in tokens:
        if tok in {";", ".", ",", ":"}:
            text += tok
        else:
            if text:
                text += " "
            text += tok
    return text


def sample_frequency(d24_int: int) -> str:
    if d24_int <= 0:
        return ""
    variants = FREQUENCY_MAP.get(d24_int)
    if not variants:
        return f"{d24_int} fois par jour"
    return random.choices([e for e, _ in variants], weights=[w for _, w in variants], k=1)[0]


def sample_form_variant(canonical_form: str) -> str:
    variants = FORM_VARIANTS.get(canonical_form)
    if variants:
        return random.choice(variants)
    for vs in FORM_VARIANTS.values():
        if canonical_form in vs:
            return random.choice(vs)
    return canonical_form


def map_route_to_french(route_str: str) -> str:
    if not route_str:
        return ""
    return ROUTE_MAP_FR.get(route_str.strip().upper(), "")


def extract_tablet_count(dose_val: str, strength_numeric: Optional[float] = None) -> str:
    """
    Extract a plausible tablet count (1–4) from dose_val_rx.
    Returns "" if the value equals the strength quantity (duplicate) or is out of range.
    """
    if not dose_val:
        return ""
    # dose_val can be a range like "5-15" — skip those
    if re.search(r"[^\d.]", dose_val.strip()):
        return ""
    try:
        val = float(dose_val)
        if strength_numeric is not None and abs(val - strength_numeric) < 1e-9:
            return ""
        if val == int(val) and 1 <= int(val) <= 4:
            return str(int(val))
    except (ValueError, TypeError):
        pass
    return ""


# --- MIMIC PARSING ---

def posology_from_mimic(row: dict) -> Posology:
    dose_val      = str(row.get("dose_val_rx")   or "").strip()
    dose_unit     = str(row.get("dose_unit_rx")  or "").strip()
    prod_strength = str(row.get("prod_strength") or "").strip()

    # Fix 4: strip thousands-separator commas from numeric fields
    dose_val = re.sub(r"(\d),(\d)", r"\1\2", dose_val)

    # Normalize UNIT -> UI in dose_unit_rx
    dose_unit_norm = re.sub(r"\bUNITS?\b", "UI", dose_unit, flags=re.IGNORECASE)

    # Skip junk dose_unit_rx (form tokens like TAB, NEB, PUFF, etc.)
    dose_unit_is_junk = dose_unit.upper() in _JUNK_DOSE_UNITS

    try:
        dose_val_float = float(dose_val) if dose_val and re.fullmatch(r"[\d.]+", dose_val) else None
    except ValueError:
        dose_val_float = None

    # Strength: prefer dose_val + dose_unit if non-zero and unit is real,
    # otherwise fall back to prod_strength
    if dose_val_float and dose_val_float > 0 and dose_unit_norm and not dose_unit_is_junk:
        raw_strength = f"{dose_val} {dose_unit_norm}"
        strength = clean_strength(raw_strength) or ""
    else:
        strength = clean_strength(prod_strength) or ""

    # Tablet count with deduplication against strength value
    tablet_count = extract_tablet_count(dose_val, strength_numeric=dose_val_float)
    dose = tablet_count

    route = map_route_to_french(str(row.get("route") or ""))

    d24 = row.get("doses_per_24_hrs")
    frequency = ""
    if pd.notna(d24) and d24 not in ["", None]:
        try:
            d24_int = round(float(d24))
            frequency = sample_frequency(d24_int)
        except (ValueError, TypeError):
            frequency = ""

    duration = ""
    try:
        if row.get("starttime") and row.get("stoptime"):
            start = pd.to_datetime(row["starttime"])
            stop  = pd.to_datetime(row["stoptime"])
            delta = (stop - start).days
            if delta > 0:
                unit  = "mois"  if delta >= 30 else "jours"
                value = round(delta / 30) if delta >= 30 else delta
                # Fix 5: singular "jour" when value is 1
                if value == 1 and unit == "jours":
                    unit = "jour"
                duration = random.choice([f"pendant {value} {unit}", f"{value} {unit}"])
    except Exception:
        duration = ""

    form = ""
    if row.get("form_rx"):
        form = clean_form(str(row["form_rx"])) or ""

    as_needed = False
    prn_val = row.get("prn")
    if prn_val not in ["", None]:
        if str(prn_val).strip().upper() in {"1", "Y", "YES", "TRUE", "T"}:
            as_needed = True

    return Posology(
        dose=dose, strength=strength, frequency=frequency,
        duration=duration, route=route, form=form,
        as_needed=as_needed, as_needed_for="",
    )


# --- FORMATTING ---

def format_realistic_line(line: LineItem) -> List[Tuple[str, Optional[str]]]:
    p = line.posology
    output: List[Tuple[str, Optional[str]]] = [(line.drug_name, "Drug")]

    if line.strength:
        output.append((line.strength, "Strength"))

    form_variant = sample_form_variant(p.form) if p.form else ""

    available: Dict[str, Optional[List[Tuple[str, Optional[str]]]]] = {
        "frequency": [(p.frequency, "Frequency")] if p.frequency else None,
        "duration":  [(p.duration,  "Duration")]  if p.duration  else None,
    }

    if p.dose and form_variant:
        available["dose_form"] = [(p.dose, "Dosage"), (form_variant, "Form")]
    elif p.dose:
        available["dose_form"] = [(p.dose, "Dosage")]
    elif form_variant:
        available["dose_form"] = [(form_variant, "Form")]
    else:
        available["dose_form"] = None

    template = random.choices(_TEMPLATE_PATTERNS, weights=_TEMPLATE_WEIGHTS, k=1)[0]
    for key in template:
        field = available.get(key)
        if field is not None:
            output.extend(field)

    if p.as_needed:
        output.append(("si besoin", None))

    return output


# --- BIO GENERATOR ---

class BIODatasetGenerator:

    def tokenize_with_bio(self, text: str, label_base: Optional[str]) -> Tuple[List[str], List[str]]:
        tokens = re.findall(
            r"\d+(?:\.\d+)?|[A-Za-zÀ-ÿµ]+(?:[-/][A-Za-zÀ-ÿµ]+)*|[^\w\s]",
            text, re.UNICODE
        )
        if not tokens:
            return [], []
        token_list, tag_list = [], []
        if label_base is None:
            for tok in tokens:
                token_list.append(tok)
                tag_list.append("O")
            return token_list, tag_list
        token_list.append(tokens[0])
        tag_list.append(f"B-{label_base}")
        for tok in tokens[1:]:
            token_list.append(tok)
            tag_list.append(f"I-{label_base}")
        return token_list, tag_list

    def create_example(self, lines: List[LineItem]) -> Dict:
        all_tokens:    List[str] = []
        all_tag_names: List[str] = []
        for i, line in enumerate(lines):
            for text, label_base in format_realistic_line(line):
                toks, tags = self.tokenize_with_bio(text, label_base)
                all_tokens.extend(toks)
                all_tag_names.extend(tags)
            if i < len(lines) - 1:
                all_tokens.append(";")
                all_tag_names.append("O")
        all_tokens.append(".")
        all_tag_names.append("O")
        return {
            "tokens":    all_tokens,
            "ner_tags":  all_tag_names,
            "full_text": detokenize(all_tokens),
        }


# --- MAIN ---

def sample_unique_drugs(catalog: List[dict], n: int) -> List[dict]:
    seen: Set[str] = set()
    result: List[dict] = []
    for row in random.sample(catalog, len(catalog)):
        drug = clean_drug_name(str(row.get("drug", "")))
        if drug and drug.lower() not in seen:
            seen.add(drug.lower())
            result.append(row)
        if len(result) == n:
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   required=True)
    parser.add_argument("--out",   default="prescriptions_bio.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv).replace({np.nan: None})
    raw_catalog = df.to_dict("records")
    catalog = [row for row in raw_catalog if is_valid_row(row)]

    n_dropped = len(raw_catalog) - len(catalog)
    print(f"Loaded {len(raw_catalog)} rows. Dropped {n_dropped} entries. "
          f"Usable catalog: {len(catalog)} rows.")
    if not catalog:
        raise SystemExit("No valid drug entries found.")

    generator = BIODatasetGenerator()
    dataset: List[Dict] = []
    print(f"Generating {args.count} examples...")

    for _ in range(args.count):
        num_meds    = random.choices(MED_COUNT_CHOICES, weights=MED_COUNT_WEIGHTS, k=1)[0]
        sample_rows = sample_unique_drugs(catalog, num_meds)
        if not sample_rows:
            continue
        doc_lines = []
        for row in sample_rows:
            drug_name = clean_drug_name(str(row["drug"]))
            if not drug_name:
                continue
            posology = posology_from_mimic(row)
            doc_lines.append(LineItem(
                drug_name=drug_name,
                strength=posology.strength,
                posology=posology,
            ))
        if not doc_lines:
            continue
        dataset.append(generator.create_example(doc_lines))

    open_path = Path(args.out)
    open_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(open_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} examples to {open_path}")
    print(f"Label scheme: {LABEL_LIST}")


if __name__ == "__main__":
    main()