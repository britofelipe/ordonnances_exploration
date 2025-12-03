import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import re

# --- CONFIGURATION & CONSTANTS ---
CANDIDATE_FONTS_TYPED = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "arial.ttf"
]
CANDIDATE_FONTS_HAND = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "comic.ttf"
]

# Keywords used for form inference
FORM_KEYWORDS = {
    "tablet": "comprimé", "tab": "comprimé",
    "capsule": "gélule", "cap": "gélule",
    "solution": "solution", "sol": "solution",
    "syrup": "sirop",
    "suspension": "suspension",
    "cream": "pommade", "ointment": "pommade",
    "inj": "solution injectable", "injection": "solution injectable",
    "ampul": "ampoule", "vial": "flacon"
}

# --- FRENCH ROUTE MAPPING ---
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
    "INTRATRACHEAL": "intratrachéale"
}

def map_route_to_french(route_str: str) -> str:
    if not route_str:
        return ""
    r = route_str.strip().upper()
    return ROUTE_MAP_FR.get(r, route_str.lower())

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

@dataclass
class OrdoDoc:
    patient_name: str
    prescriber_name: str
    date_str: str
    lines: List[LineItem]

# --- HELPERS ---
def infer_form(strength_str: str) -> str:
    if not strength_str:
        return ""
    strength_str_lower = strength_str.lower()
    for keyword, french_form in FORM_KEYWORDS.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', strength_str_lower):
            return french_form
    return ""

def pick_font(candidates, size):
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()

def text_wh(draw, text, font):
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l), (b - t)
    return draw.textsize(text, font=font)

def jitter(x, y):
    return x + random.uniform(-1, 1), y + random.uniform(-1, 1)

def sample_name():
    first = random.choice(["Jean", "Marie", "Pierre", "Sophie", "Lucas", "Camille"])
    last = random.choice(["Martin", "Bernard", "Dubois", "Robert", "Richard"])
    return f"{first} {last}"

def sample_date():
    d, m = random.randint(1, 28), random.randint(1, 12)
    return f"{d:02d}/{m:02d}/2024"

# --- POSOLOGY FROM MIMIC ---
def posology_from_mimic(row) -> Posology:
    # Dose
    if row.get("dose_val_rx") and row.get("dose_unit_rx"):
        dose = f"{row['dose_val_rx']} {row['dose_unit_rx']}"
    elif row.get("prod_strength"):
        dose = row["prod_strength"]
    else:
        dose = ""

    # Route
    route_raw = row.get("route", "")
    route = map_route_to_french(route_raw)

    # Frequency
    d24 = row.get("doses_per_24_hrs")
    if pd.notna(d24) and d24 not in ["", None]:
        try:
            frequency = f"{int(float(d24))}/j"
        except:
            frequency = ""
    else:
        frequency = ""

    # Duration
    duration = ""
    try:
        if row.get("starttime") and row.get("stoptime"):
            start = pd.to_datetime(row["starttime"])
            stop = pd.to_datetime(row["stoptime"])
            delta = (stop - start).days
            if delta > 0:
                duration = f"{delta} jours"
    except:
        duration = ""

    # Form
    if row.get("form_rx"):
        form = row["form_rx"]
    else:
        form = infer_form(row.get("prod_strength", ""))

    return Posology(
        dose=dose,
        frequency=frequency,
        duration=duration,
        route=route,
        form=form
    )

# --- LOAD CSV ---
def load_catalog_mimic(path: str) -> List[Dict]:
    print(f"Loading catalog from {path}...")
    df = pd.read_csv(path).fillna("")

    catalog = []
    for _, row in df.iterrows():
        if not row.get("drug"):
            continue

        entry = {
            "drug": row["drug"],
            "prod_strength": row.get("prod_strength", ""),
            "route": row.get("route", ""),
            "dose_val_rx": row.get("dose_val_rx", ""),
            "dose_unit_rx": row.get("dose_unit_rx", ""),
            "doses_per_24_hrs": row.get("doses_per_24_hrs", ""),
            "starttime": row.get("starttime", ""),
            "stoptime": row.get("stoptime", ""),
            "form_rx": row.get("form_rx", "")
        }
        catalog.append(entry)

    return catalog

# --- RENDERING ENGINE ---
def make_canvas(paper="A5", dpi=150):
    w_mm, h_mm = (148, 210) if paper == "A5" else (210, 297)
    px = (int(w_mm/25.4*dpi), int(h_mm/25.4*dpi))
    return Image.new("RGB", px, (250, 250, 245))

def draw_header(draw, W, y, font):
    title = "ORDONNANCE"
    tw, th = text_wh(draw, title, font)
    draw.text(((W-tw)//2, y), title, fill="black", font=font)
    return y + th + 40

def render_ordo(doc: OrdoDoc, style="typed") -> Tuple[Image.Image, List[Dict]]:
    img = make_canvas()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    font_typed = pick_font(CANDIDATE_FONTS_TYPED, 22)
    font_hand = pick_font(CANDIDATE_FONTS_HAND, 22)
    font_small = pick_font(CANDIDATE_FONTS_TYPED, 16)

    def get_font(is_hand=False, is_small=False):
        if is_hand: return font_hand
        return font_small if is_small else font_typed

    boxes = []
    y = 50
    y = draw_header(draw, W, y, font_typed)

    # Header
    for label, text in [("Patient", doc.patient_name), ("Date", doc.date_str)]:
        ft = get_font()
        full_text = f"{label}: {text}"
        draw.text(jitter(40, y), full_text, fill="black", font=ft)
        tw, th = text_wh(draw, full_text, ft)
        boxes.append({"label": label, "box": [40, y, tw, th], "text": text})
        y += th + 10

    y += 20
    draw.line((20, y, W-20, y), fill="grey")
    y += 30

    # Line Items
    for i, line in enumerate(doc.lines, 1):
        ft = get_font()
        str_txt = f" ({line.strength})" if line.strength else ""
        txt_main = f"{i}. {line.drug_name}{str_txt}"

        draw.text(jitter(40, y), txt_main, fill="black", font=ft)
        tw, th = text_wh(draw, txt_main, ft)
        boxes.append({"label": "DRUG", "box": [40, y, tw, th], "text": line.drug_name})
        y += th + 5

        # Form & Route
        details_parts = []
        if line.posology.form:
            details_parts.append(f"Forme: {line.posology.form}")
        if line.posology.route:
            details_parts.append(f"Voie: {line.posology.route}")

        if details_parts:
            ft_small = get_font(is_small=True)
            txt_det = "   ".join(details_parts)
            draw.text(jitter(60, y), txt_det, fill=(50,50,50), font=ft_small)
            tw, th = text_wh(draw, txt_det, ft_small)
            boxes.append({"label": "DETAILS", "box": [60, y, tw, th], "text": txt_det})
            y += th + 5

        # Posology
        ft_small = get_font(is_small=True)
        txt_pos = f"Posologie: {line.posology.dose}  {line.posology.frequency}  {line.posology.duration}"
        draw.text(jitter(60, y), txt_pos, fill=(50,50,50), font=ft_small)
        tw, th = text_wh(draw, txt_pos, ft_small)
        boxes.append({"label": "INSTRUCT", "box": [60, y, tw, th], "text": txt_pos})
        y += th + 25

    return img, boxes

# --- IMAGE GENERATION ---
def generate_one(catalog):
    n = random.randint(1, 4)
    selection = random.sample(catalog, k=min(n, len(catalog)))
    lines = []

    for drug in selection:
        pos = posology_from_mimic(drug)
        lines.append(LineItem(
            drug_name=drug["drug"],
            strength=drug["prod_strength"],
            posology=pos
        ))

    return OrdoDoc(sample_name(), sample_name(), sample_date(), lines)

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to MIMIC prescriptions CSV")
    parser.add_argument("--out", default="output_mimic", help="Output directory")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    catalog = load_catalog_mimic(args.csv)
    print(f"Loaded {len(catalog)} drugs.")

    for i in range(args.count):
        doc = generate_one(catalog)
        img, boxes = render_ordo(doc)

        img_path = os.path.join(args.out, f"ordo_{i:04d}.png")
        img.save(img_path)

        with open(os.path.join(args.out, f"ordo_{i:04d}.json"), "w") as f:
            json.dump(boxes, f, indent=2)

    print(f"Generated {args.count} ordonnance images in '{args.out}'")
