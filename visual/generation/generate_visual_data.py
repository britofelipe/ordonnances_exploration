import os
import sys
import json
import random
import argparse
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --- CONSTANTS & CONFIGURATION ---

# Fonts (Paths on the cluster)
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

# French Medical Mapping
ROUTE_MAP_FR = {
    "PO": "orale", "ORAL": "orale", "P.O.": "orale", "PER OS": "orale",
    "IV": "intraveineuse", "INTRAVENOUS": "intraveineuse",
    "IM": "intramusculaire", "INTRAMUSCULAR": "intramusculaire",
    "SC": "sous-cutanée", "SUBCUTANEOUS": "sous-cutanée", "SUB-Q": "sous-cutanée",
    "PR": "rectale", "RECTAL": "rectale",
    "SL": "sublinguale", "SUBLINGUAL": "sublinguale",
    "INHALATION": "inhalée", "INH": "inhalée",
    "TOPICAL": "cutanée", "TP": "cutanée",
}

FORM_KEYWORDS = {
    "tablet": "comprimé", "tab": "comprimé",
    "capsule": "gélule", "cap": "gélule",
    "solution": "solution", "sol": "solution",
    "syrup": "sirop",
    "cream": "pommade", "ointment": "pommade",
    "inj": "solution injectable", "injection": "solution injectable",
    "ampul": "ampoule", "vial": "flacon"
}

FORM_MAP_FR = {
    "VIAL": "flacon",
    "SYRINGE": "seringue",
    "BAG": "sac",
}

# YOLO Class Mapping
CLASS_MAPPING = {
    "header": 0,            # Patient/Doctor/Date Header
    "medication_block": 1,  # Full Meds Block
    "footer": 2,            # Signature/Footer
    "medication_item": 3    # Individual Drug Lines
}
# Simplified mapping for user request:
# 0: Header, 1: Medication Block, 2: Footer, 3: Medication Item

# --- DATA CLASSES ---

@dataclass
class Posology:
    dose: str
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
    refills: Optional[int] = None

@dataclass
class OrdoDoc:
    patient_name: str
    prescriber_name: str
    date_str: str
    lines: List[LineItem]

# --- HELPERS ---

def map_route_to_french(route_str: str) -> str:
    if not route_str: return ""
    r = route_str.strip().upper()
    return ROUTE_MAP_FR.get(r, route_str.lower())

def infer_form(strength_str: str) -> str:
    if not strength_str: return ""
    s = strength_str.lower()
    for k, v in FORM_KEYWORDS.items():
        if k in s: return v
    return ""

def sample_name():
    first_names = ["Jean", "Marie", "Lucas", "Camille", "Léa", "Paul", "Jules", "Zoé", "Hugo", "Anaïs", "Thomas", "Emma"]
    last_names = ["Martin", "Bernard", "Dubois", "Thomas", "Robert", "Petit", "Durand", "Leroy", "Moreau", "Simon"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def sample_date():
    d, m, y = random.randint(1, 28), random.randint(1, 12), 2024
    return f"{d:02d}/{m:02d}/{y}"

def pick_font(candidates, size):
    for p in candidates:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size=size)
            except: continue
    return ImageFont.load_default()

def text_wh(draw, text, font):
    if hasattr(draw, "textbbox"):
        try:
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            return (r - l), (b - t)
        except: pass
    return font.getsize(text)

def jitter(x, y):
    return x + random.uniform(-1, 1), y + random.uniform(-1, 1)

def apply_noise(img: Image.Image, blur=0.0, jpeg=0.0, skew=0.0, stains=0.0):
    if skew > 0:
        ang = random.uniform(-skew, skew)
        img = img.rotate(ang, resample=Image.BICUBIC, expand=1, fillcolor=(255,255,255))
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, max(blur, 0.5))))
    if stains > 0:
        dr = ImageDraw.Draw(img)
        W, H = img.size
        for _ in range(random.randint(1, max(1, int(3 * stains)))):
            x, y = random.randint(10, W-10), random.randint(10, H-10)
            r = random.randint(6, 20)
            dr.ellipse((x-r, y-r, x+r, y+r), fill=(random.randint(200,240),)*3, outline=None)
    if jpeg > 0:
        # Simplified JPEG artifact simulation
        import io
        buf = io.BytesIO()
        # Fix: ensure low bound doesn't exceed high bound (90)
        q_min = min(90, max(10, int(100 - 60*jpeg)))
        q = random.randint(q_min, 90)
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img

def sample_noise_from_args(args) -> Dict[str, float]:
    def s(val): return random.uniform(0.3 * val, val) if val > 0 else 0
    return {"blur": s(args.blur), "jpeg": s(args.jpeg), "skew": s(args.skew), "stains": s(args.stains)}

def posology_from_mimic(row) -> Posology:
    dose = f"{row.get('dose_val_rx','')}{row.get('dose_unit_rx','')}" or row.get('prod_strength','') or ""
    route = map_route_to_french(row.get('route', ''))
    freq = f"{int(float(row.get('doses_per_24_hrs')))}/j" if row.get('doses_per_24_hrs') else ""
    dur = "" # simplified
    form = FORM_MAP_FR.get(row.get('form_rx','').strip().upper(), infer_form(row.get('prod_strength','')))
    return Posology(dose=dose, frequency=freq, duration=dur, route=route, form=form)

# --- FHIR EXPORT HELPERS ---

def frequency_to_timing(freq: str) -> Dict:
    if not freq: return None
    m = re.match(r"\s*(\d+)\s*/\s*j", freq)
    if not m: return None
    f = int(m.group(1))
    return {"repeat": {"frequency": f, "period": 1, "periodUnit": "d"}}

def parse_dose_to_quantity(dose: str) -> Optional[Dict]:
    if not dose: return None
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s+(.+)$", dose)
    if not m: return None
    try: value = float(m.group(1))
    except: return None
    unit = m.group(2).strip()
    return {"value": value, "unit": unit}

def duration_to_bounds_duration(duration: str) -> Optional[Dict]:
    if not duration: return None
    m = re.match(r"\s*([0-9]+)", duration)
    if not m: return None
    value = int(m.group(1))
    return {"value": value, "unit": "d"}

def lineitem_to_medication_request(doc: OrdoDoc, line: LineItem, idx: int, base_id: str) -> Dict:
    poso = line.posology
    drug_line = line.drug_name
    if line.strength: drug_line += f" ({line.strength})"
    
    details_segments = []
    if poso.form: details_segments.append(f"Forme: {poso.form}")
    if poso.route: details_segments.append(f"Voie: {poso.route}")
    
    poso_segments = [p for p in [poso.dose, poso.frequency, poso.duration] if p]
    
    dosage_text = drug_line
    if details_segments: dosage_text += " " + " ".join(details_segments)
    if poso_segments: dosage_text += " Posologie: " + " ".join(poso_segments)
    
    dosage_instruction = {"text": dosage_text, "sequence": idx}
    
    if poso.as_needed:
        dosage_instruction["asNeededBoolean"] = True
        if poso.as_needed_for: dosage_instruction["asNeededCodeableConcept"] = {"text": poso.as_needed_for}
    if poso.route:
        dosage_instruction["route"] = {"text": poso.route}
        
    timing = frequency_to_timing(poso.frequency)
    bounds = duration_to_bounds_duration(poso.duration)
    if bounds:
        if not timing: timing = {"repeat": {}}
        if "repeat" not in timing: timing["repeat"] = {}
        timing["repeat"]["boundsDuration"] = bounds
    if timing: dosage_instruction["timing"] = timing
    
    if poso.dose:
        dose_entry = {"doseString": poso.dose}
        q = parse_dose_to_quantity(poso.dose)
        if q: dose_entry["doseQuantity"] = q
        dosage_instruction["doseAndRate"] = [dose_entry]
        
    mr = {
        "resourceType": "MedicationRequest",
        "id": f"{base_id}-med-{idx}",
        "status": "active",
        "intent": "order",
        "subject": {"display": doc.patient_name},
        "authoredOn": doc.date_str,
        "requester": {"display": doc.prescriber_name},
        "medicationCodeableConcept": {"text": drug_line},
        "dosageInstruction": [dosage_instruction]
    }
    if line.refills is not None:
        mr["dispenseRequest"] = {"numberOfRepeatsAllowed": line.refills}
    return mr

def to_fhir_bundle(doc: OrdoDoc, bundle_id: str = "") -> Dict:
    if not bundle_id:
        bundle_id = f"ordo-{abs(hash((doc.patient_name, doc.date_str))) % 10_000_000}"
    entries = [{"resource": lineitem_to_medication_request(doc, li, i, bundle_id)} for i, li in enumerate(doc.lines, 1)]
    return {"resourceType": "Bundle", "type": "collection", "id": bundle_id, "entry": entries}

def load_catalog_mimic(path: str) -> List[Dict]:
    print(f"Loading catalog from {path}...")
    df = pd.read_csv(path).fillna("")
    return [row.to_dict() for _, row in df.iterrows() if row.get("drug")]

# --- RENDERING ENGINE (Self-Contained) ---

def wrap_text_to_width(draw, text, font, max_w):
    text = str(text).strip()
    if not text: return [""]
    words = text.split()
    lines, cur = [], ""
    def fits(s): return text_wh(draw, s, font)[0] <= max_w
    
    for w in words:
        test = (cur + " " + w) if cur else w
        if fits(test):
            cur = test
        else:
            if cur: lines.append(cur)
            cur = w if fits(w) else w # rough handling of super long words
    if cur: lines.append(cur)
    return lines

def draw_wrapped(draw, x, y, text, font, fill, max_w, line_gap=6):
    lines = wrap_text_to_width(draw, text, font, max_w)
    yy = y
    minx, miny, maxx, maxy = 1e9, 1e9, -1e9, -1e9
    for ln in lines:
        draw.text((x, yy), ln, fill=fill, font=font)
        w, h = text_wh(draw, ln, font)
        minx, miny = min(minx, x), min(miny, yy)
        maxx, maxy = max(maxx, x+w), max(maxy, yy+h)
        yy += h + line_gap
    return yy - line_gap, [minx, miny, maxx-minx, maxy-miny]

def render_visual(doc: OrdoDoc, template_cfg: Dict) -> Tuple[Image.Image, Dict[str, List]]:
    """
    Renders the ordonnance using the specific template configuration.
    Returns the image and a dictionary of YOLO-compatible boxes (unnormalized).
    """
    img = Image.open(template_cfg["path"]).convert("RGB")
    W, H = img.size
    
    # Text Area
    fx0, fy0, fx1, fy1 = template_cfg.get("text_area_frac", (0.1, 0.25, 0.9, 0.8))
    area_x0, area_y0 = int(fx0 * W), int(fy0 * H)
    area_x1, area_y1 = int(fx1 * W), int(fy1 * H)
    area_W = area_x1 - area_x0
    
    draw = ImageDraw.Draw(img)
    
    # Scale fonts
    scale = max(0.8, area_W / (1240 * 0.55))
    font_main = pick_font(CANDIDATE_FONTS_TYPED, int(30 * scale))
    font_small = pick_font(CANDIDATE_FONTS_TYPED, int(24 * scale))
    
    boxes = {"header": [], "medication_block": [], "footer": [], "medication_item": []}
    
    y = area_y0
    
    # 1. Header (Patient, Dr, Date)
    header_start_y = y
    header_fields = [
        ("Patient", doc.patient_name),
        ("Dr", doc.prescriber_name),
        ("Date", doc.date_str)
    ]
    random.shuffle(header_fields)
    
    for label, text in header_fields:
        full_text = f"{label}: {text}"
        y_end, box = draw_wrapped(draw, area_x0, y, full_text, font_main, "black", area_W)
        boxes["header"].append(box)
        y = y_end + 10
        
    y += 20
    draw.line((area_x0, y, area_x1, y), fill="grey")
    y += 30
    
    # 2. Medication Block
    med_start_y = y
    
    for i, line in enumerate(doc.lines, 1):
        # Drug Name
        txt_drug = f"{i}. {line.drug_name} {line.strength or ''}"
        y_end, box_drug = draw_wrapped(draw, area_x0, y, txt_drug, font_main, "black", area_W)
        boxes["medication_item"].append(box_drug)
        y = y_end + 5
        
        # Details (Posology/Form)
        details = []
        if line.posology.form: details.append(f"Forme: {line.posology.form}")
        if line.posology.route: details.append(f"Voie: {line.posology.route}")
        if details:
            y_end, box_det = draw_wrapped(draw, area_x0 + 20, y, "  ".join(details), font_small, "black", area_W - 20)
            boxes["medication_item"].append(box_det) # Treat as part of item
            y = y_end + 5

        # Instruction
        poso_txt = f"Posologie: {line.posology.dose} {line.posology.frequency}"
        y_end, box_pos = draw_wrapped(draw, area_x0 + 20, y, poso_txt, font_small, "black", area_W - 20)
        boxes["medication_item"].append(box_pos)
        y = y_end + 20
        
    med_end_y = y
    
    # Add full block box
    boxes["medication_block"].append([area_x0, med_start_y, area_W, med_end_y - med_start_y])

    # 3. Footer (Signature)
    sig_y = area_y1 - 50
    draw.text((area_x1 - 200, sig_y), "Signature:", fill="black", font=font_small)
    draw.line((area_x1 - 200, sig_y + 30, area_x1, sig_y + 30), fill="black")
    boxes["footer"].append([area_x1 - 200, sig_y, 200, 50])
    
    return img, boxes

def normalize_box(box, W, H):
    x, y, w, h = box
    return [(x + w/2)/W, (y + h/2)/H, w/W, h/H]

# --- MAIN GENERATOR CLASS ---

class VisualGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.img_dir = output_dir / "images"
        self.label_dir = output_dir / "labels"
        self.debug_dir = output_dir / "debug_vis"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def generate_batch(self, count: int, catalog: list, args):
        print(f"[{time.strftime('%H:%M:%S')}] Starting Self-Contained Visual Generation: {count} images")
        
        # Custom Templates
        # Resolve relative to this script: visual/generation/../../data/templates
        template_dir = (Path(__file__).parent / "../../data/templates").resolve()

        templates = [
            {"path": template_dir / f"template{i}.jpeg", "text_area_frac": (0.1, 0.25, 0.9, 0.8)}
            for i in range(1, 5) if (template_dir / f"template{i}.jpeg").exists()
        ]
        
        if not templates:
            print("WARNING: No data/templates/templateX.jpeg found!")
            return

        start_time = time.time()
        with tqdm(total=count) as pbar:
            for i in range(count):
                # Data Sample
                n_items = random.randint(1, 5)
                lines = []
                for _ in range(n_items):
                    row = random.choice(catalog)
                    lines.append(LineItem(row['drug'], row.get('prod_strength',''), posology_from_mimic(row)))
                
                doc = OrdoDoc(sample_name(), f"Dr. {sample_name()}", sample_date(), lines)
                
                # Render
                template = random.choice(templates)
                img, boxes_dict = render_visual(doc, template)
                
                # Noise
                noise = sample_noise_from_args(args)
                noise['skew'] = 0 # keep strict boxes
                img = apply_noise(img, **noise)
                
                # Save Image
                fid = f"ordo_vis_{i:06d}"
                img.save(self.img_dir / f"{fid}.png")
                
                # Save Raw Boxes (JSON) - Unnormalized
                with open(self.output_dir / f"{fid}.json", "w") as f:
                    json.dump(boxes_dict, f, indent=2)

                # Save FHIR Ground Truth
                fhir_data = to_fhir_bundle(doc, bundle_id=fid)
                with open(self.output_dir / f"{fid}.fhir.json", "w") as f:
                    json.dump(fhir_data, f, indent=2)
                
                # Save YOLO Labels
                W, H = img.size
                with open(self.label_dir / f"{fid}.txt", "w") as f:
                    for cls_name, box_list in boxes_dict.items():
                        cid = CLASS_MAPPING.get(cls_name)
                        if cid is None: continue
                        for box in box_list:
                            nb = normalize_box(box, W, H)
                            f.write(f"{cid} {' '.join(map(str, nb))}\n")
                            
                pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", default="visual/data_generated")
    parser.add_argument("--csv", default="../../data/prescriptions_demo.csv")
    parser.add_argument("--blur", type=float, default=0.2)
    parser.add_argument("--jpeg", type=float, default=0.2)
    parser.add_argument("--stains", type=float, default=0.2)
    parser.add_argument("--skew", type=float, default=0.0)
    args = parser.parse_args()
    
    catalog = load_catalog_mimic(args.csv)
    gen = VisualGenerator(Path(args.out))
    gen.generate_batch(args.count, catalog, args)
