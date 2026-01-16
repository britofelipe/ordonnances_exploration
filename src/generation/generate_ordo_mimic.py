import os
import json
import io
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd
import re

# --- IMPORTS FOR OCR ---
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path

#   How to run this code:
#  python generate_ordo_mimic.py \
#  --csv prescriptions_demo.csv \
#  --out output_mimic_noisy \
#  --count 10 \
#  --blur 2.0 \
#  --jpeg 1.0 \
#  --stains 1.0 \
#  --skew 3.0
#
#

# ---------------- OCR SERVICE ----------------
class OCRService:
    def __init__(self):
        self.lang = 'fra'

    def process_file(self, file_path: str) -> str:
        """
        Ponto de entrada principal: trata PDFs e imagens.
        Retorna o texto extraído.
        """
        ext = file_path.split('.')[-1].lower()
        extracted_text = ""

        if ext == 'pdf':
            # Converts PDF list of PIL Images
            images = convert_from_path(file_path)
            for i, img in enumerate(images):
                # PIL -> OpenCV (numpy)
                open_cv_image = np.array(img)
                # RGB -> BGR
                open_cv_image = open_cv_image[:, :, ::-1].copy()

                text = self._process_single_image(open_cv_image)
                extracted_text += f"\n--- Page {i+1} ---\n{text}"
        else:
            # Its an image file (png, jpg, ...)
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Could not load image at {file_path}")
            extracted_text = self._process_single_image(img)

        return extracted_text

    def _process_single_image(self, img_cv2) -> str:
        """
        Applies vision pre-processing and runs Tesseract.
        """
        # 1. Grayscale
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

        # 2. Denoising (ruído sal e pimenta)
        denoised = cv2.medianBlur(gray, 3)

        # 3. Threshold (binarização, Otsu)
        _, thresh = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 4. OCR
        config = "--psm 6"  # bloco único de texto
        text = pytesseract.image_to_string(thresh, lang=self.lang, config=config)

        return text.strip()

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

FORM_MAP_FR = {
    "VIAL": "flacon",
    "SYRINGE": "seringue",
    "BAG": "sac",
}

def map_route_to_french(route_str: str) -> str:
    if not route_str:
        return ""
    r = route_str.strip().upper()
    return ROUTE_MAP_FR.get(r, route_str.lower())

# --- PAPER / STYLE / NOISE HELPERS ---
PAPER_CHOICES = ["A5", "A4"]
STYLE_CHOICES = ["typed", "hand", "mixed"]
# === BACKGROUNDS DE TEMPLATE ===
TEMPLATE_BACKGROUNDS = [
    {
        "name": "bicetre",
        "path": "../../data/templates/template_bicetre.png",
        "text_area_frac": (0.10, 0.22, 0.95, 0.93),
    },
    {
        "name": "saint_malo",
        "path": "../../data/templates/template_saint_malo.png",
        "text_area_frac": (0.10, 0.22, 0.95, 0.93),
    },
    {
        "name": "centre_sante_1",
        "path": "../../data/templates/template_centre_sante_1.png",
        "text_area_frac": (0.10, 0.16, 0.95, 0.86),
        "date_positions": ["header", "top_right"],  # evita rodapé
    },
    {
        "name": "centre_sante_2",
        "path": "../../data/templates/template_centre_sante_2.png",
        # evita cabeçalho + selo RPPS à esquerda e deixa o bloco central livre
        "text_area_frac": (0.10, 0.22, 0.95, 0.93),
    },
]

def sample_template_and_style():
    """
    Sorteia um background de template e um estilo de fonte.
    """
    template_cfg = random.choice(TEMPLATE_BACKGROUNDS)
    style = random.choice(STYLE_CHOICES)
    return template_cfg, style

def apply_noise(img: Image.Image, blur=0.0, jpeg=0.0, skew=0.0, stains=0.0):
    """
    Apply ALL noise types that have value > 0.
    Now:
      - skew   : maximum rotation amplitude (degrees)
      - blur   : approximate blur intensity (maximum radius)
      - stains : stain intensity (scales the number of stains)
      - jpeg   : compression artefact intensity
    """

    # 1) Rotation / skew: whenever skew > 0
    if skew > 0:
        ang = random.uniform(-skew, skew)
        img = img.rotate(ang, resample=Image.BICUBIC, expand=1,
                         fillcolor=(255, 255, 255))

    # 2) Blur: whenever blur > 0
    if blur > 0:
        # radius between 0.5 and blur
        radius = random.uniform(0.5, max(blur, 0.5))
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 3) Stains: whenever stains > 0
    if stains > 0:
        dr = ImageDraw.Draw(img)
        W, H = img.size
        # number of stains proportional to intensity
        n_stains = random.randint(1, max(1, int(3 * stains)))
        for _ in range(n_stains):
            x = random.randint(10, W - 10)
            y = random.randint(10, H - 10)
            r = random.randint(6, 20)
            col = (random.randint(200, 240),) * 3
            dr.ellipse((x - r, y - r, x + r, y + r), fill=col, outline=None)

    # 4) JPEG artefacts: whenever jpeg > 0
    if jpeg > 0:
        buf = io.BytesIO()
        # the larger jpeg is, the lower the minimum quality → more artefacts
        q_min = max(10, int(100 - 60 * jpeg))  # jpeg=1 → q_min ~ 40
        q = random.randint(q_min, 90)
        img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img

def sample_noise_from_args(args) -> Dict[str, float]:
    """
    For each image, sample a noise configuration.
    Each CLI arg (blur/jpeg/stains) is treated as a maximum probability,
    and skew as a maximum rotation amplitude.
    """
    def sample_intensity(max_val, min_frac=0.3):
        if max_val <= 0:
            return 0.0
        return random.uniform(min_frac * max_val, max_val)

    return {
        "blur":   sample_intensity(args.blur),
        "jpeg":   sample_intensity(args.jpeg),
        "skew":   sample_intensity(args.skew),
        "stains": sample_intensity(args.stains),
    }

def sample_paper_and_style() -> Tuple[str, str]:
    """
    Randomly choose a paper format and a global style for a given ordonnance.
    """
    paper = random.choice(PAPER_CHOICES)
    style = random.choice(STYLE_CHOICES)
    return paper, style

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
    first_names = [
        "Jean", "Marie", "Lucas", "Camille", "Léa", "Paul", "Jules", "Zoé", "Hugo", "Anaïs",
        "Thomas", "Emma", "Louis", "Chloé", "Nathan", "Manon", "Mathis", "Inès", "Arthur", "Clara",
        "Gabriel", "Sarah", "Théo", "Elena", "Noah", "Lucie", "Maxime", "Margaux", "Romain", "Julie",
        "Alexandre", "Nina", "Baptiste", "Élise", "Julien", "Maëlle", "Victor", "Pauline", "Quentin", "Eva"
    ]

    last_names = [
        "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Petit", "Durand", "Leroy", "Moreau", "Simon",
        "Laurent", "Lefebvre", "Roux", "Fournier", "Girard", "Bonnet", "Dupont", "Lambert", "Fontaine", "Rousseau",
        "Vincent", "Muller", "Blanc", "Guerin", "Boyer", "Garnier", "Chevalier", "Francois", "Lopez", "Fernandez",
        "Mercier", "Henry", "Renaud", "Marchand", "Barbier", "Picard", "Gaillard", "Perrot", "Charpentier", "Renault"
    ]

    first = random.choice(first_names)
    last  = random.choice(last_names)
    return f"{first} {last}"

def sample_date():
    """
    Sample a date string, always in day-month-year logical order,
    but with random formatting (separators or month naming).
    """
    d = random.randint(1, 28)
    m = random.randint(1, 12)
    y = 2024

    # Different textual formats, all keeping day-month-year order
    month_names_short = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
                         "juil.", "août", "sept.", "oct.", "nov.", "déc."]

    fmt = random.choice(["slash", "dash", "dot", "short_year", "text"])
    if fmt == "slash":
        return f"{d:02d}/{m:02d}/{y}"
    elif fmt == "dash":
        return f"{d:02d}-{m:02d}-{y}"
    elif fmt == "dot":
        return f"{d:02d}.{m:02d}.{y}"
    elif fmt == "short_year":
        return f"{d:02d}/{m:02d}/{y % 100:02d}"
    else:  # text month
        month_txt = month_names_short[m - 1]
        # e.g. "01 févr. 2024" or "1 févr. 2024"
        return f"{d:02d} {month_txt} {y}"

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
        form_raw = row["form_rx"].strip()
        key = form_raw.upper()
        form = FORM_MAP_FR.get(key, form_raw)
    else:
        form = infer_form(row.get("prod_strength", ""))

    # As needed (PRN)
    as_needed = False
    as_needed_for = ""
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
        as_needed_for=as_needed_for,
    )

# --- LOAD CSV ---
def load_catalog_mimic(path: str) -> List[Dict]:
    print(f"Loading catalog from {path}...")
    df = pd.read_csv(path).fillna("")

    catalog = []
    for row_idx, row in df.iterrows():   # row_idx = índice da linha no CSV original
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
            "form_rx": row.get("form_rx", ""),
            "prn": row.get("prn", ""),
            "refills": row.get("refills", ""),
            "_row_idx": int(row_idx),   # <-- índice da linha no CSV
        }
        catalog.append(entry)

    return catalog

# --- FHIR EXPORT HELPERS ---

def frequency_to_timing(freq: str) -> Dict:
    """
    Convert a frequency string like '3/j' into a FHIR Timing structure.
    Returns a dict or None if we can't parse the frequency.
    """
    if not freq:
        return None
    m = re.match(r"\s*(\d+)\s*/\s*j", freq)
    if not m:
        return None
    f = int(m.group(1))
    return {
        "repeat": {
            "frequency": f,
            "period": 1,
            "periodUnit": "d"
        }
    }


# --- DOSE AND DURATION PARSING HELPERS ---
def parse_dose_to_quantity(dose: str) -> Optional[Dict]:
    """
    Try to parse a dose string like '250 mL' or '1 CAP' into a FHIR Quantity.
    Returns a dict with 'value' and 'unit', or None if parsing fails.
    """
    if not dose:
        return None
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s+(.+)$", dose)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).strip()
    return {"value": value, "unit": unit}


def duration_to_bounds_duration(duration: str) -> Optional[Dict]:
    """
    Parse a duration string like '3 jours' into a FHIR Duration.
    For now we assume duration is expressed in days.
    """
    if not duration:
        return None
    m = re.match(r"\s*([0-9]+)", duration)
    if not m:
        return None
    value = int(m.group(1))
    return {"value": value, "unit": "d"}


def lineitem_to_medication_request(doc: OrdoDoc,
                                   line: LineItem,
                                   idx: int,
                                   base_id: str) -> Dict:
    """
    Build a FHIR MedicationRequest for a single LineItem of the ordonnance.
    """
    # Posology text components
    poso = line.posology

    # 1) Drug line: name (+ strength in parentheses, like on the image)
    drug_line = line.drug_name
    if line.strength:
        drug_line += f" ({line.strength})"
    # This will be used as MedicationRequest.medicationCodeableConcept.text
    med_text = drug_line

    # 2) Details line: Forme / Voie
    details_segments = []
    if poso.form:
        details_segments.append(f"Forme: {poso.form}")
    if poso.route:
        details_segments.append(f"Voie: {poso.route}")

    # 3) Posology line: dose / frequency / duration
    poso_segments = []
    if poso.dose:
        poso_segments.append(poso.dose)
    if poso.frequency:
        poso_segments.append(poso.frequency)
    if poso.duration:
        poso_segments.append(poso.duration)

    # Build a single human-readable text, close to what is written on the ordonnance
    dosage_text = drug_line
    if details_segments:
        dosage_text += " " + " ".join(details_segments)
    if poso_segments:
        dosage_text += " Posologie: " + " ".join(poso_segments)

    # Structured timing (if we can parse frequency)
    timing = frequency_to_timing(poso.frequency)

    dosage_instruction: Dict = {
        "text": dosage_text,
        "sequence": idx
    }
    if poso.as_needed:
        dosage_instruction["asNeededBoolean"] = True
        if poso.as_needed_for:
            dosage_instruction["asNeededCodeableConcept"] = {"text": poso.as_needed_for}
    if poso.route:
        dosage_instruction["route"] = {"text": poso.route}
    # Structured timing (with boundsDuration if available)
    timing = frequency_to_timing(poso.frequency)
    bounds = duration_to_bounds_duration(poso.duration)
    if bounds:
        if not timing:
            timing = {"repeat": {}}
        if "repeat" not in timing:
            timing["repeat"] = {}
        timing["repeat"]["boundsDuration"] = bounds
    if timing:
        dosage_instruction["timing"] = timing
    if poso.dose:
        dose_entry: Dict = {"doseString": poso.dose}
        q = parse_dose_to_quantity(poso.dose)
        if q:
            dose_entry["doseQuantity"] = q
        dosage_instruction["doseAndRate"] = [dose_entry]

    mr = {
        "resourceType": "MedicationRequest",
        "id": f"{base_id}-med-{idx}",
        "status": "active",
        "intent": "order",
        "subject": {"display": doc.patient_name},
        "authoredOn": doc.date_str,
        "requester": {"display": doc.prescriber_name},
        "medicationCodeableConcept": {
            "text": med_text
        },
        "dosageInstruction": [dosage_instruction],
        #"note": [{
        #    "text": "Synthetic ordonnance generated from MIMIC-IV prescriptions"
        #}]
    }
    # Dispense / repeats information, if available
    if line.refills is not None:
        mr["dispenseRequest"] = {
            "numberOfRepeatsAllowed": line.refills
        }
    return mr


def to_fhir_bundle(doc: OrdoDoc, bundle_id: str = "") -> Dict:
    """
    Build a minimal FHIR Bundle (type=collection) containing one
    MedicationRequest per line of the ordonnance.
    """
    if not bundle_id:
        # deterministic-ish id based on patient + date
        bundle_id = f"ordo-{abs(hash((doc.patient_name, doc.date_str))) % 10_000_000}"

    entries = []
    for idx, li in enumerate(doc.lines, start=1):
        mr = lineitem_to_medication_request(doc, li, idx, bundle_id)
        entries.append({"resource": mr})

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "id": bundle_id,
        "entry": entries
    }
    return bundle

# --- RENDERING ENGINE ---
def make_canvas(paper="A5", dpi=150):
    w_mm, h_mm = (148, 210) if paper == "A5" else (210, 297)
    px = (int(w_mm/25.4*dpi), int(h_mm/25.4*dpi))
    return Image.new("RGB", px, (250, 250, 245))

def draw_header(draw, x0, x1, y, font):
    title = "ORDONNANCE"
    tw, th = text_wh(draw, title, font)
    area_w = x1 - x0
    x = x0 + (area_w - tw) // 2
    draw.text((x, y), title, fill="black", font=font)

    title_box = [float(x), float(y), float(tw), float(th)]
    return y + th + 40, title_box

def render_ordo(doc: OrdoDoc, paper="A5", style="typed",
                noise=None, template_cfg=None) -> Tuple[Image.Image, List[Dict]]:
    # Se não houver template, usa canvas em branco A4/A5 como antes
    if template_cfg is None:
        img = make_canvas(paper=paper)
        W, H = img.size
        # área inteira da página
        area_x0, area_y0, area_x1, area_y1 = 40, 50, W - 40, H - 80
    else:
        # Carrega a imagem de background
        img = Image.open(template_cfg["path"]).convert("RGB")
        W, H = img.size
        fx0, fy0, fx1, fy1 = template_cfg.get(
            "text_area_frac", (0.40, 0.18, 0.95, 0.92)
        )
        area_x0 = int(fx0 * W)
        area_y0 = int(fy0 * H)
        area_x1 = int(fx1 * W)
        area_y1 = int(fy1 * H)

    area_W = area_x1 - area_x0
    draw = ImageDraw.Draw(img)

    ref_area_w = 1240.0 * 0.55  # ~largura típica da área editável (0.40->0.95) em templates 1240px
    scale = max(0.8, area_W / ref_area_w)

    size_main  = max(30, int(36 * scale))
    size_small = max(24, int(28 * scale))  # pode manter p/ assinatura etc.

    font_typed = pick_font(CANDIDATE_FONTS_TYPED, size_main)
    font_hand  = pick_font(CANDIDATE_FONTS_HAND,  size_main)
    font_small = pick_font(CANDIDATE_FONTS_TYPED, size_small)

    def get_font(is_small=False):
        if style == "hand":
            base = font_hand
        elif style == "typed":
            base = font_typed
        else:
            base = font_hand if random.random() < 0.6 else font_typed
        return font_small if is_small else base

    def wrap_text_to_width(draw, text, font, max_w):
        text = str(text).strip()
        if not text:
            return [""]
        words = text.split()
        lines, cur = [], ""

        def fits(s):  # width check
            return text_wh(draw, s, font)[0] <= max_w

        for w in words:
            test = w if not cur else (cur + " " + w)
            if fits(test):
                cur = test
                continue

            if cur:
                lines.append(cur)
                cur = ""

            if fits(w):
                cur = w
            else:
                # quebra palavra longa (sem espaços) por caractere
                chunk = ""
                for ch in w:
                    test2 = chunk + ch
                    if fits(test2):
                        chunk = test2
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                cur = chunk

        if cur:
            lines.append(cur)
        return lines

    def draw_wrapped(draw, x, y, text, font, fill, max_w, line_gap=6, do_jitter=True):
        lines = wrap_text_to_width(draw, text, font, max_w)
        y_cursor = y
        minx, miny = 1e9, 1e9
        maxx, maxy = -1e9, -1e9

        for ln in lines:
            xj, yj = (jitter(x, y_cursor) if do_jitter else (x, y_cursor))
            draw.text((xj, yj), ln, fill=fill, font=font)
            w, h = text_wh(draw, ln, font)
            minx, miny = min(minx, xj), min(miny, yj)
            maxx, maxy = max(maxx, xj + w), max(maxy, yj + h)
            y_cursor += h + line_gap

        y_end = y_cursor - line_gap
        box = [float(minx), float(miny), float(maxx - minx), float(maxy - miny)]
        return y_end, box, lines

    def boxes_intersect(a, b, pad=0):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (
            (ax + aw + pad) < (bx - pad) or
            (bx + bw + pad) < (ax - pad) or
            (ay + ah + pad) < (by - pad) or
            (by + bh + pad) < (ay - pad)
        )
        
    boxes = []

    # Escolhas de layout (iguais ao original)
    list_style = random.choice(["numbered", "hyphen", "plain"])
    header_align = random.choice(["left", "center", "right"])
    date_choices = ["header", "top_right", "footer"]
    if template_cfg and template_cfg.get("date_positions"):
        date_choices = template_cfg["date_positions"]
    date_position = random.choice(date_choices)
    signature_style = random.choice(["none", "line_only", "label_and_line"])

    # === AQUI começa a desenhar dentro da área em branco ===
    area_W = area_x1 - area_x0
    y = area_y0

    y, title_box = draw_header(draw, area_x0, area_x1, y, font_typed)
    y_header_start = y

    # Header: patient, (optionally) date, prescriber (requester)
    header_fields = [
        ("Patient", doc.patient_name),
        ("Prescripteur", doc.prescriber_name),
    ]
    if date_position == "header":
        header_fields.append(("Date", doc.date_str))

    random.shuffle(header_fields)

    for label, text in header_fields:
        ft = get_font()
        full_text = f"{label}: {text}"

        # pré-wrap para calcular alinhamento por largura máxima das linhas
        tmp_lines = wrap_text_to_width(draw, full_text, ft, area_W)
        max_line_w = max(text_wh(draw, ln, ft)[0] for ln in tmp_lines) if tmp_lines else 0

        if header_align == "left":
            x = area_x0
        elif header_align == "right":
            x = area_x1 - max_line_w
        else:
            x = area_x0 + (area_W - max_line_w) // 2

        y_end, box, _ = draw_wrapped(draw, x, y, full_text, ft, "black", area_x1 - x, line_gap=6)
        boxes.append({"label": label.upper(), "box": box, "text": str(text)})
        y = y_end + 10

    # Data no topo direito dentro da área
    if date_position == "top_right":
        ft = get_font(is_small=True)  # data menor para não competir com o título
        full_text = f"Date: {doc.date_str}"
        tw, th = text_wh(draw, full_text, ft)

        x = area_x1 - tw
        # tenta alinhar com o título primeiro
        y_date = title_box[1]

        date_box = [float(x), float(y_date), float(tw), float(th)]
        # se colidir com o título, joga para baixo do título (dentro do "gap" de 40px)
        if boxes_intersect(date_box, title_box, pad=6):
            y_date = title_box[1] + title_box[3] + 6

        # garante que fica acima do começo dos headers (Patient/Prescripteur)
        y_date = min(y_date, y_header_start - th - 4)
        y_date = max(area_y0, y_date)

        # aqui eu recomendo NÃO aplicar jitter na data, pra não reintroduzir colisão
        draw.text((x, y_date), full_text, fill="black", font=ft)
        boxes.append({"label": "DATE", "box": [x, y_date, tw, th], "text": doc.date_str})

    y += 20
    # linha separadora só dentro da área
    draw.line((area_x0, y, area_x1, y), fill="grey")
    y += 30

    # === LINHAS DE MEDICAMENTOS ===
    for i, line in enumerate(doc.lines, 1):
        ft = get_font()
        str_txt = f" ({line.strength})" if line.strength else ""

        if list_style == "numbered":
            prefix = f"{i}. "
        elif list_style == "hyphen":
            prefix = "- "
        else:
            prefix = ""

        # --- linha do medicamento (WRAP) ---
        txt_main = f"{prefix}{line.drug_name}{str_txt}"
        x_main = area_x0
        y_end, box_main, _ = draw_wrapped(
            draw, x_main, y, txt_main, ft, "black",
            max_w=(area_x1 - x_main),
            line_gap=6
        )
        boxes.append({"label": "DRUG", "box": box_main, "text": line.drug_name})
        y = y_end + 8

        # --- Forme / Voie (MESMO TAMANHO E COR do medicamento) ---
        details_parts = []
        if line.posology.form:
            details_parts.append(f"Forme: {line.posology.form}")
        if line.posology.route:
            details_parts.append(f"Voie: {line.posology.route}")

        if details_parts:
            txt_det = "   ".join(details_parts)
            x_det = area_x0 + 20
            y_end, box_det, _ = draw_wrapped(
                draw, x_det, y, txt_det, ft, "black",
                max_w=(area_x1 - x_det),
                line_gap=6
            )
            boxes.append({"label": "DETAILS", "box": box_det, "text": txt_det})
            y = y_end + 8

        # --- Posologia (MESMO TAMANHO E COR do medicamento) ---
        txt_pos = f"Posologie: {line.posology.dose}  {line.posology.frequency}  {line.posology.duration}".strip()
        x_pos = area_x0 + 20
        y_end, box_pos, _ = draw_wrapped(
            draw, x_pos, y, txt_pos, ft, "black",
            max_w=(area_x1 - x_pos),
            line_gap=6
        )
        boxes.append({"label": "INSTRUCT", "box": box_pos, "text": txt_pos})
        y = y_end + 18

        # (não estamos controlando estritamente se passa de area_y1, mas em geral cabe bem)

    # Data no rodapé (fora ou dentro da área: aqui uso a largura total)
    if date_position == "footer":
        ft = get_font(is_small=True) if template_cfg is not None else get_font()
        full_text = f"Date: {doc.date_str}"
        tw, th = text_wh(draw, full_text, ft)

        if template_cfg is not None:
            # coloca no canto inferior direito DENTRO da área editável (não no rodapé global)
            pad = 10
            x = area_x1 - tw - pad
            y_footer = area_y1 - th - pad
        else:
            x = W - 40 - tw
            y_footer = H - 40 - th

        xj, yj = jitter(x, y_footer)
        draw.text((xj, yj), full_text, fill="black", font=ft)
        boxes.append({"label": "DATE", "box": [xj, yj, tw, th], "text": doc.date_str})

    # Linha de assinatura como antes (no canto inferior direito global)
    if signature_style != "none":
        line_y = H - 80
        draw.line((W - 220, line_y, W - 40, line_y), fill="grey", width=2)
        if signature_style == "label_and_line":
            ft_sig = get_font(is_small=True)
            draw.text((W - 220, line_y - 20), "Signature", fill="black", font=ft_sig)

    # Noise na imagem final (texto + template)
    if noise:
        img = apply_noise(img, **noise)

    return img, boxes

# --- IMAGE GENERATION ---
def build_doc_from_selection(selection: List[Dict]) -> OrdoDoc:
    """
    Cria um OrdoDoc a partir de uma lista de entries do catálogo.
    """
    lines = []
    for drug in selection:
        pos = posology_from_mimic(drug)
        refills_raw = drug.get("refills", "")
        refills: Optional[int] = None
        if refills_raw not in ["", None]:
            try:
                refills = int(float(refills_raw))
            except Exception:
                refills = None
        lines.append(LineItem(
            drug_name=drug["drug"],
            strength=drug["prod_strength"],
            posology=pos,
            refills=refills
        ))

    return OrdoDoc(sample_name(), sample_name(), sample_date(), lines)


def generate_doc_from_anchor(anchor_entry: Dict, catalog_segment: List[Dict]) -> OrdoDoc:
    """
    Gera uma ordonnance onde:
      - anchor_entry é garantidamente um dos medicamentos
      - opcionalmente adiciona mais alguns meds do mesmo segmento, aleatórios
    """
    # número total de medicamentos nesta ordonnance (1 a 4)
    n = random.randint(1, 4)

    # outros candidatos (do mesmo segmento), exceto o anchor
    others = [e for e in catalog_segment if e is not anchor_entry]

    extra = []
    if n > 1 and len(others) > 0:
        k = min(n - 1, len(others))
        extra = random.sample(others, k=k)

    selection = [anchor_entry] + extra
    return build_doc_from_selection(selection)

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to MIMIC prescriptions CSV")
    parser.add_argument("--out", default="output_mimic", help="Output directory")

    # NOVO: intervalo de linhas do CSV (índice da linha no arquivo original, 0-based)
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Índice (0-based) da primeira linha do CSV a ser usada (inclusive)."
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Índice (0-based) da última linha do CSV a ser usada (exclusivo). "
             "Se None, vai até o final do CSV."
    )
    parser.add_argument(
        "--only_train",
        action="store_true",
        help="Se definido, salva apenas TXT (OCR) e FHIR (LLM ground truth), "
             "apagando as imagens depois do OCR e não salvando o JSON de boxes."
    )

    # Mantemos count como limite máximo de ordonnances dentro desse intervalo
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Número máximo de ordonnances a gerar dentro do intervalo [start_idx, end_idx)."
    )

    # Noise / degradation parameters (interpreted as maxima)
    parser.add_argument("--blur", type=float, default=0.4,
                        help="Maximum blur probability per image (0-1).")
    parser.add_argument("--jpeg", type=float, default=0.4,
                        help="Maximum JPEG artefact probability per image (0-1).")
    parser.add_argument("--skew", type=float, default=2.0,
                        help="Maximum rotation (degrees) for skew per image.")
    parser.add_argument("--stains", type=float, default=0.2,
                        help="Maximum stains probability per image (0-1).")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    catalog = load_catalog_mimic(args.csv)
    print(f"Loaded {len(catalog)} drugs (com linhas válidas).")

    # Instância única do serviço de OCR
    ocr_service = OCRService()

    # --- Selecionar intervalo do CSV pelo _row_idx ---
    start_idx = args.start_idx
    end_idx = args.end_idx

    if end_idx is None:
        # Se não foi dado, vai até a maior linha observada + 1
        max_row = max(e["_row_idx"] for e in catalog) if catalog else -1
        end_idx = max_row + 1

    # Filtra só as entradas cujo _row_idx está em [start_idx, end_idx)
    segment = [e for e in catalog if start_idx <= e["_row_idx"] < end_idx]

    if not segment:
        print(f"Nenhuma linha com drug no intervalo [{start_idx}, {end_idx}). Nada a fazer.")
        exit(0)

    print(f"Usando linhas do CSV no intervalo [{start_idx}, {end_idx}) → {len(segment)} entradas válidas.")

    # Quantas ordonnances vamos gerar neste intervalo?
    num_docs = min(args.count, len(segment))
    # Vamos pegar as primeiras num_docs entradas do segmento, em ordem de linha
    segment_sorted = sorted(segment, key=lambda e: e["_row_idx"])
    selected_entries = segment_sorted[:num_docs]

    for entry in selected_entries:
        row_idx = entry["_row_idx"]  # índice da linha no CSV original

        # Sorteia template / estilo / ruído por imagem
        template_cfg, style_i = sample_template_and_style()
        noise_i = sample_noise_from_args(args)

        # Synthetic ordonnance document (ancorada nesta linha do CSV)
        doc = generate_doc_from_anchor(entry, segment)

        # File base name: usa o índice da linha → não sobrescreve entre rodadas
        base_name = f"ordo_{row_idx:06d}"

        # Render image + layout boxes, usando o template como background
        img, boxes = render_ordo(
            doc,
            paper="A4",              # irrelevante quando template_cfg != None
            style=style_i,
            noise=noise_i,
            template_cfg=template_cfg,
        )

        # --- IMAGEM: necessária para OCR, mas pode ser apagada depois ---
        img_path = os.path.join(args.out, base_name + ".png")
        img.save(img_path)

        # Run OCR on the saved image and save extracted text
        try:
            ocr_text = ocr_service.process_file(img_path)
        except Exception as e:
            ocr_text = f"[OCR ERROR] {e}"

        txt_path = os.path.join(args.out, base_name + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ocr_text)

        # Save layout / boxes (for OCR / detection ground truth) – só se NÃO for only_train
        if not args.only_train:
            with open(os.path.join(args.out, base_name + ".json"), "w", encoding="utf-8") as f:
                json.dump(boxes, f, ensure_ascii=False, indent=2)

        # Build and save FHIR Bundle (LLM ground truth) – sempre salva
        fhir_bundle = to_fhir_bundle(doc, bundle_id=base_name)
        with open(os.path.join(args.out, base_name + ".fhir.json"), "w", encoding="utf-8") as f:
            json.dump(fhir_bundle, f, ensure_ascii=False, indent=2)

        # Se for only_train, apagamos a imagem pra não lotar disco
        if args.only_train:
            try:
                os.remove(img_path)
            except OSError:
                pass

    print(f"Generated {num_docs} ordonnances in '{args.out}' "
          f"for CSV lines [{start_idx}, {end_idx}) "
          f"(only_train={args.only_train})")
