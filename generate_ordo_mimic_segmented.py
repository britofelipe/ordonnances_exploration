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


# ---------------- OCR SERVICE ----------------
class OCRService:
    def __init__(self):
        self.lang = "fra"

    def process_file(self, file_path: str) -> str:
        ext = file_path.split(".")[-1].lower()
        extracted_text = ""

        if ext == "pdf":
            images = convert_from_path(file_path)
            for i, img in enumerate(images):
                open_cv_image = np.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                text = self._process_single_image(open_cv_image)
                extracted_text += f"\n--- Page {i+1} ---\n{text}"
        else:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Could not load image at {file_path}")
            extracted_text = self._process_single_image(img)

        return extracted_text

    def _process_single_image(self, img_cv2) -> str:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        config = "--psm 6"
        text = pytesseract.image_to_string(thresh, lang=self.lang, config=config)
        return text.strip()


# --- CONFIGURATION & CONSTANTS ---
CANDIDATE_FONTS_TYPED = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "arial.ttf",
]
CANDIDATE_FONTS_HAND = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "comic.ttf",
]

STYLE_CHOICES = ["typed", "hand", "mixed"]

FORM_KEYWORDS = {
    "tablet": "comprimé",
    "tab": "comprimé",
    "capsule": "gélule",
    "cap": "gélule",
    "solution": "solution",
    "sol": "solution",
    "syrup": "sirop",
    "suspension": "suspension",
    "cream": "pommade",
    "ointment": "pommade",
    "inj": "solution injectable",
    "injection": "solution injectable",
    "ampul": "ampoule",
    "vial": "flacon",
}

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


def map_route_to_french(route_str: str) -> str:
    if not route_str:
        return ""
    r = route_str.strip().upper()
    return ROUTE_MAP_FR.get(r, route_str.lower())


# --- NOISE HELPERS ---
def apply_noise(img: Image.Image, blur=0.0, jpeg=0.0, skew=0.0, stains=0.0):
    if skew > 0:
        ang = random.uniform(-skew, skew)
        img = img.rotate(ang, resample=Image.BICUBIC, expand=1, fillcolor=(255, 255, 255))

    if blur > 0:
        radius = random.uniform(0.5, max(blur, 0.5))
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    if stains > 0:
        dr = ImageDraw.Draw(img)
        W, H = img.size
        n_stains = random.randint(1, max(1, int(3 * stains)))
        for _ in range(n_stains):
            x = random.randint(10, max(11, W - 10))
            y = random.randint(10, max(11, H - 10))
            r = random.randint(6, 20)
            col = (random.randint(200, 240),) * 3
            dr.ellipse((x - r, y - r, x + r, y + r), fill=col, outline=None)

    if jpeg > 0:
        buf = io.BytesIO()
        q_min = max(10, int(100 - 60 * jpeg))
        q = random.randint(q_min, 90)
        img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


def sample_noise_from_args(args) -> Dict[str, float]:
    def sample_intensity(max_val, min_frac=0.3):
        if max_val <= 0:
            return 0.0
        return random.uniform(min_frac * max_val, max_val)

    return {
        "blur": sample_intensity(args.blur),
        "jpeg": sample_intensity(args.jpeg),
        "skew": sample_intensity(args.skew),
        "stains": sample_intensity(args.stains),
    }


# --- DATA CLASSES (ANÔNIMO) ---
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
    # Agora é 100% anônimo: só lista de meds
    lines: List[LineItem]


# --- HELPERS ---
def infer_form(strength_str: str) -> str:
    if not strength_str:
        return ""
    s = strength_str.lower()
    for keyword, french_form in FORM_KEYWORDS.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", s):
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


def jitter(x, y, amp=1.0):
    return x + random.uniform(-amp, amp), y + random.uniform(-amp, amp)


def wrap_text_to_width(draw, text, font, max_w):
    text = str(text).strip()
    if not text:
        return [""]

    words = text.split()
    lines, cur = [], ""

    def fits(s):
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
            # quebra palavra longa
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


def crop_to_content(img: Image.Image, boxes: List[Dict], pad_range=(25, 70)) -> Tuple[Image.Image, List[Dict]]:
    """
    Simula um 'crop feito pelo usuário' recortando ao redor do conteúdo desenhado.
    """
    if not boxes:
        return img, boxes

    W, H = img.size
    xs, ys, xe, ye = [], [], [], []
    for b in boxes:
        x, y, w, h = b["box"]
        xs.append(x)
        ys.append(y)
        xe.append(x + w)
        ye.append(y + h)

    x0 = max(0, int(min(xs)))
    y0 = max(0, int(min(ys)))
    x1 = min(W, int(max(xe)))
    y1 = min(H, int(max(ye)))

    pad = random.randint(pad_range[0], pad_range[1])
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad)
    y1 = min(H, y1 + pad)

    cropped = img.crop((x0, y0, x1, y1))

    # ajusta boxes
    new_boxes = []
    for b in boxes:
        bx, by, bw, bh = b["box"]
        new_b = dict(b)
        new_b["box"] = [bx - x0, by - y0, bw, bh]
        new_boxes.append(new_b)

    return cropped, new_boxes


# --- POSOLOGY FROM MIMIC ---
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


# --- LOAD CSV ---
def load_catalog_mimic(path: str) -> List[Dict]:
    print(f"Loading catalog from {path}...")
    df = pd.read_csv(path).fillna("")

    catalog = []
    for row_idx, row in df.iterrows():
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
            "_row_idx": int(row_idx),
        }
        catalog.append(entry)
    return catalog


# --- FHIR EXPORT (ANÔNIMO) ---
def frequency_to_timing(freq: str) -> Optional[Dict]:
    if not freq:
        return None
    m = re.match(r"\s*(\d+)\s*/\s*j", freq)
    if not m:
        return None
    f = int(m.group(1))
    return {"repeat": {"frequency": f, "period": 1, "periodUnit": "d"}}


def parse_dose_to_quantity(dose: str) -> Optional[Dict]:
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
    if not duration:
        return None
    m = re.match(r"\s*([0-9]+)", duration)
    if not m:
        return None
    value = int(m.group(1))
    return {"value": value, "unit": "d"}


def lineitem_to_medication_request(line: LineItem, idx: int, base_id: str) -> Dict:
    poso = line.posology

    drug_line = line.drug_name
    if line.strength:
        drug_line += f" ({line.strength})"

    details_segments = []
    if poso.form:
        details_segments.append(f"Forme: {poso.form}")
    if poso.route:
        details_segments.append(f"Voie: {poso.route}")

    poso_segments = []
    if poso.dose:
        poso_segments.append(poso.dose)
    if poso.frequency:
        poso_segments.append(poso.frequency)
    if poso.duration:
        poso_segments.append(poso.duration)

    dosage_text = drug_line
    if details_segments:
        dosage_text += " " + " ".join(details_segments)
    if poso_segments:
        dosage_text += " Posologie: " + " ".join(poso_segments)

    dosage_instruction: Dict = {"text": dosage_text, "sequence": idx}

    if poso.as_needed:
        dosage_instruction["asNeededBoolean"] = True
        if poso.as_needed_for:
            dosage_instruction["asNeededCodeableConcept"] = {"text": poso.as_needed_for}

    if poso.route:
        dosage_instruction["route"] = {"text": poso.route}

    timing = frequency_to_timing(poso.frequency)
    bounds = duration_to_bounds_duration(poso.duration)
    if bounds:
        if not timing:
            timing = {"repeat": {}}
        timing.setdefault("repeat", {})
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
        "medicationCodeableConcept": {"text": drug_line},
        "dosageInstruction": [dosage_instruction],
    }

    if line.refills is not None:
        mr["dispenseRequest"] = {"numberOfRepeatsAllowed": line.refills}

    return mr


def to_fhir_bundle(doc: OrdoDoc, bundle_id: str = "") -> Dict:
    if not bundle_id:
        bundle_id = f"list-{random.randint(0, 9_999_999):07d}"

    entries = []
    for idx, li in enumerate(doc.lines, start=1):
        mr = lineitem_to_medication_request(li, idx, bundle_id)
        entries.append({"resource": mr})

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "id": bundle_id,
        "entry": entries,
    }


# --- RENDERING: MED LIST ONLY ---
def render_med_list(
    doc: OrdoDoc,
    style: str,
    noise: Optional[Dict[str, float]] = None,
) -> Tuple[Image.Image, List[Dict]]:
    """
    Gera uma imagem contendo SOMENTE a lista de medicamentos.
    No final, faz crop ao redor do conteúdo para simular um recorte do usuário.
    """
    # canvas grande o suficiente (depois a gente recorta)
    W, H = 1400, 1800
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # fontes maiores (para OCR)
    size_main = random.randint(34, 42)
    font_typed = pick_font(CANDIDATE_FONTS_TYPED, size_main)
    font_hand = pick_font(CANDIDATE_FONTS_HAND, size_main)

    def get_font():
        if style == "hand":
            return font_hand
        if style == "typed":
            return font_typed
        return font_hand if random.random() < 0.6 else font_typed

    # layout
    margin_x = random.randint(60, 120)
    y = random.randint(60, 120)
    max_w = W - 2 * margin_x

    boxes: List[Dict] = []

    list_style = random.choice(["numbered", "hyphen", "plain"])
    line_gap = random.randint(6, 10)
    block_gap = random.randint(18, 26)

    for i, line in enumerate(doc.lines, 1):
        ft = get_font()
        str_txt = f" ({line.strength})" if line.strength else ""

        if list_style == "numbered":
            prefix = f"{i}. "
        elif list_style == "hyphen":
            prefix = "- "
        else:
            prefix = ""

        # Linha principal do medicamento
        txt_main = f"{prefix}{line.drug_name}{str_txt}"
        y, box_main, _ = draw_wrapped(
            draw, margin_x, y, txt_main, ft, "black", max_w=max_w, line_gap=line_gap
        )
        boxes.append({"label": "DRUG", "box": box_main, "text": line.drug_name})
        y += 6

        # Forme / Voie
        details_parts = []
        if line.posology.form:
            details_parts.append(f"Forme: {line.posology.form}")
        if line.posology.route:
            details_parts.append(f"Voie: {line.posology.route}")
        if details_parts:
            txt_det = "   ".join(details_parts)
            y, box_det, _ = draw_wrapped(
                draw, margin_x + 25, y, txt_det, ft, "black", max_w=max_w - 25, line_gap=line_gap
            )
            boxes.append({"label": "DETAILS", "box": box_det, "text": txt_det})
            y += 6

        # Posologie
        poso = line.posology
        txt_pos = "Posologie:"
        segs = []
        if poso.dose:
            segs.append(poso.dose)
        if poso.frequency:
            segs.append(poso.frequency)
        if poso.duration:
            segs.append(poso.duration)
        if segs:
            txt_pos += " " + "  ".join(segs)

        y, box_pos, _ = draw_wrapped(
            draw, margin_x + 25, y, txt_pos, ft, "black", max_w=max_w - 25, line_gap=line_gap
        )
        boxes.append({"label": "INSTRUCT", "box": box_pos, "text": txt_pos})
        y += block_gap

        # stop se estourar muito (evita doc gigante)
        if y > H - 150:
            break

    # "crop como usuário"
    img, boxes = crop_to_content(img, boxes, pad_range=(30, 90))

    # aplica ruído no final (depois do crop, para ficar mais realista)
    if noise:
        img = apply_noise(img, **noise)

    return img, boxes


# --- IMAGE GENERATION ---
def build_doc_from_selection(selection: List[Dict]) -> OrdoDoc:
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

        lines.append(
            LineItem(
                drug_name=drug["drug"],
                strength=drug["prod_strength"],
                posology=pos,
                refills=refills,
            )
        )

    return OrdoDoc(lines=lines)


def generate_doc_from_anchor(anchor_entry: Dict, catalog_segment: List[Dict]) -> OrdoDoc:
    n = random.randint(1, 4)
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

    parser.add_argument("--start_idx", type=int, default=0,
                        help="Índice (0-based) da primeira linha do CSV a ser usada (inclusive).")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Índice (0-based) da última linha do CSV a ser usada (exclusivo). "
                             "Se None, vai até o final do CSV.")
    parser.add_argument("--only_train", action="store_true",
                        help="Se definido, salva apenas TXT (OCR) e FHIR, apagando as imagens depois do OCR "
                             "e não salvando o JSON de boxes.")

    parser.add_argument("--count", type=int, default=10,
                        help="Número máximo de amostras a gerar dentro do intervalo [start_idx, end_idx).")

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

    ocr_service = OCRService()

    start_idx = args.start_idx
    end_idx = args.end_idx
    if end_idx is None:
        max_row = max(e["_row_idx"] for e in catalog) if catalog else -1
        end_idx = max_row + 1

    segment = [e for e in catalog if start_idx <= e["_row_idx"] < end_idx]
    if not segment:
        print(f"Nenhuma linha com drug no intervalo [{start_idx}, {end_idx}). Nada a fazer.")
        exit(0)

    print(f"Usando linhas do CSV no intervalo [{start_idx}, {end_idx}) → {len(segment)} entradas válidas.")

    num_docs = min(args.count, len(segment))
    segment_sorted = sorted(segment, key=lambda e: e["_row_idx"])
    selected_entries = segment_sorted[:num_docs]

    for entry in selected_entries:
        row_idx = entry["_row_idx"]

        style_i = random.choice(STYLE_CHOICES)
        noise_i = sample_noise_from_args(args)

        doc = generate_doc_from_anchor(entry, segment)

        base_name = f"ordo_{row_idx:06d}"

        # Render ONLY med list + crop + noise
        img, boxes = render_med_list(
            doc,
            style=style_i,
            noise=noise_i,
        )

        img_path = os.path.join(args.out, base_name + ".png")
        img.save(img_path)

        try:
            ocr_text = ocr_service.process_file(img_path)
        except Exception as e:
            ocr_text = f"[OCR ERROR] {e}"

        txt_path = os.path.join(args.out, base_name + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ocr_text)

        if not args.only_train:
            with open(os.path.join(args.out, base_name + ".json"), "w", encoding="utf-8") as f:
                json.dump(boxes, f, ensure_ascii=False, indent=2)

        # FHIR ground truth (ANÔNIMO)
        fhir_bundle = to_fhir_bundle(doc, bundle_id=base_name)
        with open(os.path.join(args.out, base_name + ".fhir.json"), "w", encoding="utf-8") as f:
            json.dump(fhir_bundle, f, ensure_ascii=False, indent=2)

        if args.only_train:
            try:
                os.remove(img_path)
            except OSError:
                pass

    print(
        f"Generated {num_docs} med-lists in '{args.out}' "
        f"for CSV lines [{start_idx}, {end_idx}) "
        f"(only_train={args.only_train})"
    )