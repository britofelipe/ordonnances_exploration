import os
import io
import json
import math
import random
import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd

CANDIDATE_FONTS_TYPED = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
CANDIDATE_FONTS_HAND = [
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
]

def pick_font(candidates, size):
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def text_wh(draw, text, font):
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l), (b - t)
    return draw.textsize(text, font=font)

FALLBACK_DRUGS = [
    {"name":"paracétamol", "form":"comprimé", "strength":"500 mg", "route":"orale"},
    {"name":"amoxicilline", "form":"gélule", "strength":"500 mg", "route":"orale"},
    {"name":"ibuprofène", "form":"comprimé", "strength":"400 mg", "route":"orale"},
    {"name":"ziconotide", "form":"solution", "strength":"25 µg/mL", "route":"intrarachidienne"},
    {"name":"héparine", "form":"solution", "strength":"5000 UI/mL", "route":"SC"},
    {"name":"lévodopa", "form":"comprimé", "strength":"250 mg", "route":"orale"},
]

UNITS = ["mg","g","µg","UI","mL"]
FREQS = ["1/j","2/j","3/j","4/j","q8h","q12h","x2/j","1 le soir","1 matin et soir"]
DURS  = ["5 jours","7 jours","10 jours","14 jours","1 mois"]
ROUTES = ["orale","IV","IM","SC","PO","intrarachidienne","topique","inhalée"]
FORMS  = ["comprimé","gélule","solution","sirop","pommade","suspension"]

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
    total_qty: Optional[str] = None

@dataclass
class OrdoDoc:
    patient_name: str
    prescriber_name: str
    date_str: str
    lines: List[LineItem]

def sample_posology(drug) -> Posology:
    form = drug.get("form", random.choice(FORMS))
    route = drug.get("route", random.choice(ROUTES))
    if "strength" in drug and drug["strength"]:
        dose = drug["strength"]
    else:
        unit = random.choice(UNITS)
        val = random.choice([125,250,400,500,750,1000,5,10,20,25,50])
        dose = f"{val} {unit}"
    frequency = random.choice(FREQS)
    duration = random.choice(DURS)
    return Posology(dose=dose, frequency=frequency, duration=duration, route=route, form=form)

def sample_name() -> str:
    first = random.choice(["Jean","Marie","Lucas","Camille","Léa","Paul","Jules","Zoé","Hugo","Anaïs"])
    last  = random.choice(["Martin","Bernard","Dubois","Thomas","Robert","Petit","Durand","Leroy","Moreau","Simon"])
    return f"{first} {last}"

def sample_date() -> str:
    d = random.randint(1,28)
    m = random.randint(1,12)
    y = random.choice([2023,2024,2025])
    return f"{d:02d}/{m:02d}/{y}"

def load_catalog(path: Optional[str]) -> List[Dict]:
    if not path:
        return FALLBACK_DRUGS
    df = pd.read_csv(path, sep=";", dtype=str).fillna("")
    cols = {c.lower(): c for c in df.columns}
    name_col = None
    for key in ["denomination","libelle","nom","specialite","name","dci"]:
        if key in cols:
            name_col = cols[key]
            break
    form_col = next((cols[k] for k in ["forme_pharmaceutique","forme","form","galenique"] if k in cols), None)
    dose_col = next((cols[k] for k in ["dosage","strength","titrage"] if k in cols), None)
    route_col = next((cols[k] for k in ["voie_administration","route","voie"] if k in cols), None)
    out = []
    for _, row in df.iterrows():
        name = row[name_col] if name_col else ""
        if not name:
            continue
        out.append({
            "name": name,
            "form": (row[form_col] if form_col else ""),
            "strength": (row[dose_col] if dose_col else ""),
            "route": (row[route_col] if route_col else ""),
        })
    return out or FALLBACK_DRUGS

def make_canvas(paper="A5", dpi=150):
    if paper == "A4":
        mm = (210, 297)
    elif paper == "A5":
        mm = (148, 210)
    else:
        mm = (160, 220)
    px = (int(mm[0]/25.4*dpi), int(mm[1]/25.4*dpi))
    img = Image.new("RGB", px, (250,250,245))
    return img

def draw_header(draw: ImageDraw.ImageDraw, W, pad, typed_font):
    title = "ORDONNANCE"
    tw, th = text_wh(draw, title, font=typed_font)
    draw.text(((W - tw)//2, pad), title, fill=(10,10,10), font=typed_font)
    return pad + th + 8

def jitter(x, y, amp=1.0):
    return x + random.uniform(-amp, amp), y + random.uniform(-amp, amp)

def render_ordo(doc: OrdoDoc, paper="A5", style="mixed", dpi=150, noise=None) -> Tuple[Image.Image, List[Dict]]:
    img = make_canvas(paper=paper, dpi=dpi)
    W, H = img.size
    draw = ImageDraw.Draw(img)
    typed = pick_font(CANDIDATE_FONTS_TYPED, size=22)
    typed_small = pick_font(CANDIDATE_FONTS_TYPED, size=16)
    hand = pick_font(CANDIDATE_FONTS_HAND, size=22)
    hand_small = pick_font(CANDIDATE_FONTS_HAND, size=18)
    def fnt(big=False, handwritten=False):
        if handwritten:
            return hand if big else hand_small
        return typed if big else typed_small
    boxes = []
    y = draw_header(draw, W, pad=18, typed_font=fnt(big=True, handwritten=False))
    sections = [
        ("Patient", doc.patient_name),
        ("Prescripteur", doc.prescriber_name),
        ("Date", doc.date_str),
    ]
    for label, val in sections:
        txt = f"{label}: {val}"
        x = 30
        use_hand = (style in ["hand","mixed"]) and (label != "Date")
        draw.text(jitter(x, y, 0.8), txt, fill=(20,20,20), font=fnt(False, handwritten=use_hand))
        tw, th = text_wh(draw, txt, font=fnt(False, handwritten=use_hand))
        boxes.append({"category": label.upper(), "bbox":[x, y, tw, th], "text": val})
        y += th + 8
    draw.line((20, y+4, W-20, y+4), fill=(120,120,120), width=1)
    y += 14
    for idx, li in enumerate(doc.lines, start=1):
        use_hand = (style in ["hand","mixed"])
        drug_txt = f"{idx}. {li.drug_name} {f'({li.strength})' if li.strength else ''}"
        x = 30
        draw.text(jitter(x, y, 0.8), drug_txt, fill=(15,15,15), font=fnt(False, handwritten=use_hand))
        tw, th = text_wh(draw, drug_txt, font=fnt(False, handwritten=use_hand))
        boxes.append({"category":"DRUG", "bbox":[x, y, tw, th], "text": li.drug_name})
        if li.strength:
            stw, _ = text_wh(draw, f"({li.strength})", font=fnt(False, handwritten=use_hand))
            boxes.append({"category":"DOSE", "bbox":[x+tw-stw, y, stw, th], "text": li.strength})
        y += th + 4
        fv_txt = f"Forme: {li.posology.form}   Voie: {li.posology.route}"
        draw.text(jitter(x+24, y, 0.8), fv_txt, fill=(35,35,35), font=fnt(False, handwritten=use_hand))
        tw2, th2 = text_wh(draw, fv_txt, font=fnt(False, handwritten=use_hand))
        boxes.append({"category":"FORM", "bbox":[x+24+6*len('Forme: '), y, 90, th2], "text": li.posology.form})
        boxes.append({"category":"ROUTE", "bbox":[x+24+6*len('Forme: ')+120, y, 80, th2], "text": li.posology.route})
        y += th2 + 2
        poso_txt = f"Posologie: {li.posology.frequency} pendant {li.posology.duration}"
        draw.text(jitter(x+24, y, 0.8), poso_txt, fill=(35,35,35), font=fnt(False, handwritten=use_hand))
        tw3, th3 = text_wh(draw, poso_txt, font=fnt(False, handwritten=use_hand))
        boxes.append({"category":"FREQ", "bbox":[x+24+6*len('Posologie: '), y, 80, th3], "text": li.posology.frequency})
        boxes.append({"category":"DURATION", "bbox":[x+24+6*len('Posologie: ')+130, y, 110, th3], "text": li.posology.duration})
        y += th3 + 8
        if li.total_qty:
            tqt = f"Qté totale: {li.total_qty}"
            draw.text(jitter(x+24, y, 0.8), tqt, fill=(35,35,35), font=fnt(False, handwritten=use_hand))
            tw4, th4 = text_wh(draw, tqt, font=fnt(False, handwritten=use_hand))
            boxes.append({"category":"TOTAL_QTY", "bbox":[x+24+6*len('Qté totale: '), y, 80, th4], "text": li.total_qty})
            y += th4 + 6
        y += 6
        if y > H - 140:
            y += 8
    draw.line((W-220, H-90, W-40, H-90), fill=(80,80,80), width=1)
    draw.text((W-210, H-85), "Signature", fill=(60,60,60), font=fnt(False, handwritten=False))
    if noise:
        img = apply_noise(img, **noise)
    return img, boxes

def apply_noise(img: Image.Image, blur=0.0, jpeg=0.0, skew=0.0, stains=0.0):
    if skew:
        ang = random.uniform(-skew, skew)
        img = img.rotate(ang, resample=Image.BICUBIC, expand=1, fillcolor=(255,255,255))
    if blur and random.random() < blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
    if stains and random.random() < stains:
        dr = ImageDraw.Draw(img)
        W,H = img.size
        for _ in range(random.randint(1,3)):
            x = random.randint(10, W-10)
            y = random.randint(10, H-10)
            r = random.randint(6, 20)
            col = (random.randint(200,240),)*3
            dr.ellipse((x-r, y-r, x+r, y+r), fill=col, outline=None)
    if jpeg and random.random() < jpeg:
        buf = io.BytesIO()
        q = random.randint(40, 80)
        img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img

def to_fhir(doc: OrdoDoc) -> Dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "status": "active",
                    "intent": "order",
                    "subject": {"display": doc.patient_name},
                    "authoredOn": doc.date_str,
                    "requester": {"display": doc.prescriber_name},
                    "note": [{"text":"Synthetic ordonnance"}],
                    "groupIdentifier": {"value": f"ordo-{abs(hash((doc.patient_name, doc.date_str)))%10_000_000}"},
                    "dosageInstruction": [
                        {
                            "text": f"{li.drug_name} {li.strength} — {li.posology.frequency} pendant {li.posology.duration} (voie {li.posology.route}, forme {li.posology.form})"
                        } for li in doc.lines
                    ]
                }
            }
        ]
    }

def coco_from_boxes(img_path: str, boxes: List[Dict], image_id: int, cat_map: Dict[str,int]):
    with Image.open(img_path) as im:
        W, H = im.size
    images = [{"id": image_id, "file_name": os.path.basename(img_path), "width": W, "height": H}]
    annotations = []
    ann_id = image_id*1000
    for b in boxes:
        x,y,w,h = b["bbox"]
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))
        cat = b["category"].upper()
        if cat not in cat_map:
            continue
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_map[cat],
            "bbox":[x,y,w,h],
            "area": w*h,
            "iscrowd":0,
            "text": b.get("text","")
        })
        ann_id += 1
    return images, annotations

CATEGORIES = [
    "PATIENT","PRESCRIPTEUR","DATE","DRUG","DOSE","FORM","ROUTE","FREQ","DURATION","TOTAL_QTY"
]
CAT_MAP = {c:i+1 for i,c in enumerate(CATEGORIES)}

def generate_one(catalog: List[Dict], style="mixed") -> OrdoDoc:
    n_lines = random.randint(1, 4)
    drugs = random.sample(catalog, k=min(n_lines, len(catalog)))
    lines = []
    for d in drugs:
        pos = sample_posology(d)
        total_qty = random.choice(["boîte x1","boîte x2","flacon x1","30 unités","—"])
        if total_qty == "—":
            total_qty = None
        lines.append(LineItem(drug_name=d["name"], strength=d.get("strength",""), posology=pos, total_qty=total_qty))
    return OrdoDoc(
        patient_name=sample_name(),
        prescriber_name=sample_name(),
        date_str=sample_date(),
        lines=lines
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--paper", choices=["A5","A4"], default="A5")
    ap.add_argument("--style", choices=["typed","hand","mixed"], default="mixed")
    ap.add_argument("--catalog", help="CSV BDPM ou equivalente (semicolon-separated)")
    ap.add_argument("--blur", type=float, default=0.4)
    ap.add_argument("--jpeg", type=float, default=0.4)
    ap.add_argument("--skew", type=float, default=2.0)
    ap.add_argument("--stains", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_img = os.path.join(args.out_dir, "images")
    os.makedirs(out_img, exist_ok=True)

    catalog = load_catalog(args.catalog)
    coco = {"images": [], "annotations": [], "categories":[{"id":v,"name":k} for k,v in CAT_MAP.items()]}

    for i in range(1, args.n+1):
        doc = generate_one(catalog, style=args.style)
        img, boxes = render_ordo(
            doc, paper=args.paper, style=args.style,
            noise={"blur":args.blur, "jpeg":args.jpeg, "skew":args.skew, "stains":args.stains}
        )
        img_path = os.path.join(out_img, f"ordo_{i:05d}.png")
        img.save(img_path)
        fhir = to_fhir(doc)
        with open(os.path.join(args.out_dir, f"ordo_{i:05d}.fhir.json"), "w", encoding="utf-8") as f:
            json.dump(fhir, f, ensure_ascii=False, indent=2)
        imgs, anns = coco_from_boxes(img_path, boxes, image_id=i, cat_map=CAT_MAP)
        coco["images"].extend(imgs)
        coco["annotations"].extend(anns)

    with open(os.path.join(args.out_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"OK: {args.n} imagens em {out_img}")
    print(f"COCO: {len(coco['annotations'])} anotações → {os.path.join(args.out_dir,'annotations.json')}")
    print("Exemplo de categorias:", CATEGORIES)

if __name__ == "__main__":
    main()
