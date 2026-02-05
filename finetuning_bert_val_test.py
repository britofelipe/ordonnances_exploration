import json
import glob
from pathlib import Path
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
from typing import List, Optional

# importa as dataclasses e o gerador de FHIR do script de geração
from generate_ordo_mimic import Posology, LineItem, OrdoDoc, to_fhir_bundle

# =============================
# 1. Carregar pares txt + FHIR
# =============================

DATA_DIR = Path("output_mimic_fhir_ocr_template_prod")  # ajuste para o diretório real

def ordo_to_linear_text(doc: OrdoDoc) -> str:
    lines = []
    lines.append("ORDO")
    lines.append(f"PATIENT: {doc.patient_name}")
    lines.append(f"PRESCRIPTEUR: {doc.prescriber_name}")
    lines.append(f"DATE: {doc.date_str}")
    lines.append("")  # linha em branco

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
        lines.append("")  # linha em branco entre meds

    lines.append("END")
    return "\n".join(lines)

def fhir_bundle_to_ordo(bundle: dict) -> OrdoDoc:
    """
    Reconstrói um OrdoDoc a partir do Bundle FHIR sintético que você gerou.
    Supõe que cada entry é um MedicationRequest criado por to_fhir_bundle.
    """
    entries = bundle.get("entry", [])

    # pegamos o primeiro MR só pra pegar patient/prescriber/date (são iguais em todos)
    patient_name = "UNKNOWN"
    prescriber_name = "UNKNOWN"
    date_str = "1900-01-01"
    lines: List[LineItem] = []

    for e in entries:
        mr = e.get("resource", {})
        if mr.get("resourceType") != "MedicationRequest":
            continue

        # subject / requester / authoredOn
        if patient_name == "UNKNOWN":
            patient_name = mr.get("subject", {}).get("display", "UNKNOWN")
        if prescriber_name == "UNKNOWN":
            prescriber_name = mr.get("requester", {}).get("display", "UNKNOWN")
        if date_str == "1900-01-01":
            date_str = mr.get("authoredOn", "1900-01-01")

        # DRUG + STRENGTH
        med_text = mr.get("medicationCodeableConcept", {}).get("text", "")
        # no seu gerador era "Drug (strength)" — vamos separar por parênteses
        drug_name = med_text
        strength = ""
        if "(" in med_text and med_text.endswith(")"):
            i = med_text.rfind("(")
            drug_name = med_text[:i].strip()
            strength = med_text[i+1:-1].strip()

        # dosageInstruction[0]
        di = (mr.get("dosageInstruction") or [{}])[0]
        route = di.get("route", {}).get("text", "")
        # dose / freq / duration
        poso = Posology(
            dose="",
            frequency="",
            duration="",
            route=route,
            form="",
            as_needed=False,
            as_needed_for=""
        )

        # dose
        if "doseAndRate" in di and di["doseAndRate"]:
            d0 = di["doseAndRate"][0]
            dose_str = d0.get("doseString", "")
            if dose_str:
                poso.dose = dose_str

        # timing / freq
        timing = di.get("timing", {})
        rep = timing.get("repeat", {})
        if "frequency" in rep:
            f = rep["frequency"]
            poso.frequency = f"{f}/j"

        # duration
        if "boundsDuration" in rep:
            bd = rep["boundsDuration"]
            val = bd.get("value")
            unit = bd.get("unit", "d")
            if val is not None:
                # unit "d" já significa dias
                poso.duration = f"{int(val)} jours"

        # form — na sua serialização não está exposta diretamente, então ignoramos
        # ou, se quiser, pode tentar recuperar do texto, mas não é obrigatório
        poso.form = ""

        # refills
        refills = None
        disp = mr.get("dispenseRequest", {})
        if "numberOfRepeatsAllowed" in disp:
            try:
                refills = int(disp["numberOfRepeatsAllowed"])
            except Exception:
                refills = None

        li = LineItem(
            drug_name=drug_name,
            strength=strength,
            posology=poso,
            refills=refills,
        )
        lines.append(li)

    return OrdoDoc(
        patient_name=patient_name,
        prescriber_name=prescriber_name,
        date_str=date_str,
        lines=lines,
    )

def linear_text_to_ordo(text: str) -> OrdoDoc:
    patient = ""
    prescriber = ""
    date_str = ""
    items: List[LineItem] = []

    current: Optional[LineItem] = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # cabeçalho
        if line.startswith("PATIENT:"):
            patient = line[len("PATIENT:"):].strip()
        elif line.startswith("PRESCRIPTEUR:"):
            prescriber = line[len("PRESCRIPTEUR:"):].strip()
        elif line.startswith("DATE:"):
            date_str = line[len("DATE:"):].strip()

        # blocos de medicamento
        elif line == "MED_START":
            current = LineItem(
                drug_name="",
                strength="",
                posology=Posology(
                    dose="",
                    frequency="",
                    duration="",
                    route="",
                    form="",
                    as_needed=False,
                    as_needed_for="",
                ),
                refills=None,
            )
        elif line == "MED_END":
            if current is not None and current.drug_name:
                items.append(current)
            current = None

        # campos internos
        elif current is not None:
            if line.startswith("DRUG:"):
                current.drug_name = line[len("DRUG:"):].strip()
            elif line.startswith("STRENGTH:"):
                current.strength = line[len("STRENGTH:"):].strip()
            elif line.startswith("FORM:"):
                current.posology.form = line[len("FORM:"):].strip()
            elif line.startswith("ROUTE:"):
                current.posology.route = line[len("ROUTE:"):].strip()
            elif line.startswith("DOSE:"):
                current.posology.dose = line[len("DOSE:"):].strip()
            elif line.startswith("FREQ:"):
                current.posology.frequency = line[len("FREQ:"):].strip()
            elif line.startswith("DURATION:"):
                current.posology.duration = line[len("DURATION:"):].strip()
            elif line.startswith("REFILLS:"):
                val = line[len("REFILLS:"):].strip()
                try:
                    current.refills = int(val)
                except ValueError:
                    current.refills = None

    if not patient:
        patient = "UNKNOWN"
    if not prescriber:
        prescriber = "UNKNOWN"
    if not date_str:
        date_str = "1900-01-01"

    return OrdoDoc(
        patient_name=patient,
        prescriber_name=prescriber,
        date_str=date_str,
        lines=items,
    )

def load_one_pair(txt_path: Path):
    json_path = txt_path.with_suffix(".fhir.json")
    with open(txt_path, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    with open(json_path, "r", encoding="utf-8") as f:
        fhir = json.load(f)

    # serialização CANÔNICA do FHIR -> string (sempre mesma ordem de chaves)
    #target_text = json.dumps(
    #    fhir,
    #    ensure_ascii=False,
    #    sort_keys=True,
    #    separators=(",", ":")  # tira espaços para padronizar
    #)
    doc = fhir_bundle_to_ordo(fhir)
    target_text = ordo_to_linear_text(doc)
    return {"input_text": input_text, "target_text": target_text}

all_pairs = []
for txt_file in DATA_DIR.glob("*.txt"):
    all_pairs.append(load_one_pair(txt_file))

print(len(all_pairs), "exemplos")
print(all_pairs[0]["input_text"][:300])
print(all_pairs[0]["target_text"][:300])

# =============================
# 2. Dataset -> train / val / test
# =============================

dataset = Dataset.from_list(all_pairs)

### NOVO: primeiro separa TEST (10%)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
test_ds = dataset["test"]
train_val = dataset["train"]

### NOVO: dentro do "train", separa VAL (10% do restante ≈ 9% do total)
train_val = train_val.train_test_split(test_size=0.1, seed=42)
train_ds = train_val["train"]
val_ds = train_val["test"]

print("train:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))

# =============================
# 3. Modelo encoder-decoder (BERT2BERT)
# =============================

model_name = "camembert-base"  # BERT francês

tokenizer = AutoTokenizer.from_pretrained(model_name)

# cria encoder-decoder usando BERT nos dois lados
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    model_name, model_name
)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

# IMPORTANTE: amarra embeddings e cria lm_head consistente
model.config.tie_word_embeddings = True
model.tie_weights()


# =============================
# 4. Tokenização
# =============================

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 512

def preprocess_batch(batch):
    # encoder
    inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    )
    # decoder (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text_target=batch["target_text"], 
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
        )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    label_ids = labels["input_ids"]

    # substitui tokens de padding por -100 para ignorar na loss
    label_ids = [
        [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
        for seq in label_ids
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_ids,
    }

tokenized_train = train_ds.map(
    preprocess_batch,
    batched=True,
    remove_columns=train_ds.column_names,
)
tokenized_val = val_ds.map(
    preprocess_batch,
    batched=True,
    remove_columns=val_ds.column_names,
)
tokenized_test = test_ds.map(          # NOVO: tokenizar test também
    preprocess_batch,
    batched=True,
    remove_columns=test_ds.column_names,
)

# =============================
# 5. Trainer + argumentos
# =============================

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./toobib-ordo-bert2bert-prod-100000",
    eval_strategy="epoch",          # avalia no conjunto de validação a cada época
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_exact_match",
    greater_is_better=True,
    save_total_limit=3, 
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    # Se estiver em GPU moderna, pode deixar bf16=True.
    # Se estiver em CPU ou GPU antiga, comente a linha abaixo:
    # bf16=True,
    # Para forçar CPU:
    # no_cuda=True,
)

# métrica simples: igualdade exata do JSON serializado
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # Em seq2seq com predict_with_generate=True, às vezes preds vem como tuple
    if isinstance(preds, tuple):
        preds = preds[0]

    # Garante que são arrays NumPy inteiros
    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    # Sanitize: remove valores inválidos de preds (negativos ou >= vocab_size)
    vocab_size = tokenizer.vocab_size
    preds = np.where((preds < 0) | (preds >= vocab_size),
                     tokenizer.pad_token_id,
                     preds)

    # Nos labels, troca -100 por pad_token_id para poder decodificar
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Agora podemos decodificar com segurança
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact = [int(p == t) for p, t in zip(pred_str, label_str)]
    return {"exact_match": float(np.mean(exact))}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,             # usa VAL para avaliação durante treino
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# =============================
# 6. Treinar
# =============================

trainer.train()

# garanta que o modelo que você quer salvar é o melhor/final
trainer.save_model("./toobib-ordo-bert2bert-prod-100000")
tokenizer.save_pretrained("./toobib-ordo-bert2bert-prod-100000")


# =============================
# 7. Avaliação FINAL no test set
# =============================

print("== Avaliação final no TESTE ==")
test_metrics = trainer.evaluate(
    eval_dataset=tokenized_test,
    metric_key_prefix="test",   # as chaves viram "test_loss", "test_exact_match", etc.
)
print(test_metrics)

# =============================
# 8. Inferência em um exemplo
# =============================

from transformers import EncoderDecoderModel, AutoTokenizer

model_path = "./toobib-ordo-bert2bert-prod-100000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = EncoderDecoderModel.from_pretrained(model_path)
model.eval()

def ordo_txt_to_fhir_json(txt: str):
    # tokenização normal
    inputs = tokenizer(
        txt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )
    # garantir que está no mesmo device que o modelo (CPU ou CUDA)
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=MAX_TARGET_LEN,
            num_beams=4,
        )
    gen_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # 1) texto gerado está na DSL (ORDO / MED_START / ...)
    # 2) convertemos DSL -> OrdoDoc
    doc_pred = linear_text_to_ordo(gen_text)

    # 3) usamos sua função existente para gerar FHIR JSON
    fhir_bundle = to_fhir_bundle(doc_pred, bundle_id="inference-0")

    return gen_text, fhir_bundle

# exemplo: mesma ordo_0002
with open("output_mimic_fhir_ocr_template_prod/ordo_0002.txt", encoding="utf-8") as f:
    txt = f.read()
linear_str, fhir = ordo_txt_to_fhir_json(txt)

print("Texto DSL gerado (primeiros 400 chars):")
print(linear_str[:400])

print("\nFHIR JSON (começo):")
print(json.dumps(fhir, ensure_ascii=False, indent=2)[:600])