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

# =============================
# 1. Carregar pares txt + FHIR
# =============================

DATA_DIR = Path("output_mimic_fhir_ocr_demo")  # ajuste para o diretório real

def load_one_pair(txt_path: Path):
    json_path = txt_path.with_suffix(".fhir.json")
    with open(txt_path, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    with open(json_path, "r", encoding="utf-8") as f:
        fhir = json.load(f)

    # serialização CANÔNICA do FHIR -> string (sempre mesma ordem de chaves)
    target_text = json.dumps(
        fhir,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")  # tira espaços para padronizar
    )
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

# configurar tokens especiais
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

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
            batch["target_text"],
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
    output_dir="./toobib-ordo-bert2bert",
    eval_strategy="epoch",          # avalia no conjunto de validação a cada época
    save_strategy="epoch",
    num_train_epochs=5,
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
    # converte -100 de volta para pad_token_id para decodificar
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

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
trainer.save_model("./toobib-ordo-bert2bert")
tokenizer.save_pretrained("./toobib-ordo-bert2bert")

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

model_path = "./toobib-ordo-bert2bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = EncoderDecoderModel.from_pretrained(model_path)
model.eval()

def ordo_txt_to_fhir_json(txt: str):
    inputs = tokenizer(
        txt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )
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

    # tenta converter para JSON; se falhar, você já sabe que o modelo saiu do formato
    try:
        fhir_obj = json.loads(gen_text)
    except json.JSONDecodeError:
        fhir_obj = None
    return gen_text, fhir_obj

# exemplo: mesma ordo_0002
with open("output_mimic_fhir_ocr_demo/ordo_0002.txt", encoding="utf-8") as f:
    txt = f.read()
json_str, fhir = ordo_txt_to_fhir_json(txt)
print("Saída gerada (primeiros 400 chars):")
print(json_str[:400])