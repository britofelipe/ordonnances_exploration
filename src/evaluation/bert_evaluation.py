import json
from pathlib import Path

import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# =============================
# 1. Carregar pares txt + FHIR
# =============================

DATA_DIR = Path("../generation/output_treino")  # mesmo diretório do treino

def load_one_pair(txt_path: Path):
    json_path = txt_path.with_suffix(".fhir.json")
    with open(txt_path, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    with open(json_path, "r", encoding="utf-8") as f:
        fhir = json.load(f)

    # serialização CANÔNICA do FHIR -> string
    target_text = json.dumps(
        fhir,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {"input_text": input_text, "target_text": target_text}

all_pairs = [load_one_pair(p) for p in DATA_DIR.glob("*.txt")]

print(len(all_pairs), "exemplos no total")

# =============================
# 2. Dataset -> train / val / test
#    (mesmas seeds do script de treino)
# =============================

dataset = Dataset.from_list(all_pairs)

# 10% para TEST
dataset = dataset.train_test_split(test_size=0.1, seed=42)
test_ds = dataset["test"]
train_val = dataset["train"]

# 10% do restante para VAL
train_val = train_val.train_test_split(test_size=0.1, seed=42)
train_ds = train_val["train"]
val_ds = train_val["test"]

print("train:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))

# =============================
# 3. Carregar último checkpoint
# =============================

output_dir = "../training/toobib-ordo-bert2bert"

last_checkpoint = get_last_checkpoint(output_dir)
if last_checkpoint is None:
    raise ValueError(f"Nenhum checkpoint encontrado em {output_dir}")

print("Usando checkpoint:", last_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
model = EncoderDecoderModel.from_pretrained(last_checkpoint)
model.eval()

# =============================
# 4. Tokenização do TEST SET
# =============================

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 512

def preprocess_batch(batch):
    inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    )
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

    label_ids = [
        [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
        for seq in label_ids
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_ids,
    }

tokenized_test = test_ds.map(
    preprocess_batch,
    batched=True,
    remove_columns=test_ds.column_names,
)

# =============================
# 5. Trainer só para avaliação
# =============================

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    # se estiver em CPU / GPU antiga:
    # no_cuda=True,
)

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
    train_dataset=None,          # não vamos treinar
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# =============================
# 6. Avaliação FINAL no TESTE
# =============================

print("== Avaliação final no TESTE (último checkpoint) ==")
test_metrics = trainer.evaluate(
    eval_dataset=tokenized_test,
    metric_key_prefix="test",
)
print(test_metrics)

# =============================
# 7. Inspecionar alguns exemplos do TESTE
# =============================

num_examples = 20  # mude se quiser mais/menos exemplos
rng = np.random.default_rng(0)  # seed fixa para reproduzir sempre os mesmos
indices = rng.choice(len(test_ds), size=num_examples, replace=False)

device = model.device  # cpu ou cuda, conforme o Trainer colocou

print("\n== Exemplos de saídas no TESTE ==")
for i, idx in enumerate(indices, start=1):
    sample = test_ds[int(idx)]
    input_text = sample["input_text"]
    target_text = sample["target_text"]

    # gerar com o modelo
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )
    # garantir que tudo está no mesmo device que o modelo
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=MAX_TARGET_LEN,
            num_beams=4,
        )
    pred_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print(f"\n----- Exemplo {i} (idx={idx}) -----")
    print("INPUT (OCR):")
    print(input_text[:600])
    print("\nTARGET (FHIR JSON canonizado):")
    print(target_text[:600])
    print("\nPREDICTION (modelo):")
    print(pred_text[:600])
    print("\n" + "-" * 80)