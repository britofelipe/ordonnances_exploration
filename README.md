# Ordonnances Exploration Project

Este projeto foca na geração de prescrições médicas sintéticas (ordonnances) e no treinamento de modelos para extração estruturada de informações (OCR -> FHIR JSON).

## Estrutura do Projeto

O projeto é dividido em geração de dados, treinamento e avaliação.

### 1. Geração de Dados Sintéticos

O script principal para geração de dados modernos é `generate_ordo_mimic.py`.
Ele utiliza:
- Dados seed do dataset MIMIC-IV (via CSV).
- Templates de fundo reais (em `templates/`).
- Simulação de escrita manual e digitada com várias fontes.
- Aplicação de ruídos (blur, skew, manchas, compressão).

**Como usar:**
```bash
python generate_ordo_mimic.py \
  --csv prescriptions_demo.csv \
  --out output_folder \
  --count 100 \
  --blur 0.5 \
  --skew 2.0
```
Isso gerará imagens (`.png`), anotações FHIR (`.fhir.json`) e um arquivo de anotações COCO.

> **Nota:** Existe um script mais antigo/básico `generate_synthetic_ordonnances.py` (usado por `generate_all.sh`), que gera dados puramente aleatórios sem base no MIMIC. Para o workflow de treinamento atual, prefira usar `generate_ordo_mimic.py`.

### 2. Treinamento (Finetuning)

O script `finetuning_bert_val_test.py` realiza o finetuning de um modelo Camembert (BERT francês) em arquitetura Encoder-Decoder (BERT2BERT) para converter o texto extraído (OCR) em formato estruturado.

**Como usar:**
1. Certifique-se de ter gerado os dados em uma pasta (ex: `output_mimic_fhir_ocr_template_demo_simplified`).
2. Ajuste a variável `DATA_DIR` no script `finetuning_bert_val_test.py` se necessário.
3. Execute:
```bash
python finetuning_bert_val_test.py
```
O script separam, automaticamente, train/val/test, tokeniza os dados e inicia o treinamento com `Seq2SeqTrainer`. O modelo final será salvo em `./toobib-ordo-bert2bert`.

### 3. Avaliação

O script `bert_evaluation.py` carrega o modelo treinado e avalia no conjunto de teste separado, reportando métricas "Exact Match" e exibindo exemplos de gerações.

**Como usar:**
```bash
python bert_evaluation.py
```

### 4. Exploração de Dados (Legacy)

- `main.ipynb`: Notebook exploratório usado para analisar o dataset QUAERO (corpus médico francês real). Pode ser útil para criar baselines ou entender o domínio, mas não é usado diretamente no pipeline sintético principal.

## Dependências

Principais bibliotecas necessárias (verificar imports):
- `transformers`
- `datasets`
- `torch`
- `Pillow` (PIL)
- `pandas`
- `numpy`
- `scikit-learn` (opcional)
