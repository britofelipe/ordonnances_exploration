# Projet d'Exploration d'Ordonnances

Ce projet se concentre sur la génération d'ordonnances médicales synthétiques et sur l'entraînement de modèles pour l'extraction structurée d'informations (OCR -> FHIR JSON).

## Structure du Projet

Le projet a été réorganisé pour séparer clairement la génération, l'entraînement et l'évaluation.

```text
.
├── src/
│   ├── generation/     # Scripts de génération de données
│   ├── training/       # Scripts d'entraînement (Fine-tuning)
│   └── evaluation/     # Scripts d'évaluation
├── data/               # Données (CSV, Templates)
└── old/                # Fichiers archivés/légacy
```

## Utilisation

### 1. Génération de Données Synthétiques

Le script principal est `generate_ordo_mimic.py`. Il utilise des données du dataset MIMIC-IV et des templates de fond pour créer des images réalistes.

**Emplacement :** `src/generation/`

**Pré-requis (Environnement) :**
Avant toute commande, activez l'environnement configuré :
```bash
source /etc/profile
module load anaconda3/2022.10/gcc-13.1.0
source activate ordonnances
```

**Comment l'utiliser :**
Il est recommandé d'exécuter le script depuis son répertoire pour garantir que les chemins relatifs (vers `../../data`) fonctionnent correctement.

```bash
cd src/generation
python generate_ordo_mimic.py \
  --count 100 \
  --blur 0.5 \
  --skew 2.0 \
  --out output_folder
```

> **Note :** Le fichier CSV d'entrée par défaut est configuré pour chercher dans `../../data/prescriptions_demo.csv`.

Pour lancer sur le cluster SLURM :
```bash
cd src/generation
sbatch slurm-ordo-gen.sbatch
```

### 2. Entraînement (Fine-tuning)

Les scripts d'entraînement permettent d'affiner un modèle (ex: Camembert BERT2BERT) pour convertir le texte OCR en format FHIR structuré.

**Emplacement :** `src/training/`

**Fichiers principaux :**
- `finetuning_bert_val_test.py` : Script principal (BERT).
- `train_flan_t5_small.py` : Alternative utilisant Flan-T5.

**Comment l'utiliser :**
```bash
cd src/training
python finetuning_bert_val_test.py
```
Assurez-vous d'avoir ajusté la variable `DATA_DIR` dans le script si vos données générées ne sont pas dans le dossier par défaut.

Pour lancer sur le cluster SLURM :
```bash
cd src/training
sbatch slurm-finetuning.sbatch
```

### 3. Évaluation

Ces scripts évaluent la performance du modèle sur un jeu de test, en comparant la sortie prédite avec le JSON attendu.

**Emplacement :** `src/evaluation/`

**Fichiers principaux :**
- `bert_evaluation.py` : Évaluation standard.
- `bert_evaluation_struct.py` : Évaluation plus robuste.

**Comment l'utiliser :**
```bash
cd src/evaluation
python bert_evaluation.py
```

Pour lancer sur le cluster SLURM :
```bash
cd src/evaluation
sbatch slurm-evaluation.sbatch
```

## Dépendances

Les principales bibliothèques (voir `requirements.txt` ou scripts de setup conda) incluent :
- `transformers`
- `datasets`
- `torch`
- `Pillow` (PIL)
- `pandas`
- `numpy`
- `scikit-learn`
