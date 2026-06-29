# Guide d'exécution sur PC GPU (mode réel)

Le dépôt tourne **partout** en mode synthétique (backends `toy` / `noisy`, sans GPU,
sans dataset). Ce guide décrit le **mode réel** (vrais modèles + vraies données), à
exécuter sur un **PC équipé d'un GPU NVIDIA**. Rien de lourd ne tourne en CI ni sur
Mac : `torch`, `transformers`, `unsloth`, `peft`, `timm` sont **importés
paresseusement**, uniquement quand un backend réel est appelé.

## 1. Pré-requis

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, transformers, accelerate, peft, datasets...
pip install unsloth                       # pour le LoRA Gemma (P1)
pip install timm                          # pour le classifieur léger
huggingface-cli login                     # accès MedGemma (licence) / Gemma
```

Accès aux ressources (à documenter dans le rapport : source, version, licence,
restrictions) :

- **MedGemma 4B** — `google/medgemma-4b-it` (licence Hugging Face).
- **Gemma 4** — via Unsloth (`unsloth/gemma-3-4b-it`).
- **RSNA Pneumonia** — Kaggle (compte + acceptation des règles). Non redistribuable :
  ne **jamais** committer d'images réelles dans le dépôt.

## 2. Variables d'environnement

| Variable | Rôle | Exemple |
|---|---|---|
| `RADIO_BACKEND` | backend d'inférence | `vlm`, `classifier`, `noisy`, `toy` |
| `RADIO_VLM_MODEL` | modèle VLM | `google/medgemma-4b-it` |
| `RADIO_PROMPT_MODE` | prompt utilisé par l'API | `improved`, `structured` |
| `RADIO_CLASSIFIER_CKPT` | poids du classifieur | `finetuning/outputs/light_classifier.pt` |
| `RADIO_DB_PATH` | base SQLite de logs | `/data/runs.sqlite` |

## 3. Inférence réelle (P0 — baseline prompting)

```bash
# API avec le vrai VLM médical
RADIO_BACKEND=vlm RADIO_VLM_MODEL=google/medgemma-4b-it \
  uvicorn api.main:app --host 0.0.0.0 --port 8000

# Évaluation sur les cas finaux avec le vrai modèle
RADIO_BACKEND=vlm python eval/run_evaluation.py --mode improved --backend vlm \
  --split final --out-dir /tmp/eval --db-path /tmp/runs.sqlite
```

En Python : `from src.inference import predict; predict(path, backend="vlm", mode="improved")`.

## 4. Classifieur léger de support (CNN/ViT)

```bash
python finetuning/train_light_classifier.py --csv data/rsna_cases.csv --split dev \
    --backbone resnet18 --epochs 10 --out finetuning/outputs/light_classifier.pt
RADIO_BACKEND=classifier RADIO_CLASSIFIER_CKPT=finetuning/outputs/light_classifier.pt \
  python eval/run_evaluation.py --backend classifier --split final \
  --out-dir /tmp/eval --db-path /tmp/runs.sqlite
```

## 5. Fine-tuning (P1 LoRA → P3 QLoRA), seulement après baseline validée

```bash
# 1) Préparer le dataset d'instructions (tourne aussi sur Mac)
python finetuning/prepare_dataset.py --csv data/rsna_cases.csv --split dev \
    --out finetuning/data/train.jsonl

# 2a) Gemma 4 + Unsloth (LoRA multimodal) — image avant texte, vision layers gelées
python finetuning/gemma4_unsloth_lora.py --dataset finetuning/data/train.jsonl

# 2b) MedGemma 4B en PEFT/QLoRA 4-bit
python finetuning/medgemma_peft_qlora.py --dataset finetuning/data/train.jsonl

# 3) Ré-évaluer AVANT toute décision (JSON, hallucinations, sensibilité, spécificité)
python eval/run_evaluation.py --backend vlm --split final \
    --out-dir /tmp/eval --db-path /tmp/runs.sqlite
```

Guide matériel : GPU modeste → prompting ; GPU moyen → LoRA E2B/E4B ; A100 → grands
modèles. **LoRA d'abord, pas de full fine-tuning** (≈ 4× plus de VRAM).

## 6. Données réelles : préparer un CSV compatible

Réutiliser les colonnes de `data/synthetic_cases.csv`
(`case_id, image_path, source, label, split, quality, notes`) en pointant vers des
images **dé-identifiées** (le prétraitement applique aussi `anonymize_image` /
suppression des balises PHI DICOM, voir `src/preprocessing.py`). Mapper les labels
RSNA vers `normal` / `suspected_opacity` / `uncertain`.

## 7. Garde-fous inchangés

Quel que soit le backend, la sortie passe par `src/guardrails.py` : validation du
schéma, règle d'incertitude (§7.2), avertissement non clinique forcé, journalisation
SQLite. Le contrat de sortie reste identique en synthétique et en réel.
