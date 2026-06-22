# Guide d'exécution sur le PC GPU

> **Author :** projet étudiant — Assistant radiologue virtuel
> **Année académique :** 2025-2026

Ce guide enchaîne, dans l'ordre, toutes les commandes pour passer du prototype
jouet (CPU) au prototype réel (MedGemma / Gemma 4 + RSNA) sur une machine
équipée d'un GPU NVIDIA. Le Mac sert au développement et au smoke test ; le PC
GPU sert à la baseline réelle, au classifieur et au fine-tuning.

## 0. Environnement

```bash
git clone <repo> && cd ARDX_RX_mastercamp
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
# 1) torch CUDA d'abord (adapter à votre version CUDA) : https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install -r requirements-gpu.txt
```

Vérification GPU :

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1. Accès aux modèles et aux données

```bash
# MedGemma / Gemma : accepter la licence sur Hugging Face puis se connecter
huggingface-cli login

# RSNA Pneumonia (Kaggle) : token dans ~/.kaggle/kaggle.json (chmod 600)
kaggle competitions download -c rsna-pneumonia-detection-challenge -p ~/datasets/rsna
unzip ~/datasets/rsna/rsna-pneumonia-detection-challenge.zip -d ~/datasets/rsna
```

## 2. Construire les splits réels (PNG + cases.csv)

```bash
python -m src.datasets build-rsna --src ~/datasets/rsna --out data/rsna \
  --n-smoke 20 --n-dev 150 --n-final 30 --seed 13
```

Produit `data/rsna/cases.csv` et `data/rsna/images/<split>/<classe>/`.

## 3. Baseline VLM réelle : baseline vs amélioré (MedGemma 4B)

```bash
export RADIO_VLM_MODEL=google/medgemma-4b-it
python eval/run_evaluation.py --backend vlm --mode toy \
  --cases data/rsna/cases.csv --split final \
  --out-dir eval/outputs --db-path runs.sqlite
```

`--mode toy` lance successivement le prompt **baseline** et le prompt
**amélioré** et écrit `before_after_summary.csv` (le delta attendu par le brief).

## 4. Classifieur léger (support de confiance)

```bash
python -m src.classifier train --data-dir data/rsna/images/dev \
  --backbone resnet18 --epochs 5 --out models/classifier.pt

export RADIO_CLASSIFIER_CKPT=models/classifier.pt
python eval/run_evaluation.py --backend classifier --mode baseline \
  --cases data/rsna/cases.csv --split final --out-dir eval/outputs --db-path runs.sqlite
```

## 5. Fine-tuning (COULD) — uniquement après une baseline validée

```bash
# Voie rapide : Gemma 4 multimodal LoRA (Unsloth)
python finetuning/gemma4_unsloth_lora.py --cases data/rsna/cases.csv --split dev \
  --model unsloth/gemma-3-4b-it --out models/gemma4_lora --epochs 1

# Option médicale avancée : MedGemma 4B QLoRA (PEFT)
python finetuning/medgemma_peft_qlora.py --cases data/rsna/cases.csv --split dev \
  --model google/medgemma-4b-it --out models/medgemma_qlora --epochs 1
```

Réévaluer ensuite sur le split `final` et comparer au point 3 **avant** de
conclure à un gain.

## 6. Démo web réelle

```bash
export RADIO_BACKEND=vlm          # ou classifier
streamlit run app/streamlit_app.py
# API : uvicorn api.main:app --reload   (lit aussi RADIO_BACKEND)
```

## 7. Mettre à jour le rapport

Reporter les nombres de `eval/outputs/*_metrics.json` et
`before_after_summary.csv` dans le tableau de [docs/rapport.md](rapport.md),
puis exporter le registre d'erreurs (voir le rapport, section analyse d'erreurs).

## Aide-mémoire matériel (rappel du brief)

| Ressource | Voie conseillée |
|---|---|
| GPU modeste | prompting baseline |
| GPU moyen | LoRA Gemma 4 E2B/E4B |
| A100 | grands modèles, FFT |

> Garde-fous toujours actifs : warning obligatoire, classe `uncertain`,
> validation JSON, journalisation SQLite. Le fine-tuning ne les remplace pas.
