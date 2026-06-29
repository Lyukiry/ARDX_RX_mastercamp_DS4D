# Lancer les modèles sur le PC GPU

Guide clair, copier-coller, pour exécuter les **vrais modèles** (MedGemma 4B,
Gemma 4 + Unsloth, classifieur CNN/ViT) sur ta machine fixe équipée d'un **GPU
NVIDIA**.

> Le dépôt tourne **partout** sans GPU en mode synthétique (backends `toy` /
> `noisy`). Les dépendances lourdes (`torch`, `transformers`, `unsloth`, `peft`,
> `timm`) sont importées **paresseusement** : elles ne se chargent que quand un
> backend réel est appelé. Version détaillée : [docs/guide_execution_gpu.md](docs/guide_execution_gpu.md).
>
> ⚠️ **Usage pédagogique uniquement — non destiné au diagnostic.** Ne jamais
> committer d'images patient réelles.

---

## 1. Prérequis

- GPU NVIDIA + pilotes CUDA récents (vérifier `nvidia-smi`).
- Python 3.10–3.11.
- Un compte **Hugging Face** avec licence MedGemma acceptée (voir §1.b).
- (Optionnel, données réelles) un compte **Kaggle** pour RSNA Pneumonia.

## 1.b Récupérer MedGemma (« Gemma med ») depuis Hugging Face

MedGemma est un modèle **à accès restreint (gated)** : il faut **accepter la licence
sur le site** une fois, puis s'authentifier avec un **token**. Le téléchargement des
poids est ensuite automatique au premier appel du backend `vlm`.

**Étape 1 — Accepter la licence (sur le web, une seule fois)**
1. Se connecter sur <https://huggingface.co>.
2. Ouvrir la page du modèle : <https://huggingface.co/google/medgemma-4b-it>.
3. Cliquer **« Agree and access repository »** et accepter les conditions
   (*Health AI Developer Foundations* de Google). La page doit ensuite afficher
   « You have been granted access to this model ».

> Variantes : `google/medgemma-4b-it` (multimodal, *instruction-tuned* — celui
> utilisé ici), `google/medgemma-4b-pt` (pré-entraîné), `google/medgemma-27b-text-it`
> (texte uniquement, beaucoup plus lourd). Garder **`medgemma-4b-it`** pour ce projet.

**Étape 2 — Créer un token d'accès**
1. <https://huggingface.co/settings/tokens> → **« Create new token »**.
2. Type **Read** (suffisant pour un modèle gated). Copier le token (`hf_...`).

**Étape 3 — S'authentifier**
```bash
pip install -U huggingface_hub
hf auth login                # coller le token  (CLI moderne = `hf`)
# huggingface_hub < 1.0 utilisait :  huggingface-cli login
# alternative sans interaction :     export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

**Étape 4 — Vérifier l'accès** (doit afficher ton identifiant, puis télécharger le modèle)
```bash
hf auth whoami
python -c "from huggingface_hub import model_info; print(model_info('google/medgemma-4b-it').id)"
```

En cas de `401/403 Gated`/`Access denied` : la licence n'est pas acceptée (étape 1)
ou le token n'est pas chargé (étape 3). Le même token donne accès à **Gemma**
(`unsloth/gemma-3-4b-it`) après acceptation de sa licence respective.

## 2. Installation

```bash
git clone <repo> && cd ARDX_RX_mastercamp
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate

# 1) torch en build CUDA D'ABORD (adapter à ta version CUDA) :
#    https://pytorch.org/get-started/locally/
# 2) le reste des dépendances :
pip install -r requirements.txt

# 3) extras GPU pour le fine-tuning (non inclus dans requirements.txt) :
pip install unsloth bitsandbytes

# 4) connexion Hugging Face (accès MedGemma / Gemma) — détails en §1.b :
hf auth login
```

Vérifier le GPU :

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Les 4 backends et les variables d'environnement

Le **même contrat JSON** (7 champs) est produit par tous les backends. On choisit
le backend par la variable `RADIO_BACKEND` (ou l'argument `backend=` en Python).

| Backend | Modèle | GPU | Usage |
|---|---|---|---|
| `toy` | règle déterministe | non | smoke test / démo hors-ligne (défaut) |
| `noisy` | synthétique imparfait reproductible | non | métriques & analyse d'erreurs réalistes |
| `vlm` | **MedGemma 4B / Gemma** (Hugging Face) | oui | vraie baseline par prompting |
| `classifier` | **CNN/ViT léger** (resnet18…) | oui/cpu | support de confiance / calibration |

| Variable | Rôle | Défaut |
|---|---|---|
| `RADIO_BACKEND` | backend d'inférence | `toy` |
| `RADIO_VLM_MODEL` | modèle VLM Hugging Face | `google/medgemma-4b-it` |
| `RADIO_PROMPT_MODE` | prompt utilisé (`baseline`/`improved`/`structured`) | `improved` |
| `RADIO_CLASSIFIER_CKPT` | chemin du checkpoint classifieur | (vide) |
| `RADIO_CLASSIFIER_BACKBONE` | architecture du classifieur | `resnet18` |
| `RADIO_DB_PATH` | base SQLite des logs | dossier temp |

## 4. Inférence réelle avec MedGemma (P0 — baseline prompting)

**API web** (upload + JSON + warning + log SQLite) :

```bash
RADIO_BACKEND=vlm RADIO_VLM_MODEL=google/medgemma-4b-it \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
# test :
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/sample_images/CXR_SYN_002_suspected_opacity.png"
```

**Interface Streamlit** (tout se pilote depuis l'UI) :

```bash
streamlit run app/streamlit_app.py     # le backend se choisit dans l'onglet « Analyse »
```

Dans l'onglet **Analyse**, on choisit le **backend** (`toy`/`noisy`/`vlm`/`classifier`),
le **mode de prompt**, puis la **source de l'image** :
- **Cas de test (catalogue)** — les images jouet de `data/sample_images` ;
- **Dataset RSNA / externe** — un **dossier** d'images RSNA (`.dcm`/`.png`/`.jpg`,
  DICOM anonymisé automatiquement) ou un **CSV** `data/rsna_cases.csv` ;
- **Téléverser** — n'importe quelle image (DICOM accepté).

Astuce : `RADIO_BACKEND=vlm streamlit run ...` ne fait que pré-sélectionner le backend.

**En Python** (DICOM accepté : anonymisation PHI automatique) :

```python
from src.inference import predict
from src.guardrails import apply_safety_guardrails

out = apply_safety_guardrails(predict("chemin/vers/radio.dcm", backend="vlm", mode="improved"))
print(out["predicted_class"], out["confidence"])
```

> Changer de modèle : `RADIO_VLM_MODEL=unsloth/gemma-3-4b-it` (Gemma 4),
> ou le chemin local d'un modèle fusionné après fine-tuning.

## 5. Classifieur léger (support de confiance)

```bash
# Entraînement (sur le jeu fourni, ou ton CSV réel — voir §7)
python finetuning/train_light_classifier.py --csv data/synthetic_cases.csv \
  --split all --backbone resnet18 --epochs 10 \
  --out finetuning/outputs/light_classifier.pt

# Utilisation comme backend
RADIO_BACKEND=classifier RADIO_CLASSIFIER_CKPT=finetuning/outputs/light_classifier.pt \
  uvicorn api.main:app --port 8000
```

## 6. Fine-tuning (COULD) — seulement après une baseline validée

```bash
# 1) Construire le dataset d'instructions (image + prompt -> JSON cible). Tourne sans GPU.
python finetuning/prepare_dataset.py --csv data/synthetic_cases.csv \
  --split final --out finetuning/data/train.jsonl

# 2a) Voie rapide : Gemma 4 multimodal LoRA (Unsloth) — image avant texte, vision gelée
python finetuning/gemma4_unsloth_lora.py --dataset finetuning/data/train.jsonl \
  --model unsloth/gemma-3-4b-it --epochs 2 --output finetuning/outputs/gemma_lora

# 2b) Option médicale : MedGemma 4B en PEFT/QLoRA 4-bit
python finetuning/medgemma_peft_qlora.py --dataset finetuning/data/train.jsonl \
  --model google/medgemma-4b-it --epochs 2 --output finetuning/outputs/medgemma_qlora

# 3) RÉÉVALUER avant toute conclusion (JSON valide, hallucinations, sensibilité, spécificité)
python eval/run_evaluation.py --backend vlm --split final \
  --out-dir /tmp/eval --db-path /tmp/runs.sqlite
```

Repères matériels : GPU modeste → prompting ; GPU moyen → LoRA E2B/E4B ; A100 →
grands modèles. **LoRA d'abord, pas de full fine-tuning** (≈ 4× plus de VRAM).

## 7. Données réelles (RSNA Pneumonia)

1. Télécharger via Kaggle (compte + acceptation des règles ; **non redistribuable**) :
   ```bash
   kaggle competitions download -c rsna-pneumonia-detection-challenge -p ~/datasets/rsna
   ```
2. Construire un CSV au **même format** que `data/synthetic_cases.csv` :
   colonnes `case_id, image_path, source, label, split, quality, notes`, en
   pointant vers des images **dé-identifiées** (`.png`, `.jpg` ou `.dcm`).
   Mapping conseillé : RSNA `Target=1 → suspected_opacity`, `Target=0 → normal`
   (la classe `uncertain` reste produite par le seuil de confiance, jamais un label).
3. Brancher ce CSV dans les scripts qui acceptent `--csv` :
   `finetuning/train_light_classifier.py` et `finetuning/prepare_dataset.py`.

> Les `.dcm` sont gérés par `src/preprocessing.py` : conversion + VOI LUT +
> **suppression des balises PHI** avant tout traitement.

## 8. Preuves attendues (évaluation, comparaison, erreurs)

```bash
# Métriques pipeline (jeu fourni) — accuracy, macro-F1, sensibilité, spécificité, JSON, latence
python eval/run_evaluation.py --backend noisy --mode toy --split final \
  --out-dir /tmp/eval --db-path /tmp/runs.sqlite

# Comparaison baseline vs amélioré (avant garde-fous)
python eval/compare_prompts.py --split final

# Registre d'erreurs des 30 cas finaux -> eval/error_register_final.csv
python eval/build_error_register.py
```

> ⚠️ Ne jamais écrire les sorties dans le dépôt : `eval/outputs/` et
> `medical_ai_evidence.sqlite` sont **interdits par la CI**. Utiliser `/tmp/...`.
> `eval/run_evaluation.py` évalue le jeu **embarqué** (`data/synthetic_cases.csv`) ;
> pour évaluer un vrai jeu RSNA, le préparer comme en §7.

## 9. Garde-fous (toujours actifs, quel que soit le backend)

Toute sortie passe par `src/guardrails.py` : validation du schéma JSON, règle
d'incertitude (confiance < 0.60 ou qualité `poor` → `uncertain`), **avertissement
non clinique forcé**, journalisation SQLite. Le fine-tuning ne les remplace pas.

## 10. Dépannage rapide

| Symptôme | Cause / solution |
|---|---|
| `torch.cuda.is_available() == False` | installer le build CUDA de torch (§2) |
| `401` / accès refusé au modèle | `huggingface-cli login` + accepter la licence MedGemma |
| `unsloth`/`bitsandbytes` introuvable | `pip install unsloth bitsandbytes` (GPU only) |
| smoke test échoue sur « forbidden paths » | `rm -rf eval/outputs medical_ai_evidence.sqlite` |
| OOM VRAM au fine-tuning | réduire `--batch-size`, garder `load_in_4bit`, LoRA (pas FFT) |
