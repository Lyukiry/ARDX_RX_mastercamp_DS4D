# Architecture cible
> **Author :** Badr TAJINI 
> **Solution Delivery - filière Data** 
>  **Année académique :** 2025-2026
## Pipeline

```text
Image upload → Preprocessing → Backend (toy | vlm | classifier) → Guardrails → JSON → UI → SQLite logs
```

Le classifieur léger fournit en plus un *support de confiance* (classe probable +
score), conformément au brief.

## Composants

- `src/preprocessing.py` : validation de fichier, chargement image, resizing.
- `src/inference.py` : dispatcher de backends (`predict`) + prédicteur jouet.
- `src/vlm_inference.py` : backend VLM réel (MedGemma 4B / Gemma), parsing JSON robuste.
- `src/classifier.py` : backend CNN/ViT léger (support de confiance / calibration).
- `src/datasets.py` : builders RSNA (DICOM→PNG, splits, `cases.csv`).
- `src/guardrails.py` : validation JSON, warning, incertitude.
- `src/metrics.py` : accuracy, macro-F1, sensibilité, spécificité, validité JSON, latence.
- `src/database.py` : initialisation SQLite et stockage des runs.
- `api/main.py` : endpoint FastAPI `/predict` (backend via `RADIO_BACKEND`).
- `app/streamlit_app.py` et `app/gradio_app.py` : interfaces rapides (sélecteur de backend).
- `finetuning/` : LoRA Gemma 4 (Unsloth) et QLoRA MedGemma (PEFT).

## Endpoint attendu

```http
POST /predict
Content-Type: multipart/form-data
```

Réponse :

```json
{
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": [],
  "justification": "...",
  "limitations": [],
  "warning": "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.",
  "model_name": "toy-rule-baseline",
  "prompt_version": "baseline_v1",
  "latency_ms": 0
}
```

## Objectifs d'intégration

- '>= 95 %' JSON valide.
- 100 % des sorties avec warning.
- 100 % des runs sauvegardés.
- Latence cible < 10 s en mode prototype.
