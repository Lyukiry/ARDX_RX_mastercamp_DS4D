# Architecture cible
> **Author :** Badr TAJINI
> **Solution Delivery - filière Data**
>  **Année académique :** 2025-2026

## Pipeline en 7 étapes (cahier des charges §5.1)

```text
1. Upload CXR        → chargement de l'image (UI / API)
2. Prétraitement     → normalisation + redimensionnement + ANONYMISATION (EXIF/PHI DICOM)
3. Modèle            → backend toy | noisy | vlm (MedGemma/Gemma) | classifier (CNN/ViT)
4. Garde-fous        → validation JSON, règle d'incertitude §7.2, avertissement forcé
5. JSON structuré    → contrat à 7 champs (+ uncertainty_warning)
6. Interface web     → Streamlit (onglets Cas / Analyse / Apprentissage / Suivi) ou Gradio
7. SQLite / logs     → résultats, logs d'inférence, métadonnées (100 % journalisé)
```

## Composants

- `src/preprocessing.py` : chargement, **anonymisation** (`anonymize_image`,
  suppression des balises PHI DICOM), **normalisation** (`normalize_image`), pipeline
  `preprocess`.
- `src/inference.py` : aiguilleur `predict(backend=...)` + prédicteur jouet `toy_predict`.
- `src/synthetic_eval.py` : backend `noisy` (modèle synthétique imparfait, déterministe).
- `src/vlm_inference.py` : backend `vlm` réel (MedGemma/Gemma), imports paresseux.
- `src/classifier.py` : backend `classifier` léger (timm CNN/ViT), imports paresseux.
- `src/guardrails.py` : validation du schéma, règle d'incertitude (§7.2), avertissement.
- `src/metrics.py` : accuracy, macro-F1, sensibilité, spécificité, matrice de confusion.
- `src/database.py` : initialisation SQLite et stockage des runs.
- `api/main.py` : endpoint FastAPI `/predict` + journalisation systématique.
- `app/streamlit_app.py`, `app/gradio_app.py` : interfaces de démonstration.

## Backends d'inférence

| Backend | Exécution | Usage |
|---|---|---|
| `toy` | partout (défaut) | valide la chaîne ; lit le label dans le nom de fichier (parfait) |
| `noisy` | partout | modèle synthétique déterministe ; métriques/erreurs réalistes |
| `vlm` | PC GPU | vrai VLM médical MedGemma/Gemma (P0/P3) |
| `classifier` | GPU/CPU | classifieur léger de support (score de confiance) |

Sélection via l'argument `backend=`, la variable `RADIO_BACKEND`, ou `--backend`.

## Endpoint attendu

```http
POST /predict
Content-Type: multipart/form-data
```

Réponse (extrait) :

```json
{
  "image_quality": "good | limited | poor",
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": [],
  "justification": "...",
  "limitations": [],
  "warning": "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.",
  "uncertainty_warning": "(escalade si confidence < 0.60 ou qualité = mauvaise)",
  "model_name": "toy-rule-improved",
  "prompt_version": "improved_v1",
  "latency_ms": 0,
  "logged": true
}
```

## Objectifs d'intégration

- `>= 95 %` JSON valide (100 % en production via garde-fous).
- 100 % des sorties avec avertissement.
- 100 % des runs journalisés (SQLite).
- Latence cible `< 10 s` (mode VLM réel ; ≈ ms en synthétique).
