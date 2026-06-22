# Rapport final — Assistant radiologue virtuel responsable

> Prototype pédagogique d'IA médicale multimodale — EFREI, Solution Delivery / Data, 2025-2026
>
> **Position non clinique.** Ce système n'est pas un dispositif médical. Aucune
> sortie ne doit servir à diagnostiquer, trier ou orienter un patient.

Ce rapport suit les critères d'évaluation du brief (`docs/appel_offre.md`) :
périmètre + dataset, baseline, amélioration mesurée, intégration, évaluation +
erreurs, éthique + limites.

---

## 1. Périmètre et dataset (15 %)

- **Entrée :** une radiographie thoracique frontale (PNG/JPG ; DICOM converti par `src/datasets.py`).
- **Sortie :** un JSON unique contenant `image_quality`, `predicted_class`
  (`normal` | `suspected_opacity` | `uncertain`), `confidence`, `visual_evidence`,
  `justification`, `limitations`, `warning` (+ `model_name`, `prompt_version`, `latency_ms`).
- **Dataset principal :** RSNA Pneumonia Detection Challenge (Kaggle). Mapping
  `Target=1 → suspected_opacity`, `Target=0 → normal`. La classe `uncertain`
  n'est jamais un label : elle est produite par le seuil de confiance et les
  garde-fous. Détails, licences et accès : [data/datasets.md](../data/datasets.md).
- **Splits :** `smoke` (20, vérif pipeline), `dev` (≈150, développement),
  `final` (30, cas commentés pour la soutenance), reproductibles (`seed=13`).
- **Jeu synthétique :** `data/synthetic_cases.csv` (30 images) — sert
  exclusivement de **smoke test logiciel**, jamais de preuve de performance médicale.

## 2. Modèles (rappel du brief, section « Quels modèles utiliser ? »)

| Rôle | Modèle | Où |
|---|---|---|
| Baseline recommandée | **MedGemma 4B** (`google/medgemma-4b-it`) | `src/vlm_inference.py` |
| Voie rapide / fine-tuning | **Gemma 4 E2B/E4B + Unsloth** | `finetuning/gemma4_unsloth_lora.py` |
| Support de confiance | **CNN/ViT léger** (resnet18 par défaut) | `src/classifier.py` |
| Référence recherche | MAIRA-2 | non exécuté (cité pour contexte) |
| Repli sans GPU | prédicteur jouet déterministe | `src/inference.py` (`toy`) |

Le backend se choisit par `--backend` (éval), par `RADIO_BACKEND` (API/UI) ou par
l'argument `backend=` de `src.inference.predict`.

## 3. Baseline reproductible (15 %)

- Baseline = MedGemma 4B en **prompting**, prompt `prompts/baseline_prompt.txt`.
- Notebook : `notebooks/01_baseline_vlm.ipynb`.
- Exécution reproductible (PC GPU) : voir [docs/guide_execution_gpu.md](guide_execution_gpu.md), section 3.
- Sortie contrainte par le **contrat JSON** + parsing robuste (`extract_json`,
  `coerce_to_schema`) : toute sortie non parsable retombe sur `uncertain`.

## 4. Amélioration mesurée (20 %)

Trois leviers d'amélioration légère, sans réentraînement :

1. **Prompt renforcé** (`prompts/improved_prompt.txt`) : vérification explicite
   des artefacts (projection, rotation, exposition), style factuel.
2. **Règle d'incertitude** : si `confidence < 0.60` ou qualité `poor`,
   `predicted_class = uncertain` (`apply_uncertainty_rule`).
3. **Garde-fous** (`src/guardrails.py`) : validation du schéma, warning forcé,
   repli `uncertain` sur sortie invalide.

Comparaison baseline vs amélioré produite automatiquement par
`eval/run_evaluation.py --mode toy` (deux passes) → `before_after_summary.csv`.

### Résultats — pipeline (jeu synthétique, backend `toy`)

> Démontre que la chaîne tourne de bout en bout. **Non médical** (le prédicteur
> jouet lit le label dans le nom de fichier). Régénérer avec :
> `python eval/run_evaluation.py --mode toy --out-dir /tmp/eval --db-path /tmp/runs.sqlite`

<!-- TOY_RESULTS_START -->
| Métrique | Baseline (toy) | Amélioré (toy) |
|---|---|---|
| n | 30 | 30 |
| Accuracy | 1.00 | 1.00 |
| Macro-F1 | 1.00 | 1.00 |
| Sensibilité | 1.00 | 1.00 |
| Spécificité | 1.00 | 1.00 |
| JSON valide | 100 % | 100 % |
| Warning présent | 100 % | 100 % |
| Taux d'incertitude | 33 % | 33 % |
| Latence médiane | ~0 ms | ~0 ms |

Lecture : le score parfait est **attendu et non significatif** — le prédicteur
jouet lit le label dans le nom de fichier. Cette table prouve seulement que la
chaîne (inférence → garde-fous → JSON valide → warning → métriques → SQLite)
fonctionne et que le contrat est respecté à 100 %. Le taux d'incertitude de 33 %
correspond aux 10 cas `uncertain` du jeu synthétique (sur 30).
<!-- TOY_RESULTS_END -->

### Résultats — réels (RSNA `final`, backend `vlm` MedGemma 4B)

> À compléter après l'exécution GPU (guide, section 3). Objectifs indicatifs du
> brief : Accuracy ≥ 0.70, Macro-F1 ≥ 0.68, JSON valide ≥ 95 %, latence < 10 s,
> sensibilité élevée, spécificité équilibrée.

| Métrique | Baseline | Amélioré | Δ |
|---|---|---|---|
| Accuracy | _._ | _._ | _._ |
| Macro-F1 | _._ | _._ | _._ |
| Sensibilité (`suspected_opacity`) | _._ | _._ | _._ |
| Spécificité (`normal`) | _._ | _._ | _._ |
| JSON valide | _._ | _._ | _._ |
| Warning présent | 100 % | 100 % | — |
| Latence médiane (ms) | _._ | _._ | _._ |

Définitions : sensibilité = rappel sur `suspected_opacity` (protège des faux
négatifs) ; spécificité = part des `normal` non sur-alertés en `suspected_opacity`
(une prédiction `uncertain` n'est pas comptée comme fausse alerte). Voir
`src/metrics.py`.

## 5. Intégration applicative (15 %)

- **API** FastAPI `POST /predict` (`api/main.py`), warning systématique.
- **UI** Streamlit et Gradio (`app/`), sélecteur de mode et de backend.
- **Journalisation** SQLite (`sql/schema.sql`, `src/database.py`) : chaque run
  (modèle, prompt, JSON, classe, confiance, latence) est stocké.
- **Objectifs d'intégration :** ≥ 95 % JSON valide, 100 % warning, 100 % runs
  journalisés, latence cible < 10 s.

## 6. Évaluation et analyse d'erreurs (20 %)

Taxonomie (voir `docs/evaluation_protocol.md` et `classify_error` dans
`src/metrics.py`) : **FN**, **FP**, **UA** (incertitude acceptable),
**JF** (JSON invalide), **HT** (hallucination textuelle, repérée à la relecture).

Registre des 30 cas `final` : remplir
[eval/error_register_template.csv](../eval/error_register_template.csv) à partir
des prédictions (`eval/outputs/*_predictions.csv`) en relisant chaque cas selon la
grille : qualité image, zone suspecte, justification, cohérence du warning.

> Règle de soutenance : **ne jamais montrer uniquement des réussites** — exposer
> aussi FN, FP, incertitudes et limites de qualité image.

## 7. Éthique, limites et risques (10 %)

- Avertissement non clinique présent dans JSON, UI, API, README, rapport.
- Classe `uncertain` conservée comme garde-fou (jamais un échec).
- Aucune donnée patient réelle commitée ; datasets sous licence, non redistribués.
- **Limites documentées :** labels RSNA binaires et bruités ; supervision de
  fine-tuning *templatisée* (justifications non rédigées par un expert) ;
  confiance non calibrée formellement ; sensibilité au prompt et au modèle ;
  jeu `final` réduit (30 cas) → intervalles de confiance larges ;
  risque d'hallucination textuelle (atténué, pas éliminé).

## 8. Reproductibilité

```bash
# Smoke test (sans GPU)
pip install -r requirements-test.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python eval/run_evaluation.py --mode toy --out-dir /tmp/eval --db-path /tmp/runs.sqlite

# Chaîne réelle (PC GPU) : docs/guide_execution_gpu.md
```

## 9. Références

Sources, versions et accès : `docs/appel_offre.md` (R1–R7), `data/datasets.md`,
`README.md`. MedGemma, Gemma 4 / Unsloth, MIMIC-CXR, CheXpert restent soumis à
leurs licences propres.
