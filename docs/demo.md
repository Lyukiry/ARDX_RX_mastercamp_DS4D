# Démonstration (livrable L10)

Chaîne complète : **upload → prétraitement/anonymisation → modèle → garde-fous →
JSON → interface → logs SQLite**.

## 1. Lancer la démo (mode synthétique, sans GPU)

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Quatre onglets (cahier des charges §2.1) :

- **Cas** — catalogue des cas synthétiques (20 smoke + 30 final), image + vérité terrain.
- **Analyse** — choisir backend (`toy`/`noisy`) + prompt, lancer une prédiction,
  voir les 7 champs, l'avertissement et l'escalade d'incertitude (§7.2), le JSON brut.
- **Apprentissage** — les 3 classes, la règle d'avertissement, le tableau de
  comparaison des 3 prompts (calculé en direct).
- **Suivi** — journal SQLite des inférences + métriques baseline vs amélioré.

## 2. API `/predict`

```bash
uvicorn api.main:app --reload
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/sample_images/CXR_SYN_002_suspected_opacity.png"
```

Réponse : classe, confiance, observations, justification, limites, avertissement,
`uncertainty_warning`, plus `logged: true` (traçabilité SQLite à 100 %).

## 3. Script de soutenance suggéré

1. **Cadrage** (`docs/note_de_cadrage.md`) : périmètre gelé, 3 classes, position non
   clinique, avertissement sur 100 % des sorties.
2. **Un cas normal** (onglet Analyse) → classe `normal`, confiance, pas d'escalade.
3. **Un cas suspicion** → `suspected_opacity`, justification factuelle.
4. **Un cas incertain / mauvaise qualité** → `uncertain` + `uncertainty_warning`
   (montrer que le doute est assumé, pas masqué).
5. **Un cas d'erreur** (onglet Suivi / registre) → montrer un FN, un FP, une UA, une
   HT (ne **jamais** ne montrer que des réussites).
6. **Mesures** (`docs/rapport_mesures.md`) : matrice de confusion, tableau Δ
   baseline→amélioration, apport des garde-fous, 6 métriques vs objectifs.
7. **Analyse d'erreurs** (`docs/analyse_erreurs.md`) : 4 types, Top 5 causes, actions.
8. **Limites** (`docs/ethique_et_limites.md`) : jeu synthétique, pas de valeur
   médicale, validation humaine indispensable.

## 4. Mode réel (GPU)

Pour démontrer avec MedGemma/Gemma : `RADIO_BACKEND=vlm streamlit run
app/streamlit_app.py` (sélectionner le backend `vlm` dans l'onglet Analyse). Voir
`docs/guide_execution_gpu.md`.

## 5. Avant la soutenance — smoke test

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python eval/run_evaluation.py --mode toy \
  --out-dir /tmp/assistant-radio-eval --db-path /tmp/assistant-radio-evidence.sqlite
```
