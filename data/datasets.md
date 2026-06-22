# Datasets réels — sources, accès, licences, mapping

> Le brief impose, pour tout dataset mobilisé, d'indiquer : **source, version,
> licence / conditions d'accès, restrictions de redistribution, anonymisation,
> limites d'interprétation**. Ce fichier centralise ces informations.

Aucune image patient réelle n'est commitée dans ce dépôt. Les builders
convertissent les données **localement** (sur la machine GPU) vers `data/rsna/`,
qui est ignoré par git.

---

## 1. RSNA Pneumonia Detection Challenge — dataset principal

| Champ | Valeur |
|---|---|
| Source | Kaggle — RSNA Pneumonia Detection Challenge |
| Version | `stage_2` (labels + images DICOM) |
| Volume | ≈ 30 000 radiographies thoraciques frontales |
| Licence / accès | Compte Kaggle + acceptation des règles du challenge. Usage recherche/éducation. **Non redistribuable.** |
| Anonymisation | Images déjà dé-identifiées par l'organisateur |
| Limites | Labels binaires (opacité / pas d'opacité), centrés pneumonie ; pas de classe `uncertain` ; population et matériel spécifiques |

### Téléchargement

```bash
pip install kaggle                     # voir requirements-gpu.txt
# Déposer le token Kaggle dans ~/.kaggle/kaggle.json (chmod 600)
kaggle competitions download -c rsna-pneumonia-detection-challenge -p ~/datasets/rsna
unzip ~/datasets/rsna/rsna-pneumonia-detection-challenge.zip -d ~/datasets/rsna
```

### Construction des splits (PNG + cases.csv)

```bash
python -m src.datasets build-rsna --src ~/datasets/rsna --out data/rsna \
  --n-smoke 20 --n-dev 150 --n-final 30 --seed 13
```

### Mapping des labels

| RSNA `Target` | Classe projet |
|---|---|
| 1 | `suspected_opacity` |
| 0 | `normal` |
| — | `uncertain` (jamais un label : produit par le seuil de confiance + garde-fous) |

---

## 2. CheXpert — extension / discussion scientifique

| Champ | Valeur |
|---|---|
| Source | Stanford AIMI — CheXpert |
| Volume | 224 316 radiographies + rapports |
| Licence / accès | Accord d'utilisation Stanford (Research Use Agreement). **Non redistribuable.** |
| Mapping | `chexpert_label()` dans `src/datasets.py` : opacité/consolidation/pneumonie → `suspected_opacity` ; `-1.0` → `uncertain` ; `No Finding` → `normal` |
| Limites | Labels extraits automatiquement des rapports (bruit d'étiquetage), politique d'incertitude à documenter |

## 3. MIMIC-CXR / MIMIC-CXR-JPG — avancé

| Champ | Valeur |
|---|---|
| Source | PhysioNet — MIMIC-CXR (2.1.0) |
| Volume | 377 110 images, rapports associés |
| Licence / accès | **Accès contrôlé** : compte PhysioNet + formation CITI + signature DUA. **Non redistribuable.** |
| Limites | Richesse clinique élevée mais conditions d'accès strictes ; à réserver à la discussion |

## 4. NIH ChestX-ray14 — contexte

| Champ | Valeur |
|---|---|
| Source | NIH Clinical Center |
| Volume | ≈ 112 000 images, 14 labels larges |
| Licence / accès | Domaine public (NIH), conditions d'usage à citer |
| Limites | Labels larges et bruités (NLP sur rapports), utiles surtout comme contexte |

---

## Décision tutorale (rappel du brief)

- **RSNA** pour le projet principal (baseline + classifieur + éval).
- **CheXpert / MIMIC** pour la discussion scientifique uniquement.
- Traçabilité via `cases.csv` + base **SQLite** (`sql/schema.sql`).
- Le jeu **synthétique** (`data/synthetic_cases.csv`) reste le smoke test logiciel.
