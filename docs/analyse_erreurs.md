# Registre d'analyse d'erreurs (livrable L9)

> **Cadre.** 30 cas finaux, backend `noisy` (synthétique, graine 203), sorties du
> **système livré** (après garde-fous). Registre complet et reproductible :
> `eval/error_register_final.csv` (60 lignes = 30 cas × {baseline, improved}).
> **100 % des cas sont commentés et catégorisés.**

Reproduction :

```bash
python eval/build_error_register.py
```

## 1. Taxonomie des 4 types d'erreur (§9.1)

| Code | Type | Définition | Action corrective documentée |
|---|---|---|---|
| **FN** | Faux négatif | Anomalie réelle non détectée | Enrichir les données, revoir le seuil, renforcer la sensibilité |
| **FP** | Faux positif | Anomalie prédite à tort | Réduire les faux signaux, ajuster le seuil, améliorer la spécificité |
| **UA** | Incertitude acceptable | Doute signalé sans conclure | Calibrer l'incertitude, affiner les seuils, enrichir le contexte |
| **HT** | Hallucination textuelle | Information inventée | Contraindre la génération, prompt « pas d'invention », garde-fous |

(`OK` = cas correct, non comptabilisé comme erreur.)

## 2. Répartition sur les 30 cas finaux

| Modèle | OK | FN | FP | UA | HT |
|---|---|---|---|---|---|
| baseline | 16 | 7 | 4 | 1 | 2 |
| improved | 26 | 1 | 0 | 3 | 0 |

Lecture : le modèle amélioré **divise par ~5 les faux négatifs** (7 → 1), **élimine
les faux positifs** (4 → 0) et **les hallucinations** (2 → 0). Les erreurs résiduelles
sont surtout des **incertitudes acceptables** (UA, 3) : le système préfère assumer le
doute plutôt que se tromper — comportement recherché en sécurité clinique.

## 3. Top 5 des causes de panne

| Rang | Cause | Occurrences | Type dominant |
|---|---|---|---|
| 1 | Opacité synthétique peu marquée | 6 | FN |
| 2 | Qualité limitée / signes non concluants | 4 | UA |
| 3 | Texte inventé (mention non présente) | 2 | HT |
| 4 | Structure normale sur-interprétée | 2 | FP |
| 5 | Artefact pris pour une opacité | 2 | FP |

## 4. Une action corrective par type (§9.4)

- **FN (cause #1)** — opacités peu marquées manquées : augmenter le rappel de la
  classe `suspected_opacity` (seuil de décision, augmentation de données d'opacités
  subtiles, pondération de la perte) ; **priorité sécurité**.
- **FP (causes #4, #5)** — structures/artefacts sur-interprétés : renforcer la
  spécificité (prétraitement, exemples négatifs difficiles, calibration du seuil).
- **UA (cause #2)** — qualité limitée : c'est le garde-fou attendu ; calibrer le seuil
  d'incertitude pour ne pas sur-déclencher, enrichir le contrôle qualité image.
- **HT (cause #3)** — texte inventé : prompt « pas d'invention » (prompt `structured`)
  + validation/garde-fous ; mesuré à **0 %** sur le modèle amélioré.

## 5. Grille de relecture d'un cas (§9.3)

1. **Qualité image** : exposition, centrage, inspiration.
2. **Zone suspecte** : localisation et caractérisation.
3. **Justification textuelle** : argumentation claire, factuelle, sans invention.
4. **Cohérence du warning** : aligné avec le niveau de risque (règle §7.2).

## 6. Limite

Analyse menée sur un **jeu synthétique** : elle valide la méthode (catégorisation
exhaustive, causes, actions correctives), pas une performance médicale. Sur données
réelles autorisées, refaire l'analyse avec relecture humaine.
