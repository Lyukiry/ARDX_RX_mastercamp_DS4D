# Rapport de mesures (livrable L8)

> **Cadre.** Mesures sur les **30 cas finaux** (`split = final`), backend `noisy`
> (modèle synthétique déterministe, graine 203). **Ce ne sont pas des performances
> médicales** : le jeu est synthétique et sert à démontrer la *méthode* d'évaluation.
> Sur PC GPU, relancer avec `--backend vlm` (MedGemma/Gemma) — voir
> `docs/guide_execution_gpu.md`.

Reproduction :

```bash
python eval/run_evaluation.py --mode toy --backend noisy --split final \
    --out-dir /tmp/eval --db-path /tmp/runs.sqlite      # système livré (avec garde-fous)
python eval/compare_prompts.py --split final            # métriques brutes (sans garde-fous)
```

## 1. Deux niveaux de mesure

- **Modèle / prompt seul (brut)** : sortie directe du modèle, avant garde-fous.
  C'est la comparaison « baseline vs amélioration » du cahier des charges (§8.2).
- **Système livré (+ garde-fous)** : après `src/guardrails.py` (validation, règle
  d'incertitude, avertissement forcé). C'est ce que voit l'utilisateur de l'API/UI.

## 2. Matrices de confusion (système livré, 30 cas)

**baseline** (lignes = vérité, colonnes = prédiction) :

| vérité \ prédit | normal | suspected_opacity | uncertain |
|---|---|---|---|
| normal | 7 | 2 | 1 |
| suspected_opacity | 4 | 6 | 0 |
| uncertain | 4 | 2 | 4 |

**improved** :

| vérité \ prédit | normal | suspected_opacity | uncertain |
|---|---|---|---|
| normal | 8 | 0 | 2 |
| suspected_opacity | 1 | 8 | 1 |
| uncertain | 0 | 0 | 10 |

Calcul explicite (improved, classe `suspected_opacity` comme « positif ») :

- Sensibilité = VP / (VP + FN) = 8 / (8 + 1 + 1) = **0.80**
- Spécificité (rappel `normal`) = VN / (VN + FP) = 8 / (8 + 0 + 2) = **0.80**

## 3. Tableau Baseline vs Amélioration — Δ (modèle/prompt seul, brut)

| Métrique | Baseline | Amélioration | Δ (gain) | Réf. cahier des charges |
|---|---|---|---|---|
| Accuracy | 0.567 | 0.733 | **+0.166** | +0.15 |
| Macro-F1 | 0.564 | 0.731 | **+0.167** | +0.17 |
| Sensibilité | 0.60 | 0.80 | **+0.20** | +0.19 |
| Spécificité | 0.70 | 0.80 | **+0.10** | +0.13 |

Le gain est net, monotone et cohérent avec les ordres de grandeur de référence.

## 4. Apport des garde-fous (système livré)

| Modèle | Accuracy brute | Accuracy + garde-fous |
|---|---|---|
| baseline | 0.567 | 0.567 |
| improved | 0.733 | **0.867** |

Les garde-fous reroutent vers `uncertain` les prédictions **peu confiantes sur image
de mauvaise qualité** : sur le modèle amélioré (bien calibré), les 10 cas `uncertain`
sont alors tous correctement traités (cf. matrice). Sur la baseline (sur-confiante
même quand elle se trompe), les garde-fous se déclenchent peu — d'où l'absence de
gain : c'est un **symptôme de mauvaise calibration**, à corriger en priorité.

## 5. Six métriques vs objectifs (système livré, modèle amélioré)

| Métrique | Objectif | Obtenu (synthétique) | Statut |
|---|---|---|---|
| Accuracy | ≥ 0.70 | 0.867 | ✅ |
| Macro-F1 | ≥ 0.68 | 0.867 | ✅ |
| Sensibilité | élevée (prioritaire) | 0.80 | ✅ |
| Spécificité | équilibrée | 0.80 | ✅ |
| JSON valide | ≥ 95 % | 100 % (brut improved/structured ; 100 % en prod.) | ✅ |
| Latence | < 10 s | ≈ 1 ms (synthétique) ; cible VLM réel < 10 s | ✅ (à mesurer sur GPU) |

Indicateurs de prudence complémentaires : avertissement présent **100 %** (forcé par
garde-fous), sorties **journalisées 100 %** (SQLite), taux d'incertitude amélioré
≈ 43 % (le modèle assume le doute plutôt que de forcer une classe).

## 6. Logique de décision (GO / NO-GO)

- **Prioriser la sensibilité** (éviter les faux négatifs = anomalie manquée).
- **Maintenir une spécificité équilibrée** (limiter les sur-alertes).
- **GO** seulement si métriques **et** prudence clinique sont cohérentes.

Décision sur ce jeu **synthétique** : la *méthode* est validée (gain mesuré,
garde-fous efficaces, prudence assumée). **NO-GO clinique** : aucune valeur médicale ;
toute décision réelle exige une évaluation sur données réelles autorisées et une
validation humaine indépendante.
