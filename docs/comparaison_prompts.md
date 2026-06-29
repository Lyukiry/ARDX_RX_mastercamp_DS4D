# Comparaison des prompts (livrable L3 / cahier des charges §7.3)

Trois prompts sont comparés (≥ 3 exigés). Voir `prompts/`.

| Prompt | Fichier | Idée directrice |
|---|---|---|
| baseline | `prompts/baseline_prompt.txt` | Consigne minimale + schéma JSON |
| improved | `prompts/improved_prompt.txt` | + contrôle qualité image, « incertain si doute », confiance < 0.60 → incertain |
| structured | `prompts/structured_prompt.txt` | + ordre de raisonnement imposé, **« pas d'invention »**, justification courte, JSON strict |

## Protocole

Indicateurs mesurés sur la **sortie brute** du modèle (avant garde-fous), sur les
**30 cas finaux**, backend `noisy` (modèle synthétique reproductible), graine 203.

```bash
python eval/compare_prompts.py --split final
```

> Le backend `noisy` modélise un modèle imparfait mais déterministe. Ce ne sont pas
> des performances médicales : ils illustrent l'effet du prompt sur la **conformité
> de format** et la prudence. Sur GPU, relancer avec `--backend vlm` (MedGemma/Gemma).

## Résultats (§7.3)

| Critère | baseline | improved | structured | Objectif amélioré |
|---|---|---|---|---|
| JSON valide | 80 % | 100 % | 100 % | ≥ 95 % |
| Justification courte | 37 % | 80 % | 97 % | ≥ 90 % |
| Avertissement présent (brut) | 57 % | 83 % | 100 % | 100 % |
| Hallucination | 7 % | 0 % | 0 % | 0 % |

Lecture : le prompt `structured` atteint les quatre objectifs de format. Le prompt
`improved` les atteint presque (avertissement brut 83 %).

## Rôle des garde-fous

Les indicateurs ci-dessus portent sur le **texte brut** du modèle. En **production**,
la couche `src/guardrails.py` (validateur strict + reprompt + avertissement forcé)
porte **JSON valide et avertissement présent à 100 %**, quel que soit le prompt.
Le prompt réduit le travail des garde-fous ; les garde-fous garantissent le contrat.

## Effet sur la classification (rappel, voir L8)

Métriques brutes (avant garde-fous) sur les 30 cas finaux :

| Métrique | baseline | improved | structured |
|---|---|---|---|
| Accuracy | 0.567 | 0.733 | 0.833 |
| Macro-F1 | 0.564 | 0.731 | 0.833 |
| Sensibilité | 0.60 | 0.80 | 0.90 |
| Spécificité | 0.70 | 0.80 | 0.80 |

Le prompt amélioré apporte un gain net et monotone. Le détail (matrice de confusion,
tableau Δ, effet des garde-fous) figure dans `docs/rapport_mesures.md`.
