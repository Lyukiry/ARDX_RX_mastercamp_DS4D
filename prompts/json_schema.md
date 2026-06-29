# Contrat de sortie JSON (livrable L4)

Schéma obligatoire à **7 champs** (cahier des charges §7.1). Les clés restent en
anglais (stables pour le code, la base et les tests) ; la correspondance avec les
intitulés français du cahier des charges est documentée plus bas.

```json
{
  "image_quality": "good | limited | poor",
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": ["observation factuelle des signes visibles (opacités, consolidations, volumes...)"],
  "justification": "raisonnement clinique bref reliant les signes à la classe prédite (2 à 4 phrases).",
  "limitations": ["facteurs limitants (qualité, artefacts, incertitudes, données manquantes...)"],
  "warning": "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."
}
```

## Champ ajouté par les garde-fous

Le pipeline ajoute après validation un champ technique :

```json
"uncertainty_warning": "Incertitude élevée (...) : relecture humaine et contrôle qualité recommandés."
```

### Pourquoi deux champs d'avertissement ?

Le cahier des charges (§7.2) demande de **signaler l'incertitude** dès que
`confidence < 0.60` **OU** `image_quality = mauvaise`. Nous séparons deux rôles :

- `warning` : avertissement **non clinique inconditionnel**, présent sur **100 %**
  des sorties (objectif « avertissement présent = 100 % »).
- `uncertainty_warning` : **escalade conditionnelle** déclenchée par la règle §7.2.
  Vide lorsque la sortie est confiante et de bonne qualité.

Ce découpage est plus prudent qu'un message unique : la position non clinique est
toujours rappelée, et l'incertitude est signalée explicitement quand elle existe.

## Correspondance français ↔ clé interne

| Cahier des charges (FR) | Clé interne | Valeurs |
|---|---|---|
| Qualité image (bonne / moyenne / mauvaise) | `image_quality` | `good` / `limited` / `poor` |
| Classe prédite (Normal / Suspicion d'opacité / Incertain) | `predicted_class` | `normal` / `suspected_opacity` / `uncertain` |
| Confiance | `confidence` | `0.0` – `1.0` |
| Preuves visuelles | `visual_evidence` | liste de chaînes |
| Justification | `justification` | chaîne (2 à 4 phrases) |
| Limites | `limitations` | liste de chaînes |
| Avertissement | `warning` | constante non clinique |

## Règles de validation (`src/guardrails.py`)

- `predicted_class` ∈ {`normal`, `suspected_opacity`, `uncertain`}.
- `image_quality` ∈ {`good`, `limited`, `poor`} si présent.
- `confidence` numérique dans `[0, 1]`.
- `visual_evidence` et `limitations` doivent être des listes.
- `justification` non vide.
- `warning` obligatoirement présent.
- Toute sortie invalide bascule en `uncertain` (confiance plafonnée à 0.5).
- Si `image_quality` ∈ {`limited`, `poor`} et `confidence < 0.60` → `uncertain`.
