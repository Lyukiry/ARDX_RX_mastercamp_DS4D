# Note de cadrage (livrable L1) — périmètre gelé

> **Projet :** Assistant radiologue virtuel — prototype pédagogique (EFREI 2025-2026)
> **Statut du périmètre :** **GELÉ**
> **Position :** non clinique. Aucun diagnostic. Avertissement sur 100 % des sorties.

## 1. Objet

Analyser **une radiographie thoracique frontale** et produire une sortie
**structurée, justifiée et tracée** (JSON 7 champs + justification + avertissement),
accessible via une interface web de démonstration. Usage **pédagogique uniquement**.

## 2. Les 3 classes (taxonomie gelée)

Variante retenue (cohérente avec l'esprit prudent du cahier des charges §2.3).
Les **clés internes** restent en anglais (stables pour le code, la base et les tests) ;
l'intitulé français est l'étiquette d'affichage.

| Intitulé (FR) | Clé interne | Définition |
|---|---|---|
| Normal | `normal` | Pas d'opacité détectée |
| Suspicion d'opacité | `suspected_opacity` | Opacité possible détectée |
| Incertain | `uncertain` | Qualité ou signes non concluants |

La classe `uncertain` est un **garde-fou**, pas un échec : elle ne doit jamais être
supprimée.

## 3. Périmètre

**Inclus :** 1 radiographie thoracique frontale ; classification 3 classes ; JSON
validé + justification + avertissement ; démo web (onglets Cas / Analyse /
Apprentissage / Suivi) ; traçabilité CSV + SQLite.

**Exclu :** profil, scanner, IRM, autres modalités ; tout diagnostic clinique
définitif (objectif **0 diagnostic clinique**) ; toute utilisation en conditions
réelles de soin.

## 4. Contrat de sortie (7 champs)

`image_quality`, `predicted_class`, `confidence`, `visual_evidence`,
`justification`, `limitations`, `warning`. Détail et règles : `prompts/json_schema.md`.

Règle d'avertissement (§7.2) : escalade d'incertitude dès que `confidence < 0.60`
**OU** `image_quality = mauvaise` (champ `uncertainty_warning`).

## 5. Stratégie de modélisation (P0 → P3)

| Phase | Contenu | Implémentation dans le dépôt |
|---|---|---|
| P0 — Prompting | Baseline généraliste + prompt soigné | `prompts/*.txt`, backend `vlm` (MedGemma/Gemma) |
| P1 — LoRA rapide | Unsloth + Gemma 4 E2B/E4B, multimodal | `finetuning/gemma4_unsloth_lora.py` |
| P2 — Validation | Évaluation quanti + quali, sécurité/biais | `eval/`, `docs/rapport_mesures.md` |
| P3 — Extension | Plus de données, MedGemma PEFT/QLoRA | `finetuning/medgemma_peft_qlora.py` |

Règle d'or : **baseline → amélioration → démo**, jamais l'inverse. **Mesurer avant
d'optimiser.** LoRA d'abord, pas de full fine-tuning.

## 6. Objectifs chiffrés (rappel, voir L8)

Accuracy ≥ 0.70 · Macro-F1 ≥ 0.68 · JSON valide ≥ 95 % · Hallucination 0 % ·
Avertissement 100 % · Latence < 10 s · Sorties journalisées 100 % ·
Erreurs catégorisées 100 % (30 cas finaux).

## 7. Données

| Palier | Volume | Rôle | Dans le dépôt |
|---|---|---|---|
| Smoke | 20 images | Valider la chaîne | `data/` split `smoke` |
| Développement | 100-150 cas | Mise au point | RSNA sur PC GPU (non redistribué) |
| Évaluation finale | **30 cas commentés** | Mesures + analyse d'erreurs | `data/` split `final` |

Dataset principal d'un vrai projet : **RSNA Pneumonia** (accès Kaggle, licence
propre). Le dépôt n'embarque qu'un **jeu synthétique jouet** (sans valeur médicale)
pour rester exécutable et public. Voir `docs/guide_execution_gpu.md`.

## 8. Avertissement

Ce système ne pose aucun diagnostic et ne se substitue jamais à un radiologue.
**Aucune valeur diagnostique.**
