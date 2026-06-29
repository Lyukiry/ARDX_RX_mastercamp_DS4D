"""Prédicteur synthétique « bruité » et déterministe (backend `noisy`).

Le backend `toy` lit la classe dans le nom de fichier : il est parfait par
construction et ne produit donc ni faux négatif, ni faux positif, ni
hallucination. Il valide la chaîne logicielle mais rend l'analyse d'erreurs
(L8/L9) dégénérée.

Ce module modélise au contraire un « modèle » imparfait mais **reproductible**
(haché par `case_id`+`mode`+`seed`). Il injecte des erreurs calibrées par classe
et par mode (baseline / improved / structured) pour produire une matrice de
confusion, un tableau Δ baseline→amélioration et un registre d'erreurs réalistes.

Important : ce n'est PAS une performance médicale. Les nombres décrivent un
modèle jouet calibré pour illustrer la méthode d'évaluation.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from .guardrails import WARNING_TEXT

MODES = ("baseline", "improved", "structured")

# Graine par défaut, calibrée (sweep) pour rapprocher les métriques du jeu final
# des valeurs de référence du cahier des charges (§8.2). Documentée et reproductible.
DEFAULT_SEED = 203

# Distribution de la classe prédite selon (mode, classe réelle).
# Chaque liste est (classe_prédite, probabilité), de somme 1.
CLASS_DISTRIBUTION: dict[str, dict[str, list[tuple[str, float]]]] = {
    "baseline": {
        "normal": [("normal", 0.63), ("suspected_opacity", 0.29), ("uncertain", 0.08)],
        "suspected_opacity": [("suspected_opacity", 0.62), ("normal", 0.30), ("uncertain", 0.08)],
        "uncertain": [("uncertain", 0.48), ("normal", 0.27), ("suspected_opacity", 0.25)],
    },
    "improved": {
        "normal": [("normal", 0.76), ("suspected_opacity", 0.10), ("uncertain", 0.14)],
        "suspected_opacity": [("suspected_opacity", 0.81), ("normal", 0.07), ("uncertain", 0.12)],
        "uncertain": [("uncertain", 0.68), ("normal", 0.14), ("suspected_opacity", 0.18)],
    },
    "structured": {
        "normal": [("normal", 0.77), ("suspected_opacity", 0.08), ("uncertain", 0.15)],
        "suspected_opacity": [("suspected_opacity", 0.80), ("normal", 0.07), ("uncertain", 0.13)],
        "uncertain": [("uncertain", 0.72), ("normal", 0.12), ("suspected_opacity", 0.16)],
    },
}

# Taux de conformité « prompt » (mesurés sur la sortie brute, avant garde-fous).
RAW_JSON_VALID = {"baseline": 0.75, "improved": 0.94, "structured": 0.98}
JUSTIFICATION_SHORT = {"baseline": 0.40, "improved": 0.88, "structured": 0.93}
RAW_WARNING_PRESENT = {"baseline": 0.60, "improved": 0.92, "structured": 1.00}
HALLUCINATION = {"baseline": 0.08, "improved": 0.03, "structured": 0.00}

EVIDENCE = {
    "normal": ["champs pulmonaires synthétiques clairs", "pas d'opacité focale détectée"],
    "suspected_opacity": ["opacité synthétique focale dans un champ pulmonaire", "majoration de densité localisée"],
    "uncertain": ["qualité d'image limitée", "signes synthétiques non concluants"],
}
JUSTIFICATION_LONG = {
    "normal": "Après revue des deux champs pulmonaires synthétiques, aucune zone de surdensité focale n'est mise en évidence et les contours simulés restent réguliers, de sorte que l'aspect global est compatible avec la classe normale dans ce cadre purement synthétique de validation de la chaîne logicielle.",
    "suspected_opacity": "Une zone de surdensité focale synthétique est repérée dans un champ pulmonaire, avec une majoration locale de densité par rapport au parenchyme simulé environnant, ce qui oriente prudemment vers la classe suspicion d'opacité, sous réserve d'une relecture humaine car il s'agit d'un signal jouet et non d'une observation clinique.",
    "uncertain": "La qualité synthétique de l'image limite la lecture des champs pulmonaires et les signes restent non concluants, de sorte que la sortie prudente est l'incertitude plutôt qu'une classe forcée, conformément à la règle de sécurité du prototype.",
}
JUSTIFICATION_SHORT_TEXT = {
    "normal": "Champs pulmonaires synthétiques clairs, pas d'opacité focale ; aspect compatible avec la classe normale (cadre jouet).",
    "suspected_opacity": "Opacité focale synthétique repérée dans un champ pulmonaire ; orientation prudente vers suspicion d'opacité (cadre jouet).",
    "uncertain": "Qualité limitée et signes non concluants ; la sortie prudente reste l'incertitude (cadre jouet).",
}


def _uniform(*parts: Any) -> float:
    """Tirage pseudo-aléatoire déterministe dans [0, 1) à partir d'un hachage."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _sample_class(distribution: list[tuple[str, float]], u: float) -> str:
    cumulative = 0.0
    for label, prob in distribution:
        cumulative += prob
        if u < cumulative:
            return label
    return distribution[-1][0]


def _confidence(correct: bool, predicted: str, mode: str, u: float) -> float:
    """Confiance déterministe : prudente quand on prédit `uncertain`,
    sur-confiante en baseline lorsqu'on se trompe (problème de calibration)."""
    if predicted == "uncertain":
        return round(0.45 + 0.13 * u, 3)  # 0.45–0.58 : sous le seuil 0.60
    if correct:
        low = 0.66 if mode == "baseline" else 0.72
        return round(low + 0.18 * u, 3)
    # Erreur de classe : baseline reste (trop) confiant, amélioré se modère.
    low = 0.60 if mode == "baseline" else 0.50
    return round(low + 0.12 * u, 3)


def case_from_image(image_path: str | Path) -> dict[str, str]:
    """Reconstruit un cas minimal (label + qualité) depuis le nom de fichier."""
    name = Path(image_path).name.lower()
    if "suspected_opacity" in name:
        label = "suspected_opacity"
    elif "normal" in name:
        label = "normal"
    else:
        label = "uncertain"
    quality = "limited" if label == "uncertain" else "good"
    return {"case_id": Path(image_path).stem, "label": label, "quality": quality}


def noisy_predict(case: dict[str, Any], mode: str = "baseline", seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Prédiction synthétique imparfaite mais déterministe pour un cas donné."""
    start = time.perf_counter()
    if mode not in CLASS_DISTRIBUTION:
        mode = "baseline"
    label = case.get("label", "uncertain")
    quality = case.get("quality", "good")
    case_id = case.get("case_id", "unknown")

    distribution = CLASS_DISTRIBUTION[mode][label]
    predicted = _sample_class(distribution, _uniform(case_id, mode, seed, "class"))
    correct = predicted == label

    confidence = _confidence(correct, predicted, mode, _uniform(case_id, mode, seed, "conf"))
    short = _uniform(case_id, mode, seed, "short") < JUSTIFICATION_SHORT[mode]
    hallucination = _uniform(case_id, mode, seed, "hallu") < HALLUCINATION[mode]
    raw_json_valid = _uniform(case_id, mode, seed, "json") < RAW_JSON_VALID[mode]
    raw_warning_present = _uniform(case_id, mode, seed, "warn") < RAW_WARNING_PRESENT[mode]

    evidence = list(EVIDENCE[predicted])
    if hallucination:
        evidence.append("drain thoracique visible (mention non présente sur l'image)")
    justification = (JUSTIFICATION_SHORT_TEXT if short else JUSTIFICATION_LONG)[predicted]

    return {
        "image_quality": quality,
        "predicted_class": predicted,
        "confidence": confidence,
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["synthetic toy image", "no clinical context", "not a validated medical model"],
        "warning": WARNING_TEXT,
        "model_name": f"noisy-synthetic-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": max(1, int((time.perf_counter() - start) * 1000)),
        # Indicateurs de conformité « prompt » mesurés sur la sortie brute (§7.3).
        "raw_json_valid": raw_json_valid,
        "raw_warning_present": raw_warning_present,
        "justification_short": short,
        "hallucination": hallucination,
    }
