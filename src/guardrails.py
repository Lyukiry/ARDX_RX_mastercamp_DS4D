from __future__ import annotations

from typing import Any

ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
ALLOWED_QUALITIES = {"good", "limited", "poor"}
REQUIRED_KEYS = {
    "image_quality",
    "predicted_class",
    "confidence",
    "visual_evidence",
    "justification",
    "limitations",
    "warning",
}

# Avertissement non clinique inconditionnel : présent sur 100 % des sorties.
WARNING_TEXT = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Escalade d'incertitude conditionnelle (cahier des charges §7.2).
UNCERTAINTY_WARNING_TEXT = (
    "Incertitude élevée (confiance < 0.60 ou qualité image insuffisante) : "
    "relecture humaine et contrôle qualité recommandés avant toute interprétation."
)

# Seuil de déclenchement de l'incertitude (§7.2).
CONFIDENCE_THRESHOLD = 0.60


def validate_prediction(pred: dict[str, Any]) -> tuple[bool, list[str]]:
    """Valide une sortie contre le contrat JSON à 7 champs (livrable L4)."""
    errors: list[str] = []
    missing = REQUIRED_KEYS - set(pred)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if pred.get("predicted_class") not in ALLOWED_CLASSES:
        errors.append("invalid predicted_class")
    if "image_quality" in pred and pred["image_quality"] not in ALLOWED_QUALITIES:
        errors.append("invalid image_quality")
    try:
        conf = float(pred.get("confidence", -1))
        if not 0 <= conf <= 1:
            errors.append("confidence outside [0,1]")
    except (TypeError, ValueError):
        errors.append("confidence is not numeric")
    if "visual_evidence" in pred and not isinstance(pred["visual_evidence"], list):
        errors.append("visual_evidence must be a list")
    if "limitations" in pred and not isinstance(pred["limitations"], list):
        errors.append("limitations must be a list")
    if "justification" in pred and not str(pred.get("justification", "")).strip():
        errors.append("justification is empty")
    if not pred.get("warning"):
        errors.append("warning missing")
    return not errors, errors


def needs_uncertainty_warning(pred: dict[str, Any]) -> bool:
    """Règle §7.2 : incertitude si confidence < 0.60 OU image_quality = mauvaise."""
    try:
        conf = float(pred.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return conf < CONFIDENCE_THRESHOLD or pred.get("image_quality") == "poor"


def apply_safety_guardrails(pred: dict[str, Any]) -> dict[str, Any]:
    """Applique les garde-fous de sécurité clinique avant restitution.

    - Toute sortie invalide bascule en `uncertain` avec confiance plafonnée.
    - Qualité insuffisante + confiance faible -> `uncertain` (prudence).
    - Avertissement non clinique forcé (100 %).
    - Escalade d'incertitude conditionnelle ajoutée selon la règle §7.2.
    """
    valid, errors = validate_prediction(pred)
    if not valid:
        pred["predicted_class"] = "uncertain"
        pred["confidence"] = min(float(pred.get("confidence", 0.0) or 0.0), 0.5)
        pred.setdefault("limitations", [])
        if "not a validated medical model" not in pred["limitations"]:
            pred["limitations"].append("not a validated medical model")
        pred["limitations"].append("guardrail triggered: invalid output schema")
    if pred.get("image_quality") in {"limited", "poor"} and float(pred.get("confidence", 0)) < CONFIDENCE_THRESHOLD:
        pred["predicted_class"] = "uncertain"
    pred["warning"] = WARNING_TEXT
    pred["uncertainty_warning"] = UNCERTAINTY_WARNING_TEXT if needs_uncertainty_warning(pred) else ""
    pred["guardrail_errors"] = errors
    return pred
