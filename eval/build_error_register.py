"""Construit le registre d'analyse d'erreurs des 30 cas finaux (livrables L7/L9).

Pour chaque cas final, prédit avec le backend `noisy` en mode baseline ET amélioré,
catégorise selon la taxonomie du cahier des charges (§9.1) :

    FN = faux négatif · FP = faux positif · UA = incertitude acceptable
    HT = hallucination textuelle · OK = cas correct

Écrit `eval/error_register_final.csv` (100 % des cas commentés et tracés) et imprime
la répartition par type ainsi que le Top 5 des causes de panne.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.guardrails import apply_safety_guardrails  # noqa: E402
from src.synthetic_eval import noisy_predict  # noqa: E402

OUT_PATH = ROOT / "eval" / "error_register_final.csv"

CORRECTIVE = {
    "FN": "Enrichir les données, revoir le seuil, renforcer la sensibilité",
    "FP": "Réduire les faux signaux, ajuster le seuil, améliorer la spécificité",
    "UA": "Calibrer l'incertitude, affiner les seuils, enrichir le contexte",
    "HT": "Contraindre la génération, prompt « pas d'invention », garde-fous",
    "OK": "Aucune (cas correct)",
}


def classify(label: str, pred: str, hallucination: bool) -> str:
    if hallucination:
        return "HT"  # information inventée : défaut dominant même si la classe est correcte
    if pred == label:
        return "OK"
    if pred == "uncertain":
        return "UA"  # doute signalé sans conclure
    if pred == "normal":
        return "FN"  # anomalie/ambiguïté sous-estimée
    return "FP"  # opacité prédite à tort


def cause(error_type: str, quality: str) -> str:
    if error_type == "HT":
        return "texte inventé (mention non présente sur l'image)"
    if error_type == "FN":
        return "opacité masquée par mauvaise qualité" if quality == "poor" else "opacité synthétique peu marquée"
    if error_type == "FP":
        return "artefact pris pour une opacité" if quality != "good" else "structure normale sur-interprétée"
    if error_type == "UA":
        return "qualité limitée / signes non concluants"
    return ""


def read_final_cases() -> list[dict]:
    with (ROOT / "data" / "synthetic_cases.csv").open(newline="", encoding="utf-8") as file:
        return [r for r in csv.DictReader(file) if r["split"] == "final"]


def main() -> None:
    cases = read_final_cases()
    rows = []
    for case in cases:
        for model in ("baseline", "improved"):
            raw = noisy_predict(case, mode=model)
            # Catégorisation sur la sortie du système livré (après garde-fous) ;
            # l'hallucination est lue sur la sortie brute (défaut de génération).
            pred = apply_safety_guardrails(dict(raw))
            error_type = classify(case["label"], pred["predicted_class"], raw["hallucination"])
            rows.append({
                "case_id": case["case_id"],
                "ground_truth": case["label"],
                "model": model,
                "prediction": pred["predicted_class"],
                "confidence": pred["confidence"],
                "quality": case["quality"],
                "error_type": error_type,
                "cause": cause(error_type, case["quality"]),
                "comment": f"vérité {case['label']}, prédit {pred['predicted_class']} (qualité {case['quality']})",
                "corrective_action": CORRECTIVE[error_type],
            })

    with OUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Écrit {len(rows)} lignes ({len(cases)} cas × 2 modèles) dans {OUT_PATH}\n")
    for model in ("baseline", "improved"):
        counts = Counter(r["error_type"] for r in rows if r["model"] == model)
        print(f"{model:9s} -> " + ", ".join(f"{k}:{counts.get(k, 0)}" for k in ("OK", "FN", "FP", "UA", "HT")))
    causes = Counter(r["cause"] for r in rows if r["error_type"] != "OK")
    print("\nTop 5 des causes de panne :")
    for label, n in causes.most_common(5):
        print(f"  {n:2d}  {label}")


if __name__ == "__main__":
    main()
