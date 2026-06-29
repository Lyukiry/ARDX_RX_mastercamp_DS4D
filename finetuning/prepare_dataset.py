"""Construit un jeu d'instructions (image + prompt -> JSON cible) pour le fine-tuning.

Tourne sur Mac (stdlib uniquement) : convertit un CSV de cas (mêmes colonnes que
`data/synthetic_cases.csv`, ou un vrai dataset RSNA dé-identifié) en JSONL prêt
pour l'entraînement LoRA/QLoRA. Chaque ligne :

    {"image_path": "...", "prompt": "<prompt amélioré>", "response": "<JSON 7 champs>"}

Usage :
    python finetuning/prepare_dataset.py --csv data/synthetic_cases.csv \
        --split final --out finetuning/data/train.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPROVED_PROMPT = (ROOT / "prompts" / "improved_prompt.txt").read_text(encoding="utf-8")
WARNING_TEXT = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Réponses cibles canoniques par classe (sortie attendue du modèle).
TARGETS = {
    "normal": {
        "image_quality": "good",
        "predicted_class": "normal",
        "confidence": 0.82,
        "visual_evidence": ["lung fields appear clear", "no focal opacity detected"],
        "justification": "The lung fields look clear without focal consolidation, which is compatible with the normal class in this educational setting.",
        "limitations": ["no clinical context", "not a validated medical model"],
    },
    "suspected_opacity": {
        "image_quality": "good",
        "predicted_class": "suspected_opacity",
        "confidence": 0.80,
        "visual_evidence": ["focal area of increased density in a lung field"],
        "justification": "A focal area of increased density is visible in a lung field, cautiously suggesting a possible opacity that requires human review.",
        "limitations": ["no clinical context", "not a validated medical model"],
    },
    "uncertain": {
        "image_quality": "limited",
        "predicted_class": "uncertain",
        "confidence": 0.50,
        "visual_evidence": ["limited image quality", "inconclusive signs"],
        "justification": "Image quality is limited and the signs are inconclusive, so the safe output is uncertainty rather than a forced class.",
        "limitations": ["limited image quality", "not a validated medical model"],
    },
}


def build_response(label: str) -> str:
    target = dict(TARGETS.get(label, TARGETS["uncertain"]))
    target["warning"] = WARNING_TEXT
    return json.dumps(target, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "synthetic_cases.csv")
    parser.add_argument("--split", default="all", help="all | smoke | final | dev")
    parser.add_argument("--out", type=Path, default=ROOT / "finetuning" / "data" / "train.jsonl")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if args.split == "all" or r.get("split") == args.split]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for row in rows:
            record = {
                "image_path": str(ROOT / row["image_path"]),
                "prompt": IMPROVED_PROMPT,
                "response": build_response(row["label"]),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Écrit {len(rows)} exemples d'instruction dans {args.out}")


if __name__ == "__main__":
    main()
