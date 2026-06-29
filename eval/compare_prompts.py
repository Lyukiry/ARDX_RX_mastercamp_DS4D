"""Comparaison des prompts (livrable L3 / cahier des charges §7.3).

Compare au moins 3 variantes de prompt (baseline / improved / structured) sur le
jeu synthétique via le backend `noisy`, et calcule les indicateurs de conformité
mesurés sur la sortie BRUTE du modèle (avant garde-fous) :

- JSON valide
- Justification courte
- Avertissement présent
- Hallucination

Par défaut le script n'écrit rien sur disque (il imprime un JSON). Avec
`--out-dir`, il écrit un CSV et un tableau Markdown.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.metrics import accuracy, macro_f1, sensitivity, specificity  # noqa: E402
from src.synthetic_eval import DEFAULT_SEED, MODES, noisy_predict  # noqa: E402


def read_cases(split: str) -> list[dict]:
    with (ROOT / "data" / "synthetic_cases.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [r for r in rows if split == "all" or r["split"] == split]


def rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def evaluate_mode(cases: list[dict], mode: str, seed: int) -> dict:
    preds = [noisy_predict(c, mode=mode, seed=seed) for c in cases]
    y_true = [c["label"] for c in cases]
    y_pred = [p["predicted_class"] for p in preds]
    # Métriques calculées sur la sortie BRUTE du prompt (avant garde-fous).
    return {
        "prompt": mode,
        "n": len(cases),
        "json_valid_rate": rate([p["raw_json_valid"] for p in preds]),
        "short_justification_rate": rate([p["justification_short"] for p in preds]),
        "warning_present_rate": rate([p["raw_warning_present"] for p in preds]),
        "hallucination_rate": rate([p["hallucination"] for p in preds]),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "sensitivity": round(sensitivity(y_true, y_pred), 4),
        "specificity": round(specificity(y_true, y_pred), 4),
    }


def to_markdown(summary: list[dict]) -> str:
    header = "| Prompt | JSON valide | Justification courte | Avertissement présent | Hallucination |\n"
    header += "|---|---|---|---|---|\n"
    lines = [
        f"| {row['prompt']} | {row['json_valid_rate']:.0%} | {row['short_justification_rate']:.0%} "
        f"| {row['warning_present_rate']:.0%} | {row['hallucination_rate']:.0%} |"
        for row in summary
    ]
    return header + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["all", "smoke", "final"], default="final")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cases = read_cases(args.split)
    summary = [evaluate_mode(cases, mode, args.seed) for mode in MODES]

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "prompt_comparison.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        (args.out_dir / "prompt_comparison.md").write_text(to_markdown(summary), encoding="utf-8")

    print(json.dumps({"split": args.split, "seed": args.seed, "comparison": summary}, indent=2))


if __name__ == "__main__":
    main()
