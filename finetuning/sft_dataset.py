"""Build a supervised fine-tuning (SFT) dataset from a ``cases.csv``.

Each training example is a triple ``(image, instruction, gold JSON answer)`` that
matches the project's output contract. The gold answer is **templated** from the
dataset label (RSNA gives ``normal`` / ``suspected_opacity``): this teaches the
model the JSON format and the class vocabulary, not a hand-written radiology
report. That weak-supervision choice is a documented limitation (see
``docs/rapport.md``).

Pure Python -- safe to import on a torch-free runner.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.guardrails import WARNING_TEXT  # noqa: E402
from src.vlm_inference import load_prompt  # noqa: E402

_EVIDENCE = {
    "normal": ["lung fields appear clear", "no focal opacity identified"],
    "suspected_opacity": ["area of increased opacity in a lung field",
                          "possible consolidation pattern"],
    "uncertain": ["image features are inconclusive"],
}
_JUSTIFY = {
    "normal": "No focal opacity is visible on this frontal view; structures appear within expected limits for this educational task.",
    "suspected_opacity": "An area of increased opacity is visible and is compatible with the suspected-opacity class for this educational task.",
    "uncertain": "The visible evidence is insufficient to commit to a class; uncertainty is the safe output.",
}


def gold_answer(label: str, confidence: float = 0.9) -> dict[str, Any]:
    """Build a templated gold JSON answer for a dataset label."""
    label = label if label in _EVIDENCE else "uncertain"
    return {
        "image_quality": "good",
        "predicted_class": label,
        "confidence": confidence if label != "uncertain" else 0.5,
        "visual_evidence": _EVIDENCE[label],
        "justification": _JUSTIFY[label],
        "limitations": ["templated supervision", "no clinical context", "not a validated medical model"],
        "warning": WARNING_TEXT,
    }


def build_examples(cases_csv, mode: str = "improved", split: str | None = "dev") -> list[dict[str, Any]]:
    """Return SFT examples from a ``cases.csv`` produced by ``src.datasets``."""
    cases_csv = Path(cases_csv)
    instruction = load_prompt(mode)
    examples: list[dict[str, Any]] = []
    with cases_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split and row.get("split") != split:
                continue
            examples.append({
                "image_path": str(ROOT / row["image_path"]),
                "instruction": instruction,
                "answer": json.dumps(gold_answer(row["label"]), ensure_ascii=False),
            })
    return examples
