from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Iterable

CLASSES = ["normal", "suspected_opacity", "uncertain"]

# Binary view for clinical sensitivity / specificity:
#   positive = suspected_opacity (the finding we must not miss)
#   negative = normal
# A prediction counts as "positive" only when it is suspected_opacity; an
# ``uncertain`` prediction is treated as "not an alert" (not a false positive),
# which rewards specificity but penalises sensitivity -- the safe, conservative
# convention for a cautious assistant.
POSITIVE_CLASS = "suspected_opacity"
NEGATIVE_CLASS = "normal"


def accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def sensitivity(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Recall on the positive (suspected_opacity) class. Protects against FN."""
    y_true = list(y_true); y_pred = list(y_pred)
    positives = [p for t, p in zip(y_true, y_pred) if t == POSITIVE_CLASS]
    if not positives:
        return 0.0
    return sum(p == POSITIVE_CLASS for p in positives) / len(positives)


def specificity(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Fraction of true-normal cases that are not over-called as opacity. Limits FP."""
    y_true = list(y_true); y_pred = list(y_pred)
    negatives = [p for t, p in zip(y_true, y_pred) if t == NEGATIVE_CLASS]
    if not negatives:
        return 0.0
    return sum(p != POSITIVE_CLASS for p in negatives) / len(negatives)


def classify_error(label: str, pred: str, json_valid: bool = True) -> str:
    """Return the error taxonomy code (see docs/evaluation_protocol.md)."""
    if not json_valid:
        return "JF"
    if label == pred:
        return "correct"
    if label == POSITIVE_CLASS and pred in {"normal", "uncertain"}:
        return "FN"
    if label == NEGATIVE_CLASS and pred == POSITIVE_CLASS:
        return "FP"
    if pred == "uncertain":
        return "UA"
    return "OTHER"


def macro_f1(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    scores = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        scores.append(f1)
    return sum(scores) / len(scores)


def confusion_counts(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, int]:
    counts = Counter()
    for t, p in zip(y_true, y_pred):
        counts[f"{t}__{p}"] += 1
    return dict(counts)


def median_latency_ms(rows: list[dict]) -> float:
    values = [float(r.get("latency_ms", 0) or 0) for r in rows]
    return round(median(values), 1) if values else 0.0


def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    y_true = [r["label"] for r in rows]
    y_pred = [r["predicted_class"] for r in rows]
    json_valid = [r.get("json_valid", True) for r in rows]
    warnings = [bool(r.get("warning")) for r in rows]
    return {
        "n": len(rows),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "sensitivity": round(sensitivity(y_true, y_pred), 4),
        "specificity": round(specificity(y_true, y_pred), 4),
        "json_valid_rate": round(sum(json_valid) / len(json_valid), 4) if rows else 0,
        "warning_rate": round(sum(warnings) / len(warnings), 4) if rows else 0,
        "uncertain_rate": round(sum(p == "uncertain" for p in y_pred) / len(y_pred), 4) if rows else 0,
        "median_latency_ms": median_latency_ms(rows),
    }
