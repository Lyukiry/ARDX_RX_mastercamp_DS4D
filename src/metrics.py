from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Iterable

CLASSES = ["normal", "suspected_opacity", "uncertain"]


def accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


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


def recall_for_class(y_true: Iterable[str], y_pred: Iterable[str], target: str) -> float:
    """Rappel d'une classe : VP / (cas réels de cette classe)."""
    y_true = list(y_true); y_pred = list(y_pred)
    support = sum(t == target for t in y_true)
    if not support:
        return 0.0
    hits = sum(t == target and p == target for t, p in zip(y_true, y_pred))
    return hits / support


def sensitivity(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Sensibilité = rappel des cas `suspected_opacity` (éviter les faux négatifs)."""
    return recall_for_class(y_true, y_pred, "suspected_opacity")


def specificity(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Spécificité = rappel des cas `normal` (limiter les sur-alertes)."""
    return recall_for_class(y_true, y_pred, "normal")


def confusion_counts(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, int]:
    counts: Counter = Counter()
    for t, p in zip(y_true, y_pred):
        counts[f"{t}__{p}"] += 1
    return dict(counts)


def confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> dict[str, dict[str, int]]:
    """Matrice de confusion {classe_réelle: {classe_prédite: effectif}} (livrable L8)."""
    matrix = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    """Calcule les 6 métriques du cahier des charges + indicateurs de prudence."""
    y_true = [r["label"] for r in rows]
    y_pred = [r["predicted_class"] for r in rows]
    json_valid = [r.get("json_valid", True) for r in rows]
    warnings = [bool(r.get("warning")) for r in rows]
    latencies = [int(r.get("latency_ms", 0) or 0) for r in rows]
    return {
        "n": len(rows),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "sensitivity": round(sensitivity(y_true, y_pred), 4),
        "specificity": round(specificity(y_true, y_pred), 4),
        "json_valid_rate": round(sum(json_valid) / len(json_valid), 4) if rows else 0,
        "warning_rate": round(sum(warnings) / len(warnings), 4) if rows else 0,
        "uncertain_rate": round(sum(p == "uncertain" for p in y_pred) / len(y_pred), 4) if rows else 0,
        "median_latency_ms": round(median(latencies), 1) if latencies else 0,
    }
