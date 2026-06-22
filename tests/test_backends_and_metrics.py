"""Torch-free unit tests for the real-backend helpers and extended metrics.

These cover the pure-Python logic that backs the VLM / classifier / dataset
features without importing torch or transformers, so CI stays light.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.datasets import chexpert_label, rsna_label, stratified_split
from src.guardrails import WARNING_TEXT, validate_prediction
from src.inference import BACKENDS, predict
from src.metrics import classify_error, sensitivity, specificity, summarize_metrics
from src.vlm_inference import (apply_uncertainty_rule, coerce_to_schema,
                               extract_json, load_prompt)


# --- JSON extraction -------------------------------------------------------- #
def test_extract_json_from_fenced_block() -> None:
    text = 'Sure!\n```json\n{"predicted_class": "normal", "confidence": 0.8}\n```\nDone.'
    obj = extract_json(text)
    assert obj == {"predicted_class": "normal", "confidence": 0.8}


def test_extract_json_ignores_braces_inside_strings() -> None:
    text = 'noise {"justification": "a } brace in text", "predicted_class": "uncertain"} tail'
    obj = extract_json(text)
    assert obj["predicted_class"] == "uncertain"
    assert "brace" in obj["justification"]


def test_extract_json_returns_none_on_garbage() -> None:
    assert extract_json("no json here at all") is None


# --- schema coercion -------------------------------------------------------- #
def test_coerce_maps_synonyms_and_clamps_confidence() -> None:
    raw = {"predicted_class": "pneumonia", "confidence": 1.7, "image_quality": "bad"}
    pred = coerce_to_schema(raw, model_name="m", prompt_version="v", latency_ms=5)
    assert pred["predicted_class"] == "suspected_opacity"
    assert pred["confidence"] == 1.0
    assert pred["image_quality"] == "poor"
    assert pred["warning"] == WARNING_TEXT
    valid, errors = validate_prediction(pred)
    assert valid, errors


def test_coerce_none_is_safe_uncertain() -> None:
    pred = coerce_to_schema(None, model_name="m", prompt_version="v", latency_ms=0)
    assert pred["predicted_class"] == "uncertain"
    assert "not a validated medical model" in pred["limitations"]
    valid, _ = validate_prediction(pred)
    assert valid


def test_uncertainty_rule_downgrades_low_confidence() -> None:
    pred = {"predicted_class": "suspected_opacity", "confidence": 0.4, "image_quality": "good"}
    assert apply_uncertainty_rule(pred)["predicted_class"] == "uncertain"


def test_prompts_load() -> None:
    assert "JSON" in load_prompt("baseline")
    assert "0.60" in load_prompt("improved")


# --- metrics ---------------------------------------------------------------- #
def test_sensitivity_specificity_and_error_codes() -> None:
    y_true = ["suspected_opacity", "suspected_opacity", "normal", "normal"]
    y_pred = ["suspected_opacity", "uncertain", "normal", "suspected_opacity"]
    assert sensitivity(y_true, y_pred) == 0.5          # 1 of 2 opacities caught
    assert specificity(y_true, y_pred) == 0.5          # 1 of 2 normals not over-called
    assert classify_error("suspected_opacity", "normal") == "FN"
    assert classify_error("normal", "suspected_opacity") == "FP"
    assert classify_error("normal", "uncertain") == "UA"
    assert classify_error("normal", "normal", json_valid=False) == "JF"


def test_summarize_metrics_has_new_keys() -> None:
    rows = [{"label": "normal", "predicted_class": "normal", "json_valid": True,
             "warning": WARNING_TEXT, "latency_ms": 10}]
    metrics = summarize_metrics(rows)
    assert {"sensitivity", "specificity", "median_latency_ms"} <= set(metrics)
    assert metrics["median_latency_ms"] == 10.0


# --- dataset mapping -------------------------------------------------------- #
def test_rsna_and_chexpert_label_mapping() -> None:
    assert rsna_label(1) == "suspected_opacity"
    assert rsna_label("0") == "normal"
    assert chexpert_label({"Lung Opacity": "1.0"}) == "suspected_opacity"
    assert chexpert_label({"Consolidation": "-1.0"}) == "uncertain"
    assert chexpert_label({"No Finding": "1.0"}) == "normal"


def test_stratified_split_is_disjoint_and_balanced() -> None:
    items = [(f"p{i}", "normal" if i % 2 else "suspected_opacity") for i in range(40)]
    splits = stratified_split(items, n_smoke=4, n_dev=10, n_final=6, seed=1)
    ids = [cid for part in splits.values() for cid, _ in part]
    assert len(ids) == len(set(ids))                   # disjoint
    assert all(splits[name] for name in ("smoke", "dev", "final"))


# --- dispatcher ------------------------------------------------------------- #
def test_predict_toy_backend_default_and_validation() -> None:
    image = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"
    pred = predict(image, mode="baseline")             # default backend == toy
    assert pred["predicted_class"] == "suspected_opacity"
    assert "toy" in BACKENDS


def test_predict_rejects_unknown_backend() -> None:
    image = ROOT / "data" / "sample_images" / "CXR_SYN_001_normal.png"
    try:
        predict(image, backend="does_not_exist")
    except ValueError as exc:
        assert "Unknown backend" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown backend")
