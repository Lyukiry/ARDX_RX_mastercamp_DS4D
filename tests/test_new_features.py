from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from src.guardrails import (
    UNCERTAINTY_WARNING_TEXT,
    WARNING_TEXT,
    apply_safety_guardrails,
    needs_uncertainty_warning,
    validate_prediction,
)
from src.inference import predict
from src.metrics import accuracy, sensitivity, specificity
from src.preprocessing import DEFAULT_SIZE, anonymize_image, normalize_image
from src.synthetic_eval import MODES, noisy_predict

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"


def _valid_pred(**overrides) -> dict:
    base = {
        "image_quality": "good",
        "predicted_class": "normal",
        "confidence": 0.85,
        "visual_evidence": ["clear lung fields"],
        "justification": "Clear fields.",
        "limitations": ["not a validated medical model"],
        "warning": WARNING_TEXT,
    }
    base.update(overrides)
    return base


# ----------------------------------------------------------------- §7.2 warning
def test_uncertainty_warning_triggers_on_low_confidence() -> None:
    pred = apply_safety_guardrails(_valid_pred(confidence=0.4))
    assert pred["uncertainty_warning"] == UNCERTAINTY_WARNING_TEXT


def test_uncertainty_warning_triggers_on_poor_quality() -> None:
    assert needs_uncertainty_warning(_valid_pred(confidence=0.9, image_quality="poor"))


def test_no_uncertainty_warning_when_confident_and_good() -> None:
    pred = apply_safety_guardrails(_valid_pred(confidence=0.85, image_quality="good"))
    assert pred["uncertainty_warning"] == ""
    assert pred["warning"] == WARNING_TEXT  # disclaimer inconditionnel toujours présent


def test_validator_rejects_non_list_evidence() -> None:
    valid, errors = validate_prediction(_valid_pred(visual_evidence="not a list"))
    assert not valid and any("visual_evidence" in e for e in errors)


# ------------------------------------------------------------- préprocessing L2
def test_anonymize_strips_metadata_keeps_pixels() -> None:
    image = Image.new("RGB", (8, 8), (120, 10, 5))
    image.info["comment"] = "patient John Doe"
    clean = anonymize_image(image)
    assert clean.info == {}
    assert list(clean.getdata()) == list(image.getdata())


def test_normalize_image_returns_rgb_default_size() -> None:
    out = normalize_image(Image.open(SAMPLE))
    assert out.mode == "RGB" and out.size == DEFAULT_SIZE


# --------------------------------------------------------------------- métriques
def test_sensitivity_and_specificity() -> None:
    y_true = ["suspected_opacity", "suspected_opacity", "normal", "normal"]
    y_pred = ["suspected_opacity", "normal", "normal", "suspected_opacity"]
    assert sensitivity(y_true, y_pred) == 0.5
    assert specificity(y_true, y_pred) == 0.5


# --------------------------------------------------- backend noisy + dispatcher
def test_noisy_predict_is_deterministic_and_valid() -> None:
    case = {"case_id": "CXR_SYN_023", "label": "suspected_opacity", "quality": "good"}
    first = noisy_predict(case, mode="baseline", seed=1)
    second = noisy_predict(case, mode="baseline", seed=1)
    assert first["predicted_class"] == second["predicted_class"]
    assert validate_prediction(first)[0]
    assert set(MODES) == {"baseline", "improved", "structured"}


def test_dispatcher_backends_and_unknown() -> None:
    assert validate_prediction(predict(SAMPLE, backend="toy"))[0]
    assert validate_prediction(predict(SAMPLE, backend="noisy"))[0]
    with pytest.raises(ValueError):
        predict(SAMPLE, backend="does-not-exist")


def test_improved_beats_baseline_on_final_split() -> None:
    with (ROOT / "data" / "synthetic_cases.csv").open(encoding="utf-8") as f:
        cases = [r for r in csv.DictReader(f) if r["split"] == "final"]
    y_true = [c["label"] for c in cases]
    base = accuracy(y_true, [noisy_predict(c, mode="baseline")["predicted_class"] for c in cases])
    imp = accuracy(y_true, [noisy_predict(c, mode="improved")["predicted_class"] for c in cases])
    assert imp > base


# --------------------------------------------------------- livrables structurels
def test_three_prompts_present() -> None:
    for name in ("baseline_prompt.txt", "improved_prompt.txt", "structured_prompt.txt"):
        assert (ROOT / "prompts" / name).exists()


def test_final_split_has_30_cases() -> None:
    with (ROOT / "data" / "synthetic_cases.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert sum(r["split"] == "final" for r in rows) == 30
    assert sum(r["split"] == "smoke" for r in rows) == 20
