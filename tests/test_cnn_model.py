"""Tests du backend `cnn` (CNN maison from scratch, src/cnn_model.py).

`torch` n'est pas installé en CI : les tests qui en dépendent sont sautés
proprement via `pytest.importorskip`. Le contrat JSON reste vérifié partout où
torch est disponible (PC GPU/CPU).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.guardrails import apply_safety_guardrails, validate_prediction

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"


def test_cnn_backend_listed() -> None:
    from src.inference import BACKENDS

    assert "cnn" in BACKENDS


def test_build_cnn_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    from src.cnn_model import IMG_SIZE, build_cnn

    model = build_cnn()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 1, IMG_SIZE, IMG_SIZE))
    assert logits.shape == (2, 3)


def test_cnn_predict_respects_contract() -> None:
    pytest.importorskip("torch")
    from src.cnn_model import cnn_predict

    prediction = apply_safety_guardrails(cnn_predict(SAMPLE))
    ok, errors = validate_prediction(prediction)
    assert ok, errors
    assert prediction["model_name"] == "cnn-radio-scratch"
    assert set(prediction["class_probabilities"]) == {"normal", "suspected_opacity", "uncertain"}


def test_cnn_predict_untrained_is_prudent(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    import src.cnn_model as cnn_model

    # Sans checkpoint, la sortie doit rester `uncertain` avec confiance <= 0.5.
    monkeypatch.setenv("RADIO_CNN_CKPT", str(tmp_path / "absent.pt"))
    monkeypatch.setattr(cnn_model, "_MODEL_CACHE", {})
    prediction = cnn_model.cnn_predict(SAMPLE)
    assert prediction["predicted_class"] == "uncertain"
    assert prediction["confidence"] <= 0.50
