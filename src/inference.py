from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from .preprocessing import basic_quality_flag

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Available inference backends. ``toy`` is the default so the repository runs and
# the CI stays torch-free. ``vlm`` and ``classifier`` load real models lazily.
BACKENDS = ("toy", "vlm", "classifier")


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Deterministic toy predictor used to validate the repo pipeline.

    It reads synthetic labels from filenames. This is not medical inference.
    """
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    quality = basic_quality_flag(image_path)

    if "suspected_opacity" in name:
        pred = "suspected_opacity"
        conf = 0.78 if mode == "baseline" else 0.72
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = "The synthetic image contains a localized brighter region compatible with the toy opacity class. This is a pipeline validation result, not a medical interpretation."
    elif "normal" in name:
        pred = "normal"
        conf = 0.72 if mode == "baseline" else 0.68
        evidence = ["no synthetic opacity marker detected"]
        justification = "The synthetic image does not contain the opacity marker used by the toy generator. This conclusion is limited to the synthetic validation setting."
    else:
        pred = "uncertain"
        conf = 0.52
        evidence = ["limited synthetic image quality"]
        justification = "The image is treated as limited quality in the toy catalog. The safe output is uncertainty rather than a forced class."

    # Improved mode is more conservative.
    if mode == "improved" and quality != "good":
        pred = "uncertain"
        conf = min(conf, 0.55)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": pred,
        "confidence": round(float(conf), 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["synthetic toy image", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
        "model_name": f"toy-rule-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def vlm_predict_placeholder(image_path: str | Path, prompt: str) -> dict[str, Any]:
    """Deprecated placeholder kept for backward compatibility.

    The real VLM call now lives in :mod:`src.vlm_inference`. Use
    ``predict(image_path, mode, backend="vlm")`` instead.
    """
    return toy_predict(image_path, mode="baseline")


def predict(
    image_path: str | Path,
    mode: str = "baseline",
    backend: str | None = None,
) -> dict[str, Any]:
    """Dispatch to the selected inference backend, keeping one output schema.

    Parameters
    ----------
    image_path:
        Path to the chest X-ray image.
    mode:
        ``baseline`` or ``improved`` (drives prompt / uncertainty rule).
    backend:
        ``toy`` (default), ``vlm`` (MedGemma / Gemma via Hugging Face) or
        ``classifier`` (light CNN/ViT). Falls back to the ``RADIO_BACKEND``
        environment variable, then to ``toy``.

    The real backends import torch/transformers lazily, so a missing GPU stack
    raises only when ``vlm``/``classifier`` is actually requested.
    """
    backend = (backend or os.environ.get("RADIO_BACKEND", "toy")).lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {BACKENDS}.")

    if backend == "toy":
        return toy_predict(image_path, mode=mode)
    if backend == "vlm":
        from .vlm_inference import vlm_predict

        return vlm_predict(image_path, mode=mode)
    # classifier
    from .classifier import get_classifier

    checkpoint = os.environ.get("RADIO_CLASSIFIER_CKPT")
    if not checkpoint:
        raise RuntimeError("Set RADIO_CLASSIFIER_CKPT to a trained classifier checkpoint.")
    return get_classifier(checkpoint).predict(image_path, mode=mode)
