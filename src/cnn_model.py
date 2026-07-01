"""Backend `cnn` : CNN maison (PyTorch) entraîné from scratch — PC GPU/CPU.

Contrairement au backend `classifier` (backbone timm pré-entraîné ImageNet),
ce module définit une **architecture CNN complète écrite à la main** (blocs
convolution → batch-norm → ReLU → max-pooling), entraînée uniquement sur les
données du projet : split `dev` RSNA (réel) + split `smoke` synthétique.
C'est le livrable "modèle de deep learning" : simple, inspectable, reproductible.

Imports `torch` paresseux : rien n'est chargé tant que le backend `cnn` n'est
pas utilisé. Sans poids entraînés, la sortie reste prudente (`uncertain`).
Entraînement : `python finetuning/train_cnn.py`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .guardrails import WARNING_TEXT
from .preprocessing import preprocess

CLASSES = ["normal", "suspected_opacity", "uncertain"]
ROOT = Path(__file__).resolve().parents[1]
# Sortie standard de finetuning/train_cnn.py, utilisée si présente.
TRAINED_CKPT = ROOT / "finetuning" / "outputs" / "cnn_radio.pt"
IMG_SIZE = 224
# Normalisation en niveaux de gris (radiographies) : moyenne/écart-type fixes.
TENSOR_MEAN = 0.5
TENSOR_STD = 0.25

_MODEL_CACHE: dict[str, Any] = {}


def _default_ckpt() -> str:
    """Checkpoint : RADIO_CNN_CKPT prioritaire, sinon la sortie standard."""
    env = os.environ.get("RADIO_CNN_CKPT", "")
    if env:
        return env
    return str(TRAINED_CKPT) if TRAINED_CKPT.exists() else ""


def build_cnn(num_classes: int = len(CLASSES), channels: tuple[int, ...] = (32, 64, 128, 256)):
    """Construit le CNN maison : 4 blocs conv/BN/ReLU/pool puis tête linéaire."""
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - dépend du PC GPU
        raise RuntimeError("Le backend `cnn` requiert torch (voir README_GPU.md).") from exc

    layers: list[Any] = []
    in_ch = 1  # entrée en niveaux de gris (radiographie)
    for out_ch in channels:
        layers += [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        in_ch = out_ch
    return nn.Sequential(
        *layers,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(in_ch, num_classes),
    )


def image_to_tensor(image_path: str | Path):
    """Prétraitement L2 (anonymisation incluse) -> tenseur 1x1x224x224 normalisé."""
    import torch

    image, _meta = preprocess(image_path, size=(IMG_SIZE, IMG_SIZE))
    gray = image.convert("L")
    tensor = torch.frombuffer(bytearray(gray.tobytes()), dtype=torch.uint8).float()
    tensor = tensor.view(1, 1, IMG_SIZE, IMG_SIZE) / 255.0
    return (tensor - TENSOR_MEAN) / TENSOR_STD


def load_cnn(ckpt: str | None = None):
    """Charge (et met en cache) le CNN maison + l'indicateur `entraîné`."""
    import torch

    ckpt = _default_ckpt() if ckpt is None else ckpt
    if ckpt in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt]

    model = build_cnn()
    trained = False
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        trained = True
    model.eval()
    _MODEL_CACHE[ckpt] = (model, trained)
    return model, trained


def cnn_predict(image_path: str | Path) -> dict[str, Any]:
    """Prédit une classe + confiance avec le CNN maison (contrat 7 champs)."""
    import torch

    start = time.perf_counter()
    model, trained = load_cnn()
    tensor = image_to_tensor(image_path)

    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=-1)[0]
    confidence, index = float(probs.max()), int(probs.argmax())
    predicted = CLASSES[index] if trained else "uncertain"
    if not trained:
        # Poids non entraînés : sortie volontairement prudente.
        confidence = min(confidence, 0.50)

    return {
        "image_quality": "limited",
        "predicted_class": predicted,
        "confidence": round(confidence, 3),
        "visual_evidence": [f"score CNN maison = {confidence:.2f}"],
        "justification": (
            "CNN maison entraîné (RSNA dev + synthétique) : classe la plus probable retenue."
            if trained
            else "CNN maison non entraîné (pas de checkpoint) : sortie prudente `uncertain`."
        ),
        "limitations": [
            "CNN pédagogique from scratch, non validé cliniquement",
            "entraîné sur 140 images seulement" if trained else "poids non entraînés",
        ],
        "warning": WARNING_TEXT,
        "model_name": "cnn-radio-scratch",
        "prompt_version": "cnn_v1",
        "latency_ms": int((time.perf_counter() - start) * 1000),
        "class_probabilities": {c: round(float(probs[i]), 4) for i, c in enumerate(CLASSES)},
    }
