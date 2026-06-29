"""Backend classifieur léger CNN/ViT (support de confiance) — PC GPU/CPU.

Rôle (cahier des charges §3.2 / §5.1) : fournir une classe probable + un score
de confiance qui *assiste* le VLM. Imports `torch`/`timm` paresseux : rien n'est
chargé tant que le backend `classifier` n'est pas utilisé.

Sans poids entraînés, `classifier_predict` reste prudent et renvoie `uncertain`
avec une confiance faible. Entraînement : `finetuning/train_light_classifier.py`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .guardrails import WARNING_TEXT

CLASSES = ["normal", "suspected_opacity", "uncertain"]
DEFAULT_BACKBONE = os.environ.get("RADIO_CLASSIFIER_BACKBONE", "resnet18")
DEFAULT_CKPT = os.environ.get("RADIO_CLASSIFIER_CKPT", "")
IMG_SIZE = 224

_MODEL_CACHE: dict[str, Any] = {}


def build_model(backbone: str = DEFAULT_BACKBONE, num_classes: int = len(CLASSES)):
    """Construit un backbone léger (timm) avec une tête à 3 classes."""
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - dépend du PC GPU
        raise RuntimeError("Le backend `classifier` requiert timm + torch.") from exc
    return timm.create_model(backbone, pretrained=True, num_classes=num_classes)


def load_classifier(backbone: str = DEFAULT_BACKBONE, ckpt: str = DEFAULT_CKPT):
    """Charge (et met en cache) le classifieur et ses transforms."""
    cache_key = f"{backbone}:{ckpt}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    try:
        import timm
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Le backend `classifier` requiert timm + torch.") from exc

    model = build_model(backbone)
    trained = False
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        trained = True
    model.eval()
    config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**config, is_training=False)
    _MODEL_CACHE[cache_key] = (model, transform, trained)
    return model, transform, trained


def classifier_predict(image_path: str | Path) -> dict[str, Any]:
    """Prédit une classe + confiance avec le classifieur léger."""
    import torch

    from .preprocessing import preprocess

    start = time.perf_counter()
    model, transform, trained = load_classifier()
    image, _meta = preprocess(image_path, size=(IMG_SIZE, IMG_SIZE))
    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0]
    confidence, index = float(probs.max()), int(probs.argmax())
    predicted = CLASSES[index] if trained else "uncertain"

    if not trained:
        # Poids non entraînés : sortie volontairement prudente.
        confidence = min(confidence, 0.50)

    return {
        "image_quality": "limited",
        "predicted_class": predicted,
        "confidence": round(confidence, 3),
        "visual_evidence": [f"score classifieur {DEFAULT_BACKBONE} = {confidence:.2f}"],
        "justification": (
            "Classifieur léger entraîné : classe la plus probable retenue."
            if trained
            else "Classifieur non entraîné (pas de checkpoint) : sortie prudente `uncertain`."
        ),
        "limitations": [
            "classifieur de support, non validé cliniquement",
            "confiance non calibrée" if trained else "poids non entraînés",
        ],
        "warning": WARNING_TEXT,
        "model_name": f"classifier-{DEFAULT_BACKBONE}",
        "prompt_version": "classifier_v1",
        "latency_ms": int((time.perf_counter() - start) * 1000),
        "class_probabilities": {c: round(float(probs[i]), 4) for i, c in enumerate(CLASSES)},
    }
