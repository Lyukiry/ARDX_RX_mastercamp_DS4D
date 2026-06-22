"""Light CNN / ViT classifier backend (confidence support).

Role in the project (see brief, section "Quels modèles utiliser ?"): a small
specialised classifier that gives a *probable class + score*. It is cheaper to
run and easier to calibrate than a VLM, so it is used as a confidence signal and
as an alternative baseline. Predictions below a confidence threshold are mapped
to ``uncertain`` -- the project's safety class.

Everything heavy (``torch``, ``torchvision``) is imported lazily so the module
imports fine on a torch-free CI runner. Train on a GPU machine, then reuse the
checkpoint for fast CPU/MPS inference.

CLI
---
    python -m src.classifier train --data-dir data/rsna/images --out models/cls.pt
    python -m src.classifier predict --checkpoint models/cls.pt --image chest.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .vlm_inference import coerce_to_schema

# The classifier is trained on the two evidence classes; "uncertain" is produced
# by the confidence threshold, not learned, so it stays a genuine guard-rail.
CLASSIFIER_CLASSES = ["normal", "suspected_opacity"]
DEFAULT_THRESHOLD = 0.60
DEFAULT_BACKBONE = "resnet18"
IMG_SIZE = 224


def _build_transform(train: bool):
    from torchvision import transforms

    aug = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        aug += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(7)]
    aug += [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(aug)


def _build_backbone(backbone: str, n_classes: int):
    import torch.nn as nn
    from torchvision import models

    factory = getattr(models, backbone)
    net = factory(weights="DEFAULT")
    if hasattr(net, "fc"):  # resnet family
        net.fc = nn.Linear(net.fc.in_features, n_classes)
    elif hasattr(net, "classifier"):  # vit / convnext / efficientnet
        head = net.classifier
        in_features = head[-1].in_features if isinstance(head, nn.Sequential) else head.in_features
        if isinstance(head, nn.Sequential):
            head[-1] = nn.Linear(in_features, n_classes)
        else:
            net.classifier = nn.Linear(in_features, n_classes)
    elif hasattr(net, "heads"):  # torchvision ViT
        net.heads.head = nn.Linear(net.heads.head.in_features, n_classes)
    else:
        raise ValueError(f"Unsupported backbone head for {backbone}")
    return net


class LightClassifier:
    """Small image classifier with an ``uncertain`` confidence threshold."""

    def __init__(
        self,
        classes: list[str] | None = None,
        backbone: str = DEFAULT_BACKBONE,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | None = None,
    ) -> None:
        self.classes = classes or list(CLASSIFIER_CLASSES)
        self.backbone = backbone
        self.threshold = threshold
        self._device = device
        self._model = None

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # -- training ------------------------------------------------------------ #
    def train(self, data_dir, *, epochs: int = 5, batch_size: int = 16, lr: float = 1e-4):
        """Fine-tune the backbone on an ImageFolder dataset (one subdir per class)."""
        import torch
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder

        device = self._resolve_device()
        dataset = ImageFolder(str(data_dir), transform=_build_transform(train=True))
        self.classes = list(dataset.classes)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = _build_backbone(self.backbone, len(self.classes)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            running = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                running += loss.item() * images.size(0)
            print(f"epoch {epoch + 1}/{epochs} - loss {running / len(dataset):.4f}")

        self._model = model
        return self

    def save(self, path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._model.state_dict(),
                    "classes": self.classes, "backbone": self.backbone}, path)
        path.with_suffix(".json").write_text(
            json.dumps({"classes": self.classes, "backbone": self.backbone,
                        "threshold": self.threshold}, indent=2), encoding="utf-8")

    def load(self, checkpoint) -> "LightClassifier":
        import torch

        ckpt = torch.load(checkpoint, map_location="cpu")
        self.classes = ckpt["classes"]
        self.backbone = ckpt.get("backbone", self.backbone)
        device = self._resolve_device()
        model = _build_backbone(self.backbone, len(self.classes))
        model.load_state_dict(ckpt["state_dict"])
        self._model = model.to(device).eval()
        return self

    # -- inference ----------------------------------------------------------- #
    def predict(self, image_path, mode: str = "classifier") -> dict[str, Any]:
        import torch
        from PIL import Image

        if self._model is None:
            raise RuntimeError("Classifier not loaded. Call .train() or .load() first.")
        device = self._resolve_device()
        start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        tensor = _build_transform(train=False)(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            probs = torch.softmax(self._model(tensor), dim=-1).squeeze(0).cpu().tolist()

        top_idx = max(range(len(probs)), key=probs.__getitem__)
        top_class = self.classes[top_idx]
        confidence = float(probs[top_idx])
        ranking = ", ".join(f"{c}={p:.2f}" for c, p in zip(self.classes, probs))

        raw = {
            "image_quality": "good",
            "predicted_class": top_class if confidence >= self.threshold else "uncertain",
            "confidence": confidence,
            "visual_evidence": [f"classifier probabilities: {ranking}"],
            "justification": (
                f"Light {self.backbone} classifier scored {top_class} at {confidence:.2f}. "
                "This is a statistical signal, not a clinical interpretation."),
            "limitations": ["confidence not formally calibrated", "trained on a narrow label set"],
        }
        latency_ms = int((time.perf_counter() - start) * 1000)
        return coerce_to_schema(
            raw, model_name=f"classifier-{self.backbone}",
            prompt_version="classifier_v1", latency_ms=latency_ms)


_CLASSIFIER: LightClassifier | None = None


def get_classifier(checkpoint) -> LightClassifier:
    """Return a process-wide cached classifier loaded from ``checkpoint``."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = LightClassifier().load(checkpoint)
    return _CLASSIFIER


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Light CXR classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train on an ImageFolder dataset")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--out", default="models/classifier.pt")
    train.add_argument("--backbone", default=DEFAULT_BACKBONE)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=16)

    pred = sub.add_parser("predict", help="predict a single image")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--image", required=True)

    args = parser.parse_args()
    if args.command == "train":
        clf = LightClassifier(backbone=args.backbone)
        clf.train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
        clf.save(args.out)
        print(f"saved checkpoint to {args.out}")
    elif args.command == "predict":
        clf = LightClassifier().load(args.checkpoint)
        print(json.dumps(clf.predict(args.image), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
