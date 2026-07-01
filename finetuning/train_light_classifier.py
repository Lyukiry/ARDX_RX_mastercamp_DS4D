"""Entraîne le classifieur léger CNN/ViT de support (timm) — PC GPU/CPU.

Produit un checkpoint utilisable par `src/classifier.py` via la variable
`RADIO_CLASSIFIER_CKPT`. Rôle : fournir une classe probable + un score de
confiance qui assiste le VLM (cahier des charges §3.2 / §5.1).

Imports lourds paresseux. Prérequis :
    pip install torch timm pandas pillow
    python finetuning/train_light_classifier.py --csv data/synthetic_cases.csv \
        --out finetuning/outputs/light_classifier.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # accès à src/ quand lancé par `python finetuning/...`
CLASSES = ["normal", "suspected_opacity", "uncertain"]


def read_rows(csv_path: Path, split: str | None) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [r for r in rows if split in (None, "all") or r.get("split") == split]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "synthetic_cases.csv")
    parser.add_argument("--split", default="all")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "finetuning" / "outputs" / "light_classifier.pt")
    args = parser.parse_args()

    # --- Imports lourds paresseux ------------------------------------------
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class CXRDataset(Dataset):
        def __init__(self, rows: list[dict], transform):
            self.rows = rows
            self.transform = transform

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            path = ROOT / row["image_path"]
            if path.suffix.lower() in {".dcm", ".dicom"}:
                # DICOM (ex. RSNA) : conversion + anonymisation via src/preprocessing.
                from src.preprocessing import dicom_to_image
                image = dicom_to_image(path).convert("RGB")
            else:
                image = Image.open(path).convert("RGB")
            return self.transform(image), CLASSES.index(row["label"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(args.backbone, pretrained=True, num_classes=len(CLASSES)).to(device)
    config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**config, is_training=False)

    rows = read_rows(args.csv, args.split)
    loader = DataLoader(CXRDataset(rows, transform), batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total += float(loss)
        print(f"epoch {epoch + 1}/{args.epochs} - loss {total / max(1, len(loader)):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": CLASSES, "backbone": args.backbone}, args.out)
    print(f"Checkpoint sauvegardé : {args.out}")
    print(f"Utilisation : RADIO_CLASSIFIER_CKPT={args.out} RADIO_BACKEND=classifier")
    print("Rappel : jeu synthétique = validation logicielle, pas de performance médicale.")


if __name__ == "__main__":
    main()
