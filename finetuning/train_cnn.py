"""Entraîne le CNN maison (`src/cnn_model.py`) — PC GPU/CPU.

Contrairement à `train_light_classifier.py` (backbone timm pré-entraîné), ce
script entraîne **from scratch** l'architecture définie dans `src/cnn_model.py`.
Jeu d'entraînement par défaut : split `dev` RSNA (réel, 120 cas) + split
`smoke` synthétique (20 cas, apporte des exemples `uncertain`). L'évaluation se
fait ensuite sur les splits `final` (jamais vus) : onglet « CNN » de Streamlit.

Usage :
    python finetuning/train_cnn.py
    python finetuning/train_cnn.py --data data/rsna_cases.csv:dev \
        --data data/synthetic_cases.csv:smoke --epochs 15 \
        --out finetuning/outputs/cnn_radio.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # accès à src/ quand lancé par `python finetuning/...`
CLASSES = ["normal", "suspected_opacity", "uncertain"]
DEFAULT_DATA = ["data/rsna_cases.csv:dev", "data/synthetic_cases.csv:smoke"]


def read_rows(spec: str) -> list[dict]:
    """Lit un couple `chemin.csv:split` (`:split` optionnel = tous les cas)."""
    csv_part, _, split = spec.rpartition(":")
    if not csv_part or len(split) > 1 and Path(spec).suffix == ".csv":
        csv_part, split = spec, "all"
    with (ROOT / csv_part).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [r for r in rows if split == "all" or r.get("split") == split]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", action="append", metavar="CSV[:SPLIT]",
                        help="jeu d'entraînement, répétable (défaut : RSNA dev + synthétique smoke)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--out", type=Path, default=ROOT / "finetuning" / "outputs" / "cnn_radio.pt")
    args = parser.parse_args()

    # --- Imports lourds paresseux ------------------------------------------
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    from src.cnn_model import IMG_SIZE, TENSOR_MEAN, TENSOR_STD, build_cnn
    from src.preprocessing import preprocess

    torch.manual_seed(args.seed)

    rows = [r for spec in (args.data or DEFAULT_DATA) for r in read_rows(spec)]
    if not rows:
        raise SystemExit("Aucun cas d'entraînement (vérifier --data et les splits).")

    class CXRDataset(Dataset):
        """Images préchargées une fois (anonymisation L2 incluse), augmentation à la volée."""

        def __init__(self, rows: list[dict], transform):
            self.items = []
            for row in rows:
                image, _meta = preprocess(ROOT / row["image_path"], size=(IMG_SIZE, IMG_SIZE))
                self.items.append((image.convert("L"), CLASSES.index(row["label"])))
            self.transform = transform

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):
            image, target = self.items[index]
            return self.transform(image), target

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([TENSOR_MEAN], [TENSOR_STD]),
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_cnn().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Pondération des classes : `uncertain` est rare (absent des labels RSNA).
    counts = {c: max(1, sum(r["label"] == c for r in rows)) for c in CLASSES}
    weights = torch.tensor([len(rows) / (len(CLASSES) * counts[c]) for c in CLASSES]).to(device)

    print(f"CNN maison : {n_params:,} paramètres | {len(rows)} cas "
          f"({ {c: sum(r['label'] == c for r in rows) for c in CLASSES} }) | device={device}")

    loader = DataLoader(CXRDataset(rows, train_transform), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    model.train()
    for epoch in range(args.epochs):
        total, correct, seen = 0.0, 0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total += float(loss)
            correct += int((logits.argmax(dim=-1) == targets).sum())
            seen += len(targets)
        print(f"epoch {epoch + 1}/{args.epochs} - loss {total / max(1, len(loader)):.4f} "
              f"- acc train {correct / max(1, seen):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": CLASSES,
                "arch": "radio_cnn_4blocks", "img_size": IMG_SIZE, "n_params": n_params}, args.out)
    print(f"Checkpoint sauvegardé : {args.out}")
    print("Utilisation : backend `cnn` (auto-détecté), onglet « CNN » de Streamlit pour l'évaluation.")
    print("Rappel : prototype pédagogique, aucune validité médicale.")


if __name__ == "__main__":
    main()
