"""Dataset builders for the real chest X-ray datasets used in the brief.

Primary dataset: **RSNA Pneumonia Detection Challenge** (Kaggle). It ships DICOM
images plus ``stage_2_train_labels.csv`` with a binary ``Target`` (1 = lung
opacity / pneumonia-like, 0 = no opacity). We map it to the project labels:

    Target == 1  ->  suspected_opacity
    Target == 0  ->  normal

``uncertain`` is **not** a dataset label: it is produced at inference time by the
confidence threshold and the guard-rails, so it stays a genuine safety class and
is never "learned" from data.

The builder:

1. reads the labels CSV and stratifies a reproducible split
   (``smoke`` / ``dev`` / ``final``),
2. converts each DICOM to an 8-bit PNG (VOI LUT + MONOCHROME1 handling),
3. writes an ImageFolder tree ``images/<split>/<class>/<case_id>.png`` for the
   classifier, and a flat ``cases.csv`` (same columns as the synthetic set) for
   the VLM evaluation harness.

``pydicom`` / ``numpy`` / ``Pillow`` are imported lazily so this module imports
on a minimal runner. Other datasets (CheXpert, MIMIC-CXR, NIH ChestXray) are
documented in ``data/datasets.md``; the mapping helper :func:`chexpert_label`
is provided for the discussion experiments.

CLI
---
    python -m src.datasets build-rsna \
        --src ~/datasets/rsna-pneumonia \
        --out data/rsna --n-dev 150 --n-final 30 --seed 13
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable

RSNA_LABELS_CSV = "stage_2_train_labels.csv"
RSNA_IMAGES_DIR = "stage_2_train_images"
CASE_COLUMNS = ["case_id", "image_path", "source", "label", "split", "quality", "notes"]
LABEL_FROM_TARGET = {"0": "normal", "1": "suspected_opacity"}


# --------------------------------------------------------------------------- #
# Label mapping helpers (pure Python -- unit tested).
# --------------------------------------------------------------------------- #
def rsna_label(target: str | int) -> str:
    """Map an RSNA ``Target`` value to a project class."""
    return LABEL_FROM_TARGET.get(str(target).strip(), "uncertain")


def chexpert_label(row: dict) -> str:
    """Map a CheXpert report row to a project class (for discussion experiments).

    CheXpert uses 1.0 (positive), 0.0 (negative), -1.0 (uncertain) and blanks.
    We collapse opacity-related findings to ``suspected_opacity`` and treat the
    explicit ``-1.0`` uncertainty / ``No Finding`` accordingly.
    """
    opacity_cols = ("Lung Opacity", "Consolidation", "Pneumonia", "Edema", "Atelectasis")
    values = []
    for col in opacity_cols:
        raw = str(row.get(col, "")).strip()
        if raw:
            values.append(raw)
    if any(v == "1.0" or v == "1" for v in values):
        return "suspected_opacity"
    if any(v == "-1.0" or v == "-1" for v in values):
        return "uncertain"
    if str(row.get("No Finding", "")).strip() in {"1.0", "1"}:
        return "normal"
    return "uncertain"


def stratified_split(
    items: list[tuple[str, str]],
    n_smoke: int,
    n_dev: int,
    n_final: int,
    seed: int = 13,
) -> dict[str, list[tuple[str, str]]]:
    """Split ``(case_id, label)`` items into smoke/dev/final, balanced per class.

    Sizes are best-effort: each split draws proportionally from each class and is
    clipped to what is available. Splits are disjoint and reproducible.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[tuple[str, str]]] = {}
    for case_id, label in items:
        by_class.setdefault(label, []).append((case_id, label))
    for bucket in by_class.values():
        rng.shuffle(bucket)

    splits: dict[str, list[tuple[str, str]]] = {"smoke": [], "dev": [], "final": []}
    for size, name in ((n_smoke, "smoke"), (n_dev, "dev"), (n_final, "final")):
        per_class = max(1, size // max(1, len(by_class)))
        for bucket in by_class.values():
            take = bucket[:per_class]
            del bucket[:per_class]
            splits[name].extend(take)
        rng.shuffle(splits[name])
    return splits


# --------------------------------------------------------------------------- #
# DICOM -> PNG (lazy heavy imports).
# --------------------------------------------------------------------------- #
def dicom_to_png(dicom_path, png_path, size: int = 1024) -> None:
    """Convert a DICOM CXR to an 8-bit PNG with basic windowing."""
    import numpy as np
    import pydicom
    from PIL import Image

    try:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except Exception:  # older/newer pydicom layout
        apply_voi_lut = None

    ds = pydicom.dcmread(str(dicom_path))
    pixels = ds.pixel_array
    if apply_voi_lut is not None:
        pixels = apply_voi_lut(pixels, ds)
    pixels = pixels.astype("float32")
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels  # invert so bones are bright

    pmin, pmax = float(pixels.min()), float(pixels.max())
    pixels = (pixels - pmin) / (pmax - pmin + 1e-6) * 255.0
    image = Image.fromarray(pixels.astype("uint8")).convert("L")
    image.thumbnail((size, size))
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)


# --------------------------------------------------------------------------- #
# RSNA builder.
# --------------------------------------------------------------------------- #
def _read_rsna_labels(src: Path) -> list[tuple[str, str]]:
    labels_csv = src / RSNA_LABELS_CSV
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"{labels_csv} not found. Download the RSNA Pneumonia dataset first "
            "(see data/datasets.md).")
    # One patientId can have several bbox rows; deduplicate on max Target.
    target_by_id: dict[str, int] = {}
    with labels_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["patientId"]
            target_by_id[pid] = max(target_by_id.get(pid, 0), int(row["Target"]))
    return [(pid, rsna_label(t)) for pid, t in target_by_id.items()]


def build_rsna(
    src,
    out,
    *,
    n_smoke: int = 20,
    n_dev: int = 150,
    n_final: int = 30,
    seed: int = 13,
) -> Path:
    """Build PNG splits + ``cases.csv`` from a local RSNA dataset directory."""
    src, out = Path(src).expanduser(), Path(out)
    items = _read_rsna_labels(src)
    splits = stratified_split(items, n_smoke, n_dev, n_final, seed=seed)

    rows: list[dict] = []
    repo_root = Path(__file__).resolve().parents[1]
    for split_name, entries in splits.items():
        for case_id, label in entries:
            dicom_path = src / RSNA_IMAGES_DIR / f"{case_id}.dcm"
            if not dicom_path.exists():
                continue
            png_path = out / "images" / split_name / label / f"{case_id}.png"
            dicom_to_png(dicom_path, png_path)
            rel = png_path.relative_to(repo_root) if out.is_absolute() is False else png_path
            rows.append({
                "case_id": case_id,
                "image_path": str(rel),
                "source": "rsna_pneumonia",
                "label": label,
                "split": split_name,
                "quality": "good",
                "notes": "RSNA Pneumonia Detection Challenge (Kaggle)",
            })

    out.mkdir(parents=True, exist_ok=True)
    cases_csv = out / "cases.csv"
    with cases_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} cases to {cases_csv}")
    for name in ("smoke", "dev", "final"):
        count = sum(1 for r in rows if r["split"] == name)
        print(f"  {name}: {count}")
    return cases_csv


def _cli(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build real CXR dataset splits")
    sub = parser.add_subparsers(dest="command", required=True)

    rsna = sub.add_parser("build-rsna", help="convert a local RSNA dataset to PNG + cases.csv")
    rsna.add_argument("--src", required=True, help="RSNA dataset root (contains stage_2_train_*)")
    rsna.add_argument("--out", default="data/rsna")
    rsna.add_argument("--n-smoke", type=int, default=20)
    rsna.add_argument("--n-dev", type=int, default=150)
    rsna.add_argument("--n-final", type=int, default=30)
    rsna.add_argument("--seed", type=int, default=13)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "build-rsna":
        build_rsna(args.src, args.out, n_smoke=args.n_smoke, n_dev=args.n_dev,
                   n_final=args.n_final, seed=args.seed)


if __name__ == "__main__":
    _cli()
