"""Préparation déterministe d'un sous-ensemble RSNA Pneumonia → `data/rsna_cases.csv`.

Le dataset RSNA Pneumonia (Kaggle) est **non redistribuable** : ce script ne
copie dans le dépôt que des DICOM dé-identifiés dans `data/rsna/` (dossier
ignoré par git) et produit un CSV au même format que `synthetic_cases.csv`
(colonnes `case_id, image_path, source, label, split, quality, notes`).

Téléchargement préalable (compte Kaggle + acceptation des règles) :
    kaggle competitions download -c rsna-pneumonia-detection-challenge -p ~/datasets/rsna

Mapping (docs/guide_execution_gpu.md §6) :
- `Target=1` → `suspected_opacity`
- `Target=0` → `normal` (on privilégie la sous-classe "Normal" de
  `stage_2_detailed_class_info.csv` pour une vérité terrain plus propre ;
  la classe `uncertain` reste produite par le seuil de confiance, jamais un label).

Découpage (docs/evaluation_protocol.md) :
- `dev`   : 120 cas équilibrés (mise au point / classifieur léger / LoRA).
- `final` : 30 cas équilibrés (mesures et analyse d'erreurs).

Usage :
    python data/make_rsna_cases.py [--zip ~/datasets/rsna/rsna-pneumonia-detection-challenge.zip]
"""
from __future__ import annotations

import argparse
import csv
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = Path.home() / "datasets" / "rsna" / "rsna-pneumonia-detection-challenge.zip"
IMAGES_DIR = ROOT / "data" / "rsna" / "images"
CSV_PATH = ROOT / "data" / "rsna_cases.csv"

SEED = 20260701
N_DEV = 120
N_FINAL = 30

LABEL_BY_TARGET = {"1": "suspected_opacity", "0": "normal"}


def read_zip_csv(zf: zipfile.ZipFile, basename: str) -> list[dict]:
    """Lit un CSV membre du zip Kaggle (recherché par nom de fichier)."""
    member = next((n for n in zf.namelist() if n.endswith(basename)), None)
    if member is None:
        raise SystemExit(f"Membre introuvable dans le zip : {basename}")
    with zf.open(member) as handle:
        text = handle.read().decode("utf-8").splitlines()
    return list(csv.DictReader(text))


def build_patient_table(zf: zipfile.ZipFile) -> dict[str, dict]:
    """patientId → {label, detail} dédupliqué (plusieurs bbox par patient positif)."""
    labels = read_zip_csv(zf, "stage_2_train_labels.csv")
    details = read_zip_csv(zf, "stage_2_detailed_class_info.csv")
    detail_by_pid = {row["patientId"]: row["class"] for row in details}

    patients: dict[str, dict] = {}
    for row in labels:
        pid = row["patientId"]
        if pid not in patients:
            patients[pid] = {
                "label": LABEL_BY_TARGET[row["Target"]],
                "detail": detail_by_pid.get(pid, ""),
            }
    return patients


def pick_cases(patients: dict[str, dict]) -> list[dict]:
    """Sélection équilibrée et déterministe : dev (120) puis final (30)."""
    positives = sorted(p for p, m in patients.items() if m["label"] == "suspected_opacity")
    # Vérité terrain plus propre : uniquement la sous-classe "Normal" (Target=0).
    normals = sorted(p for p, m in patients.items() if m["detail"] == "Normal")

    rng = random.Random(SEED)
    rng.shuffle(positives)
    rng.shuffle(normals)

    cases = []
    for split, count, offset in (("dev", N_DEV, 0), ("final", N_FINAL, N_DEV)):
        half = count // 2
        start = offset // 2
        for pid in positives[start:start + half]:
            cases.append({"pid": pid, "split": split, **patients[pid]})
        for pid in normals[start:start + count - half]:
            cases.append({"pid": pid, "split": split, **patients[pid]})
    return cases


def extract_images(zf: zipfile.ZipFile, cases: list[dict]) -> None:
    """Extrait uniquement les DICOM sélectionnés vers data/rsna/images/."""
    members = {Path(n).stem: n for n in zf.namelist()
               if "stage_2_train_images" in n and n.endswith(".dcm")}
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for case in cases:
        member = members.get(case["pid"])
        if member is None:
            raise SystemExit(f"DICOM absent du zip pour le patient {case['pid']}")
        target = IMAGES_DIR / f"{case['pid']}.dcm"
        if not target.exists():
            target.write_bytes(zf.read(member))


def write_csv(cases: list[dict]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "image_path", "source", "label", "split", "quality", "notes"])
        for case in sorted(cases, key=lambda c: (c["split"], c["label"], c["pid"])):
            writer.writerow([
                f"RSNA_{case['pid']}",
                f"data/rsna/images/{case['pid']}.dcm",
                "rsna_pneumonia_kaggle",
                case["label"],
                case["split"],
                "good",
                f"RSNA dé-identifié ; classe détaillée : {case['detail']}",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP,
                        help="Zip Kaggle rsna-pneumonia-detection-challenge")
    args = parser.parse_args()

    if not args.zip.exists():
        raise SystemExit(
            f"Zip introuvable : {args.zip}\n"
            "Télécharger d'abord (compte Kaggle + règles acceptées) :\n"
            f"  kaggle competitions download -c rsna-pneumonia-detection-challenge -p {args.zip.parent}"
        )

    with zipfile.ZipFile(args.zip) as zf:
        patients = build_patient_table(zf)
        cases = pick_cases(patients)
        extract_images(zf, cases)
    write_csv(cases)

    by_split: dict[str, int] = {}
    for case in cases:
        by_split[case["split"]] = by_split.get(case["split"], 0) + 1
    print(f"{len(cases)} cas écrits dans {CSV_PATH} ({by_split})")
    print(f"DICOM extraits dans {IMAGES_DIR} (dossier ignoré par git, non redistribuable)")


if __name__ == "__main__":
    main()
