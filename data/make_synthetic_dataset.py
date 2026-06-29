"""Générateur déterministe du jeu synthétique jouet.

Ce script (re)génère les images "radiographie-like" et le fichier
`synthetic_cases.csv`. Les images ne sont PAS médicales : elles imitent
grossièrement un thorax frontal uniquement pour valider la chaîne logicielle
(chargement, prétraitement, inférence, JSON, garde-fous, logs, métriques, UI).

Convention de nommage imposée par les tests : `CXR_SYN_<NNN>_<label>.png`,
avec label dans {normal, suspected_opacity, uncertain}. Le cas CXR_SYN_002
reste un `suspected_opacity` (utilisé par les tests de l'API et du schéma).

Découpage (cahier des charges §4.2) :
- 20 cas `smoke`  : valider que la chaîne tourne.
- 30 cas `final`  : 30 cas finaux commentés pour l'évaluation et l'analyse d'erreurs.

Usage :
    python data/make_synthetic_dataset.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "data" / "sample_images"
CSV_PATH = ROOT / "data" / "synthetic_cases.csv"

# Taxonomie gelée (clés anglaises stables, voir docs/note_de_cadrage.md).
LABELS = ["normal", "suspected_opacity", "uncertain"]
SIZE = (512, 512)
BASE_SEED = 20260622
N_SMOKE = 20
N_FINAL = 30
N_TOTAL = N_SMOKE + N_FINAL

# Notes francophones courtes affichées dans l'onglet "Cas" de la démo.
NOTES = {
    "normal": "image synthétique d'aspect normal",
    "suspected_opacity": "opacité synthétique localisée dans un champ pulmonaire",
    "uncertain": "image synthétique ambiguë (qualité dégradée)",
}

# Qualité forcée à "poor"/"limited" sur quelques cas non-uncertain pour rendre
# l'analyse d'erreurs réaliste (faux négatifs / faux positifs liés à la qualité).
FORCED_POOR = {26, 33, 44}
FORCED_LIMITED = {29, 38, 47}


def quality_for(index: int, label: str) -> str:
    """Qualité d'image déterministe pour le cas n°`index`."""
    if label == "uncertain":
        return "poor" if index % 2 == 0 else "limited"
    if index in FORCED_POOR:
        return "poor"
    if index in FORCED_LIMITED:
        return "limited"
    return "good"


def _draw_thorax(draw: ImageDraw.ImageDraw) -> None:
    """Dessine un thorax frontal stylisé (fond, champs pulmonaires, médiastin)."""
    draw.rectangle([0, 0, SIZE[0], SIZE[1]], fill=18)  # fond radio sombre
    # Champs pulmonaires : deux ellipses plus claires.
    draw.ellipse([70, 120, 240, 430], fill=70)   # poumon droit
    draw.ellipse([272, 120, 442, 430], fill=70)  # poumon gauche
    # Médiastin / colonne : bande verticale centrale plus dense.
    draw.rectangle([238, 110, 274, 450], fill=120)
    # Coupole diaphragmatique : base plus dense.
    draw.ellipse([60, 360, 452, 520], fill=95)
    # Côtes : arcs faibles pour le réalisme visuel.
    for offset in range(150, 380, 38):
        draw.arc([70, offset - 40, 240, offset + 40], 200, 340, fill=110, width=2)
        draw.arc([272, offset - 40, 442, offset + 40], 200, 340, fill=110, width=2)


def _add_opacity(image: Image.Image, rng: random.Random) -> Image.Image:
    """Ajoute une opacité floue (tache claire) dans un champ pulmonaire."""
    overlay = Image.new("L", SIZE, 0)
    odraw = ImageDraw.Draw(overlay)
    left_lung = rng.random() < 0.5
    cx = rng.randint(110, 200) if left_lung else rng.randint(300, 390)
    cy = rng.randint(220, 360)
    radius = rng.randint(35, 65)
    odraw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=180)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=radius / 2))
    return Image.composite(Image.new("L", SIZE, 200), image, overlay)


def _degrade(image: Image.Image, quality: str, rng: random.Random) -> Image.Image:
    """Dégrade le contraste et ajoute du bruit selon la qualité visée."""
    if quality == "good":
        sigma, blur = 8, 0.4
    elif quality == "limited":
        sigma, blur = 26, 1.2
    else:  # poor : sous-exposition + bruit fort + flou marqué
        sigma, blur = 48, 2.4
    noise = Image.effect_noise(SIZE, sigma)
    image = Image.blend(image, noise, alpha=0.18 if quality == "good" else 0.34)
    image = image.filter(ImageFilter.GaussianBlur(radius=blur))
    if quality == "poor":
        # Réduit la dynamique (mauvaise exposition).
        image = image.point(lambda p: int(40 + p * 0.55))
    return image


def make_image(index: int, label: str, quality: str) -> Image.Image:
    """Construit une image jouet déterministe pour un cas donné."""
    rng = random.Random(BASE_SEED + index)
    image = Image.new("L", SIZE, 18)
    _draw_thorax(ImageDraw.Draw(image))
    if label == "suspected_opacity":
        image = _add_opacity(image, rng)
    image = _degrade(image, quality, rng)
    return image.convert("RGB")


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index in range(1, N_TOTAL + 1):
        label = LABELS[(index - 1) % 3]
        quality = quality_for(index, label)
        split = "smoke" if index <= N_SMOKE else "final"
        filename = f"CXR_SYN_{index:03d}_{label}.png"
        rel_path = f"data/sample_images/{filename}"
        make_image(index, label, quality).save(IMAGES_DIR / filename)
        rows.append(
            {
                "case_id": f"CXR_SYN_{index:03d}",
                "image_path": rel_path,
                "source": "synthetic_toy",
                "label": label,
                "split": split,
                "quality": quality,
                "notes": NOTES[label],
            }
        )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case_id", "image_path", "source", "label", "split", "quality", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    smoke = sum(r["split"] == "smoke" for r in rows)
    final = sum(r["split"] == "final" for r in rows)
    print(f"Généré {len(rows)} cas synthétiques ({smoke} smoke + {final} final) dans {IMAGES_DIR}")


if __name__ == "__main__":
    main()
