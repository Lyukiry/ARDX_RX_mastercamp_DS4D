from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".dcm", ".dicom"}
DEFAULT_SIZE = (512, 512)

# Balises DICOM porteuses de données identifiantes à neutraliser (anonymisation).
# Référence : champs PHI usuels d'un en-tête DICOM.
DICOM_PHI_TAGS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "InstitutionName",
    "InstitutionAddress",
    "StationName",
    "AccessionNumber",
    "StudyID",
    "DeviceSerialNumber",
)


def load_image(path: str | Path, size: tuple[int, int] | None = None) -> Image.Image:
    """Charge une image de façon sûre pour le prototype pédagogique.

    Les fichiers DICOM sont délégués à `dicom_to_image` (anonymisé). Le
    redimensionnement reste optionnel pour ne pas écraser le pipeline complet.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Format d'image non supporté : {path.suffix}")
    if suffix in {".dcm", ".dicom"}:
        image = dicom_to_image(path)
    else:
        image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size)
    return image


def anonymize_image(image: Image.Image) -> Image.Image:
    """Renvoie une copie sans métadonnées (EXIF, commentaires, info PIL).

    On reconstruit l'image à partir des seuls pixels : le dictionnaire `info`
    (qui peut contenir EXIF, GPS, annotations) est ainsi abandonné.
    """
    clean = Image.new(image.mode, image.size)
    clean.putdata(list(image.getdata()))
    return clean


def normalize_image(
    image: Image.Image, size: tuple[int, int] = DEFAULT_SIZE, equalize: bool = False
) -> Image.Image:
    """Normalise l'intensité et la taille d'une radiographie jouet.

    Étapes : niveaux de gris -> normalisation de contraste (autocontrast, ou
    égalisation d'histogramme si `equalize`) -> redimensionnement -> RGB.
    """
    gray = ImageOps.grayscale(image)
    gray = ImageOps.equalize(gray) if equalize else ImageOps.autocontrast(gray)
    gray = gray.resize(size)
    return gray.convert("RGB")


def preprocess(
    path: str | Path, size: tuple[int, int] = DEFAULT_SIZE, equalize: bool = False
) -> tuple[Image.Image, dict[str, Any]]:
    """Pipeline complet L2 : chargement -> anonymisation -> normalisation.

    Renvoie l'image prête pour l'inférence et un dictionnaire de métadonnées
    techniques (jamais de données patient) traçable dans les logs.
    """
    raw = load_image(path)
    anonymized = anonymize_image(raw)
    normalized = normalize_image(anonymized, size=size, equalize=equalize)
    meta = {
        "source_path": str(path),
        "source_format": Path(path).suffix.lower().lstrip("."),
        "original_size": raw.size,
        "processed_size": normalized.size,
        "anonymized": True,
        "equalized": equalize,
    }
    return normalized, meta


def dicom_to_image(path: str | Path) -> Image.Image:
    """Convertit un DICOM en image RGB en supprimant les balises PHI.

    `pydicom` est importé paresseusement : aucune dépendance lourde n'est
    requise pour le chemin synthétique ni pour la CI.
    """
    try:
        import pydicom  # import paresseux : non requis pour le mode jouet
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement GPU/clinique
        raise RuntimeError(
            "pydicom est requis pour lire un DICOM (pip install pydicom)."
        ) from exc

    dataset = pydicom.dcmread(str(path))
    _strip_dicom_phi(dataset)
    pixels = apply_voi_lut(dataset.pixel_array, dataset)
    # Normalisation min-max vers 0-255 pour l'affichage.
    pixels = pixels.astype("float32")
    span = pixels.max() - pixels.min()
    if span > 0:
        pixels = (pixels - pixels.min()) / span * 255.0
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 255.0 - pixels  # inversion des radios MONOCHROME1
    return Image.fromarray(pixels.astype("uint8")).convert("RGB")


def _strip_dicom_phi(dataset: Any) -> None:
    """Vide sur place les balises identifiantes d'un dataset DICOM."""
    for tag in DICOM_PHI_TAGS:
        if tag in dataset:
            dataset.data_element(tag).value = ""


def basic_quality_flag(path: str | Path) -> str:
    """Indicateur de qualité jouet basé sur le nom de fichier.

    À remplacer par de vrais contrôles (exposition, inspiration, centrage)
    dans une implémentation sérieuse. Utilisé uniquement par le backend `toy`.
    """
    name = Path(path).name.lower()
    if "uncertain" in name or "limited" in name:
        return "limited"
    return "good"
