from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .preprocessing import basic_quality_flag

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Backends d'inférence disponibles.
#   toy        : déterministe, lit le label dans le nom de fichier (parfait, CI/Mac).
#   noisy      : synthétique imparfait mais reproductible (métriques/erreurs réalistes).
#   vlm        : vrai VLM médical (MedGemma / Gemma), imports paresseux, GPU requis.
#   classifier : classifieur léger CNN/ViT, imports paresseux, GPU/CPU.
BACKENDS = ("toy", "noisy", "vlm", "classifier")


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Prédicteur jouet déterministe pour valider la chaîne du dépôt.

    Lit le label synthétique depuis le nom de fichier. Ce n'est pas une
    inférence médicale : il sert uniquement à prouver que le pipeline tourne.
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

    # Le mode amélioré est plus prudent : doute si la qualité n'est pas bonne.
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


def predict(
    image_path: str | Path,
    *,
    backend: str | None = None,
    mode: str = "baseline",
    case: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Aiguilleur d'inférence : choisit le backend (toy/noisy/vlm/classifier).

    Le backend est résolu par l'argument `backend`, puis la variable
    d'environnement `RADIO_BACKEND`, puis `toy` par défaut. Toutes les sorties
    respectent le même schéma JSON à 7 champs.
    """
    backend = (backend or os.environ.get("RADIO_BACKEND", "toy")).lower()

    if backend == "toy":
        return toy_predict(image_path, mode=mode)
    if backend == "noisy":
        from .synthetic_eval import DEFAULT_SEED, case_from_image, noisy_predict
        return noisy_predict(
            case or case_from_image(image_path),
            mode=mode,
            seed=DEFAULT_SEED if seed is None else seed,
        )
    if backend == "vlm":
        from .vlm_inference import vlm_predict  # import paresseux (torch/transformers)
        return vlm_predict(image_path, mode=mode)
    if backend == "classifier":
        from .classifier import classifier_predict  # import paresseux (torch)
        return classifier_predict(image_path)
    raise ValueError(f"Backend inconnu : {backend!r} (attendu : {BACKENDS})")
