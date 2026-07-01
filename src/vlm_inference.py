"""Backend VLM médical réel (MedGemma 4B / Gemma 4) — exécuté sur le PC GPU.

Imports paresseux : `torch` et `transformers` ne sont importés que lorsque le
backend `vlm` est réellement utilisé. Le mode jouet, la CI et le Mac sans GPU
n'en dépendent jamais.

Modèle par défaut : `google/medgemma-4b-it` (accès Hugging Face sous licence ;
voir docs/guide_execution_gpu.md). Remplaçable par `RADIO_VLM_MODEL`.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .guardrails import WARNING_TEXT, apply_safety_guardrails

ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILES = {
    "baseline": ROOT / "prompts" / "baseline_prompt.txt",
    "improved": ROOT / "prompts" / "improved_prompt.txt",
    "structured": ROOT / "prompts" / "structured_prompt.txt",
}
DEFAULT_MODEL = os.environ.get("RADIO_VLM_MODEL", "google/medgemma-4b-it")

# Cache du modèle chargé (clé = identifiant du modèle).
_MODEL_CACHE: dict[str, Any] = {}


def load_prompt(mode: str) -> str:
    """Charge le texte d'un prompt selon le mode (baseline/improved/structured)."""
    path = PROMPT_FILES.get(mode, PROMPT_FILES["improved"])
    return path.read_text(encoding="utf-8")


def _extract_json(text: str) -> dict[str, Any]:
    """Extrait le premier objet JSON d'une génération de modèle, sinon {}."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _coerce_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Ramène une sortie de modèle vers le schéma à 7 champs, prudemment."""
    allowed = {"normal", "suspected_opacity", "uncertain"}
    predicted = str(raw.get("predicted_class", "")).strip()
    if predicted not in allowed:
        predicted = "uncertain"
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    quality = raw.get("image_quality") if raw.get("image_quality") in {"good", "limited", "poor"} else "limited"
    evidence = raw.get("visual_evidence")
    limitations = raw.get("limitations")
    return {
        "image_quality": quality,
        "predicted_class": predicted,
        "confidence": max(0.0, min(1.0, confidence)),
        "visual_evidence": evidence if isinstance(evidence, list) else [str(evidence)] if evidence else [],
        "justification": str(raw.get("justification", "")).strip() or "Sortie de modèle non concluante.",
        "limitations": limitations if isinstance(limitations, list) else ["sortie modèle incomplète"],
        "warning": WARNING_TEXT,
    }


def _use_4bit(torch) -> bool:
    """Quantification 4-bit : forcée par RADIO_VLM_4BIT, sinon auto si VRAM < 10 Go.

    MedGemma 4B pèse ~8,6 Go en bf16 : sur une carte de 8 Go (ex. RTX 3070), le
    chargement plein déborde en mémoire partagée WDDM et devient inutilisable.
    """
    flag = os.environ.get("RADIO_VLM_4BIT", "").strip().lower()
    if flag:
        return flag not in {"0", "false", "no"}
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 10 * 1024**3


def load_model(model_id: str | None = None):
    """Charge (et met en cache) le processeur et le modèle VLM. Imports paresseux."""
    model_id = model_id or DEFAULT_MODEL
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - dépend du PC GPU
        raise RuntimeError(
            "Le backend `vlm` requiert torch + transformers (voir docs/guide_execution_gpu.md)."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_id)
    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},  # force all layers to GPU 0; "auto" leaves some on meta with accelerate>=1.14
    }
    if _use_4bit(torch):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    model.eval()
    _MODEL_CACHE[model_id] = (processor, model)
    return processor, model


def vlm_predict(image_path: str | Path, mode: str = "improved", model_id: str | None = None) -> dict[str, Any]:
    """Prédit avec un vrai VLM médical. L'image est placée AVANT le texte (P1)."""
    from PIL import Image  # léger, déjà disponible

    from .preprocessing import preprocess

    start = time.perf_counter()
    processor, model = load_model(model_id)
    image, _meta = preprocess(image_path)
    prompt = load_prompt(mode)

    # Format multimodal : image puis instruction (recommandation Gemma/MedGemma).
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    import torch
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    text = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )[0]

    prediction = _coerce_schema(_extract_json(text))
    prediction.update(
        {
            "model_name": (model_id or DEFAULT_MODEL),
            "prompt_version": f"{mode}_v1",
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "raw_text": text,
        }
    )
    # On applique les garde-fous ici aussi : la sortie d'un vrai modèle est moins fiable.
    return apply_safety_guardrails(prediction)
