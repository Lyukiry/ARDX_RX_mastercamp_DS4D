"""Real vision-language model (VLM) backend for the educational prototype.

This module connects the project to a real medical VLM (MedGemma 4B by default,
any Hugging Face image-text-to-text model otherwise). It keeps **exactly** the
same output schema as :func:`src.inference.toy_predict` so the rest of the
pipeline (guardrails, metrics, database, API, UI) does not change.

Heavy dependencies (``torch``, ``transformers``) are imported lazily inside the
predictor so that:

- the repository smoke test and CI stay torch-free,
- ``src.inference`` can keep ``toy`` as a default backend on a laptop,
- the real model runs on a GPU machine without changing any contract.

The pure-Python helpers (:func:`extract_json`, :func:`coerce_to_schema`,
:func:`apply_uncertainty_rule`) are unit-tested without a GPU.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .guardrails import ALLOWED_CLASSES, WARNING_TEXT

# Default model: MedGemma 4B instruction-tuned (recommended baseline in the brief).
# Override with the RADIO_VLM_MODEL environment variable or the ``model_name`` arg.
DEFAULT_VLM_MODEL = "google/medgemma-4b-it"

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_PROMPT_FILES = {"baseline": "baseline_prompt.txt", "improved": "improved_prompt.txt"}

# Class normalisation: a real model rarely answers with the exact internal label.
_CLASS_SYNONYMS = (
    ("suspected_opacity", ("suspected_opacity", "opac", "pneumo", "consolidation",
                            "infiltrat", "abnormal", "anormal", "suspect")),
    ("normal", ("normal", "no finding", "clear", "ras")),
    ("uncertain", ("uncertain", "incertain", "unsure", "doubt", "indeterminate")),
)
_QUALITY_VALUES = {"good", "limited", "poor"}
_STD_LIMITATIONS = ["no clinical context", "not a validated medical model"]


# --------------------------------------------------------------------------- #
# Pure-Python helpers (no torch) -- unit-tested on CPU.
# --------------------------------------------------------------------------- #
def load_prompt(mode: str = "baseline") -> str:
    """Read the baseline or improved prompt from the ``prompts/`` directory."""
    filename = _PROMPT_FILES.get(mode, _PROMPT_FILES["baseline"])
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from model output.

    Handles fenced ```json blocks and free text around the object. Returns
    ``None`` when no valid JSON object can be parsed (the caller then falls
    back to ``uncertain``).
    """
    if not text:
        return None

    # 1) Prefer a fenced code block.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates: list[str] = []
    if fence:
        candidates.append(fence.group(1))

    # 2) Otherwise scan for the first balanced {...} block (string-aware).
    candidates.extend(_balanced_objects(text))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _balanced_objects(text: str) -> list[str]:
    """Yield substrings that are balanced ``{...}`` blocks, ignoring braces in strings."""
    objects: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objects.append(text[start:i + 1])
    return objects


def _normalize_class(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ALLOWED_CLASSES:
        return text
    for canonical, needles in _CLASS_SYNONYMS:
        if any(needle in text for needle in needles):
            return canonical
    return "uncertain"


def _normalize_quality(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _QUALITY_VALUES:
        return text
    if any(k in text for k in ("bad", "mauvais", "poor")):
        return "poor"
    if any(k in text for k in ("limit", "moyen", "fair")):
        return "limited"
    return "good"


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if value in (None, ""):
        return []
    return [str(value)]


def _as_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, conf))


def coerce_to_schema(
    raw: dict[str, Any] | None,
    *,
    model_name: str,
    prompt_version: str,
    latency_ms: int,
    fallback_quality: str = "good",
) -> dict[str, Any]:
    """Normalise an arbitrary model dict into the project output schema.

    A ``None`` or unusable ``raw`` becomes a safe ``uncertain`` answer. The
    downstream guardrails ( :func:`src.guardrails.apply_safety_guardrails` )
    remain the single source of truth for the warning and final safety pass.
    """
    raw = raw or {}
    limitations = _as_str_list(raw.get("limitations"))
    for std in _STD_LIMITATIONS:
        if std not in limitations:
            limitations.append(std)

    if not raw:
        limitations.insert(0, "model output could not be parsed as JSON")

    return {
        "image_quality": _normalize_quality(raw.get("image_quality", fallback_quality)),
        "predicted_class": _normalize_class(raw.get("predicted_class")),
        "confidence": round(_as_confidence(raw.get("confidence", 0.5 if raw else 0.4)), 3),
        "visual_evidence": _as_str_list(raw.get("visual_evidence")) or ["no explicit finding reported"],
        "justification": str(raw.get("justification", "") or
                             "The model did not return a usable justification; defaulting to uncertainty."),
        "limitations": limitations,
        "warning": WARNING_TEXT,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "latency_ms": int(latency_ms),
    }


def apply_uncertainty_rule(pred: dict[str, Any], threshold: float = 0.60) -> dict[str, Any]:
    """Improved-prompt rule: low confidence or poor quality forces ``uncertain``."""
    conf = _as_confidence(pred.get("confidence", 0.0))
    if conf < threshold or pred.get("image_quality") == "poor":
        pred["predicted_class"] = "uncertain"
    return pred


# --------------------------------------------------------------------------- #
# Real model predictor (lazy torch / transformers).
# --------------------------------------------------------------------------- #
class VLMPredictor:
    """Hugging Face image-text-to-text predictor (MedGemma 4B by default).

    Example
    -------
    >>> predictor = VLMPredictor()                # doctest: +SKIP
    >>> predictor.predict("chest.png", mode="improved")   # doctest: +SKIP
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VLM_MODEL,
        *,
        device: str | None = None,
        dtype: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._dtype = dtype
        self._model = None
        self._processor = None

    # -- lazy loading -------------------------------------------------------- #
    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, device: str):
        import torch

        if self._dtype != "auto":
            return getattr(torch, self._dtype)
        if device == "cuda":
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32

    def load(self) -> "VLMPredictor":
        if self._model is not None:
            return self
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = self._resolve_device()
        dtype = self._resolve_dtype(device)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
        )
        if device != "cuda":
            self._model = self._model.to(device)
        self._model.eval()
        self._resolved_device = device
        return self

    # -- inference ----------------------------------------------------------- #
    def predict(self, image_path, mode: str = "baseline") -> dict[str, Any]:
        from PIL import Image
        import torch

        self.load()
        start = time.perf_counter()
        prompt = load_prompt(mode)
        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text",
             "text": "You are a cautious educational radiology assistant. Answer with valid JSON only."}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]

        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._resolved_device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        decoded = self._processor.decode(generation[0][input_len:], skip_special_tokens=True)

        latency_ms = int((time.perf_counter() - start) * 1000)
        pred = coerce_to_schema(
            extract_json(decoded),
            model_name=self.model_name,
            prompt_version=f"{mode}_vlm_v1",
            latency_ms=latency_ms,
        )
        pred["raw_model_text"] = decoded
        if mode == "improved":
            pred = apply_uncertainty_rule(pred)
        return pred


# Module-level cache so the heavy model loads only once per process.
_PREDICTOR: VLMPredictor | None = None


def get_predictor(model_name: str | None = None, **kwargs) -> VLMPredictor:
    """Return a process-wide cached :class:`VLMPredictor`."""
    global _PREDICTOR
    import os

    name = model_name or os.environ.get("RADIO_VLM_MODEL", DEFAULT_VLM_MODEL)
    if _PREDICTOR is None or _PREDICTOR.model_name != name:
        _PREDICTOR = VLMPredictor(name, **kwargs)
    return _PREDICTOR


def vlm_predict(image_path, mode: str = "baseline", model_name: str | None = None) -> dict[str, Any]:
    """Convenience wrapper used by the inference dispatcher and API."""
    return get_predictor(model_name).predict(image_path, mode=mode)
