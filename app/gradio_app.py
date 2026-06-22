from __future__ import annotations

import os

import gradio as gr
from src.inference import BACKENDS, predict
from src.guardrails import apply_safety_guardrails

DEFAULT_BACKEND = os.environ.get("RADIO_BACKEND", "toy")


def analyze(image_path, mode, backend):
    if image_path is None:
        return {"error": "no image"}
    try:
        return apply_safety_guardrails(predict(image_path, mode=mode, backend=backend))
    except Exception as exc:  # missing GPU stack / model -> surface, do not crash
        return {"error": f"backend '{backend}' unavailable: {exc}"}


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="filepath", label="Radiographie thoracique"),
        gr.Radio(["baseline", "improved"], value="improved", label="Mode"),
        gr.Radio(list(BACKENDS), value=DEFAULT_BACKEND, label="Backend"),
    ],
    outputs=gr.JSON(label="Sortie structurée"),
    title="Assistant radiologue virtuel — prototype pédagogique",
    description="Non destiné au diagnostic. Validation par un professionnel qualifié requise.",
)

if __name__ == "__main__":
    demo.launch()
