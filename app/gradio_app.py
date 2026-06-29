from __future__ import annotations

import sys
from pathlib import Path

# Permet `python app/gradio_app.py` depuis la racine du dépôt.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from src.guardrails import apply_safety_guardrails
from src.inference import predict
from src.synthetic_eval import MODES


def analyze(image_path, backend, mode):
    """Interface rapide alternative à Streamlit (démo unique)."""
    if image_path is None:
        return {"error": "aucune image fournie"}
    return apply_safety_guardrails(predict(image_path, backend=backend, mode=mode))


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="filepath", label="Radiographie thoracique frontale"),
        gr.Radio(["toy", "noisy", "vlm", "classifier"], value="toy", label="Backend"),
        gr.Radio(list(MODES), value="improved", label="Prompt / mode"),
    ],
    outputs=gr.JSON(label="Sortie structurée (7 champs + avertissements)"),
    title="Assistant radiologue virtuel — prototype pédagogique",
    description="Non destiné au diagnostic. Validation par un professionnel qualifié requise.",
)

if __name__ == "__main__":
    demo.launch()
