from __future__ import annotations

import os
import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image

from src.inference import BACKENDS, predict
from src.guardrails import apply_safety_guardrails

st.set_page_config(page_title="Assistant radiologue virtuel", layout="wide")
st.title("Assistant radiologue virtuel — prototype pédagogique")
st.warning("Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.")

default_backend = os.environ.get("RADIO_BACKEND", "toy")
col_mode, col_backend = st.columns(2)
mode = col_mode.selectbox("Mode", ["baseline", "improved"], index=1)
backend = col_backend.selectbox(
    "Backend", list(BACKENDS), index=list(BACKENDS).index(default_backend) if default_backend in BACKENDS else 0,
    help="toy = prédicteur déterministe (sans GPU) · vlm = MedGemma/Gemma · classifier = CNN/ViT léger",
)

uploaded = st.file_uploader("Déposer une radiographie thoracique frontale", type=["png", "jpg", "jpeg"])

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(Image.open(tmp_path), caption="Image uploadée", use_container_width=True)
    with col2:
        try:
            pred = apply_safety_guardrails(predict(tmp_path, mode=mode, backend=backend))
        except Exception as exc:  # missing GPU stack / model -> explain, do not crash
            st.error(f"Backend '{backend}' indisponible : {exc}")
            st.stop()
        st.metric("Classe", pred["predicted_class"])
        st.metric("Confiance", pred["confidence"])
        st.metric("Qualité image", pred.get("image_quality", "?"))
        st.write("**Observations**", pred["visual_evidence"])
        st.write("**Justification**", pred["justification"])
        st.write("**Limites**", pred["limitations"])
        st.json(pred)
else:
    st.info("Utiliser les images synthétiques dans data/sample_images pour tester le flux.")
