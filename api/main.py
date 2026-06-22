from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile

from src.inference import predict as run_inference
from src.guardrails import apply_safety_guardrails

app = FastAPI(title="Assistant radiologue virtuel EFREI", version="0.1.0")
UPLOAD_DIR = Path("tmp_uploads")

# Backend is chosen via the RADIO_BACKEND env var (toy | vlm | classifier).
# The default stays ``toy`` so the API runs anywhere without a GPU.
BACKEND = os.environ.get("RADIO_BACKEND", "toy")


@app.get("/")
def health() -> dict:
    return {"status": "ok", "scope": "educational prototype, not diagnosis", "backend": BACKEND}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = Path(file.filename or "image.png").name
    suffix = Path(filename).suffix or ".png"
    stem = Path(filename).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    target = UPLOAD_DIR / f"uploaded_{safe_stem}{suffix}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    pred = run_inference(target, mode="improved", backend=BACKEND)
    return apply_safety_guardrails(pred)
