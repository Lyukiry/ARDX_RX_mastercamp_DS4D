from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

# Racine du dépôt sur le sys.path (uvicorn lancé depuis un autre dossier / Windows).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, UploadFile

from src.guardrails import apply_safety_guardrails
from src.inference import predict
from src.database import insert_run

app = FastAPI(title="Assistant radiologue virtuel EFREI", version="1.0.0")
UPLOAD_DIR = Path("tmp_uploads")

# Backend et mode résolus par l'environnement (toy par défaut, CI/Mac sans GPU).
# Sur le PC GPU : `RADIO_BACKEND=vlm` (ou `classifier`) pour la vraie inférence.
DEFAULT_BACKEND = os.environ.get("RADIO_BACKEND", "toy")
DEFAULT_MODE = os.environ.get("RADIO_PROMPT_MODE", "improved")
# Base de logs : jamais à la racine du dépôt (le fichier y est interdit par la CI).
DB_PATH = Path(os.environ.get("RADIO_DB_PATH", Path(tempfile.gettempdir()) / "assistant_radio_runs.sqlite"))


@app.get("/")
def health() -> dict:
    return {"status": "ok", "scope": "educational prototype, not diagnosis"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict:
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = Path(file.filename or "image.png").name
    suffix = Path(filename).suffix or ".png"
    stem = Path(filename).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    target = UPLOAD_DIR / f"uploaded_{safe_stem}{suffix}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    prediction = apply_safety_guardrails(
        predict(target, backend=DEFAULT_BACKEND, mode=DEFAULT_MODE)
    )

    # Journalisation systématique (objectif : 100 % des sorties tracées).
    try:
        insert_run(DB_PATH, safe_stem, str(target), prediction)
        prediction["logged"] = True
    except Exception:  # pragma: no cover - la démo ne doit jamais planter sur un log
        prediction["logged"] = False
    return prediction
