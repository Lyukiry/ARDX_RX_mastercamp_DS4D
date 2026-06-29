from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Permet `streamlit run app/streamlit_app.py` depuis la racine : Streamlit ajoute
# le dossier du script (app/) au sys.path, pas la racine du dépôt.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from PIL import Image

from src.database import connect, init_db, insert_run
from src.guardrails import UNCERTAINTY_WARNING_TEXT, WARNING_TEXT, apply_safety_guardrails
from src.inference import predict
from src.metrics import confusion_matrix, summarize_metrics
from src.synthetic_eval import MODES, noisy_predict

ROOT = Path(__file__).resolve().parents[1]
CASES_CSV = ROOT / "data" / "synthetic_cases.csv"
DB_PATH = Path(os.environ.get("RADIO_DB_PATH", Path(tempfile.gettempdir()) / "assistant_radio_runs.sqlite"))

# Intitulés français affichés (clés internes anglaises stables).
CLASS_FR = {"normal": "Normal", "suspected_opacity": "Suspicion d'opacité", "uncertain": "Incertain"}
QUALITY_FR = {"good": "bonne", "limited": "moyenne", "poor": "mauvaise"}

st.set_page_config(page_title="Assistant radiologue virtuel", layout="wide")
st.title("Assistant radiologue virtuel — prototype pédagogique")
st.warning(WARNING_TEXT)


@st.cache_data
def load_cases() -> pd.DataFrame:
    return pd.read_csv(CASES_CSV)


def run_prediction(image_path: Path, backend: str, mode: str, case: dict | None) -> dict:
    prediction = apply_safety_guardrails(predict(image_path, backend=backend, mode=mode, case=case))
    try:
        init_db(DB_PATH)
        insert_run(DB_PATH, case.get("case_id", image_path.stem) if case else image_path.stem, str(image_path), prediction)
    except Exception:
        pass
    return prediction


def open_for_display(path: Path) -> Image.Image:
    """Ouvre une image pour l'affichage, en gérant le DICOM (anonymisé)."""
    if Path(path).suffix.lower() in {".dcm", ".dicom"}:
        from src.preprocessing import dicom_to_image
        return dicom_to_image(path)
    return Image.open(path).convert("RGB")


def show_prediction(prediction: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Classe", CLASS_FR.get(prediction["predicted_class"], prediction["predicted_class"]))
    col2.metric("Confiance", f"{prediction['confidence']:.2f}")
    col3.metric("Qualité image", QUALITY_FR.get(prediction.get("image_quality"), prediction.get("image_quality")))
    if prediction.get("uncertainty_warning"):
        st.error("⚠️ " + prediction["uncertainty_warning"])
    st.write("**Observations visuelles**")
    st.write(prediction.get("visual_evidence"))
    st.write("**Justification**", prediction.get("justification"))
    st.write("**Limites**", prediction.get("limitations"))
    st.caption(f"Modèle : {prediction.get('model_name')} · latence {prediction.get('latency_ms')} ms")
    with st.expander("Sortie JSON brute"):
        st.json(prediction)


tab_cas, tab_analyse, tab_apprentissage, tab_suivi = st.tabs(
    ["Cas", "Analyse", "Apprentissage", "Suivi"]
)

# ---------------------------------------------------------------- Onglet Cas
with tab_cas:
    st.subheader("Catalogue des cas synthétiques")
    st.caption("Jeu jouet : valide la chaîne logicielle, sans valeur médicale.")
    cases = load_cases()
    split = st.selectbox("Sous-ensemble", ["all", "smoke", "final"], index=0)
    view = cases if split == "all" else cases[cases["split"] == split]
    display = view.assign(
        classe=view["label"].map(CLASS_FR),
        qualité=view["quality"].map(QUALITY_FR),
    )[["case_id", "classe", "qualité", "split", "notes"]]
    st.dataframe(display, use_container_width=True, hide_index=True)
    selected = st.selectbox("Visualiser un cas", view["case_id"].tolist())
    row = view[view["case_id"] == selected].iloc[0]
    st.image(Image.open(ROOT / row["image_path"]), width=360,
             caption=f"{selected} — vérité terrain : {CLASS_FR[row['label']]} (qualité {QUALITY_FR[row['quality']]})")

# ------------------------------------------------------------ Onglet Analyse
with tab_analyse:
    st.subheader("Analyser une radiographie")
    backend = st.selectbox(
        "Backend", ["toy", "noisy", "vlm", "classifier"],
        help="toy/noisy tournent partout. vlm/classifier nécessitent un GPU + RADIO_BACKEND.",
    )
    mode = st.selectbox("Prompt / mode", list(MODES))
    source = st.radio(
        "Source de l'image",
        ["Cas de test (catalogue)", "Dataset RSNA / externe", "Téléverser"],
        horizontal=True,
    )

    image_path = None
    case = None
    if source == "Cas de test (catalogue)":
        cases = load_cases()
        case_id = st.selectbox("Cas", cases["case_id"].tolist(), key="analyse_case")
        row = cases[cases["case_id"] == case_id].iloc[0]
        image_path = ROOT / row["image_path"]
        case = row.to_dict()

    elif source == "Dataset RSNA / externe":
        st.caption("Pointez un **dossier** d'images (RSNA : .dcm/.png/.jpg) ou un **CSV** "
                   "(colonnes `case_id,image_path,label,...`). Les DICOM sont anonymisés automatiquement.")
        raw_path = st.text_input("Chemin d'un dossier ou d'un CSV",
                                 value=str(ROOT / "data" / "rsna_cases.csv"),
                                 help="Ex. ~/datasets/rsna/stage_2_train_images  ou  data/rsna_cases.csv")
        p = Path(raw_path).expanduser()
        if not raw_path:
            pass
        elif not p.exists():
            st.warning("Chemin introuvable.")
        elif p.suffix.lower() == ".csv":
            ext = pd.read_csv(p)
            if "image_path" not in ext.columns:
                st.error("CSV invalide : la colonne 'image_path' est requise.")
            else:
                key_col = "case_id" if "case_id" in ext.columns else "image_path"
                pick = st.selectbox("Cas", ext[key_col].astype(str).tolist(), key="rsna_csv")
                erow = ext[ext[key_col].astype(str) == pick].iloc[0]
                ip = Path(str(erow["image_path"])).expanduser()
                image_path = ip if ip.is_absolute() else (ROOT / ip)
                case = erow.to_dict()
        elif p.is_dir():
            files = sorted(f for f in p.iterdir()
                           if f.suffix.lower() in {".dcm", ".dicom", ".png", ".jpg", ".jpeg"})
            if not files:
                st.warning("Aucune image (.dcm/.png/.jpg) dans ce dossier.")
            else:
                filt = st.text_input("Filtrer par nom (sous-chaîne)", "")
                shown = [f for f in files if filt.lower() in f.name.lower()][:500]
                st.caption(f"{len(files)} image(s) ; {len(shown)} affichée(s) (max 500).")
                if shown:
                    chosen = st.selectbox("Image", [f.name for f in shown], key="rsna_dir")
                    image_path = next(f for f in shown if f.name == chosen)
        else:
            st.error("Chemin non reconnu (ni dossier, ni CSV).")
        if backend == "noisy":
            st.info("Le backend `noisy` est calibré pour le jeu synthétique. Pour de vraies "
                    "images RSNA, choisissez `vlm` (MedGemma/Gemma) ou `classifier`.")

    else:  # Téléverser
        uploaded = st.file_uploader("Radiographie thoracique frontale",
                                    type=["png", "jpg", "jpeg", "dcm", "dicom"])
        if uploaded:
            suffix = Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                image_path = Path(tmp.name)

    if image_path and st.button("Analyser", type="primary"):
        col_img, col_res = st.columns([1, 1])
        with col_img:
            try:
                st.image(open_for_display(image_path), use_container_width=True)
            except Exception as exc:
                st.warning(f"Aperçu indisponible : {exc}")
        with col_res:
            try:
                show_prediction(run_prediction(Path(image_path), backend, mode, case))
            except Exception as exc:
                st.error(f"Backend « {backend} » indisponible : {exc}\n\n"
                         "Les backends `vlm`/`classifier` nécessitent le GPU + les dépendances "
                         "(voir README_GPU.md).")

# ------------------------------------------------------ Onglet Apprentissage
with tab_apprentissage:
    st.subheader("Les 3 classes (taxonomie gelée)")
    st.table(pd.DataFrame([
        {"Classe": "Normal", "Clé": "normal", "Définition": "Pas d'opacité détectée"},
        {"Classe": "Suspicion d'opacité", "Clé": "suspected_opacity", "Définition": "Opacité possible détectée"},
        {"Classe": "Incertain", "Clé": "uncertain", "Définition": "Qualité ou signes non concluants"},
    ]))
    st.subheader("Règle d'avertissement (§7.2)")
    st.info(f"Escalade d'incertitude si **confiance < 0.60** OU **qualité = mauvaise**.\n\n> {UNCERTAINTY_WARNING_TEXT}")
    st.subheader("Comparaison des 3 prompts (jeu final, backend noisy)")
    final_cases = load_cases().query("split == 'final'").to_dict("records")
    comparison = []
    for prompt_mode in MODES:
        preds = [noisy_predict(c, mode=prompt_mode) for c in final_cases]
        comparison.append({
            "Prompt": prompt_mode,
            "JSON valide": f"{sum(p['raw_json_valid'] for p in preds) / len(preds):.0%}",
            "Justif. courte": f"{sum(p['justification_short'] for p in preds) / len(preds):.0%}",
            "Avertissement": f"{sum(p['raw_warning_present'] for p in preds) / len(preds):.0%}",
            "Hallucination": f"{sum(p['hallucination'] for p in preds) / len(preds):.0%}",
        })
    st.table(pd.DataFrame(comparison))
    st.caption("Indicateurs mesurés sur la sortie brute du modèle, avant garde-fous. "
               "En production, les garde-fous portent JSON valide et avertissement à 100 %.")

# ------------------------------------------------------------- Onglet Suivi
with tab_suivi:
    st.subheader("Traçabilité des inférences (SQLite)")
    st.caption(f"Base de logs : {DB_PATH}")
    try:
        init_db(DB_PATH)
        conn = connect(DB_PATH)
        runs = pd.read_sql_query(
            "SELECT case_id, model_name, predicted_class, confidence, latency_ms, created_at "
            "FROM runs ORDER BY id DESC LIMIT 200",
            conn,
        )
        conn.close()
    except Exception as exc:
        runs = pd.DataFrame()
        st.warning(f"Pas encore de logs : {exc}")

    if runs.empty:
        st.info("Lancez une analyse dans l'onglet « Analyse » pour alimenter le journal.")
    else:
        st.metric("Inférences journalisées", len(runs))
        runs_display = runs.assign(classe=runs["predicted_class"].map(CLASS_FR))
        st.dataframe(runs_display, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Évaluation rapide (jeu final, backend noisy)")
    if st.button("Calculer les métriques baseline vs amélioré"):
        final_cases = load_cases().query("split == 'final'").to_dict("records")
        for prompt_mode in ("baseline", "improved"):
            rows = []
            for c in final_cases:
                p = apply_safety_guardrails(noisy_predict(c, mode=prompt_mode))
                rows.append({"label": c["label"], "predicted_class": p["predicted_class"],
                             "json_valid": p.get("raw_json_valid", True), "warning": p["warning"],
                             "latency_ms": p["latency_ms"]})
            metrics = summarize_metrics(rows)
            st.write(f"**{prompt_mode}** — accuracy {metrics['accuracy']}, macro-F1 {metrics['macro_f1']}, "
                     f"sensibilité {metrics['sensitivity']}, spécificité {metrics['specificity']}")
            yt = [r["label"] for r in rows]
            yp = [r["predicted_class"] for r in rows]
            cm = confusion_matrix(yt, yp)
            st.table(pd.DataFrame(cm).T.rename(index=CLASS_FR, columns=CLASS_FR))
