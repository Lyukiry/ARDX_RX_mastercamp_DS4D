from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import predict
from src.guardrails import apply_safety_guardrails, validate_prediction
from src.metrics import summarize_metrics
from src.database import insert_run, init_db
from src.synthetic_eval import DEFAULT_SEED


def read_cases(path: Path, split: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split == "all" or r.get("split") == split]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run(mode: str, backend: str, split: str, db_path: Path, seed: int) -> tuple[list[dict], dict]:
    cases = read_cases(ROOT / "data" / "synthetic_cases.csv", split)
    rows = []
    init_db(db_path)
    for case in cases:
        image_path = ROOT / case["image_path"]
        raw = predict(image_path, backend=backend, mode=mode, case=case, seed=seed)
        # Validité JSON mesurée sur la sortie brute (avant garde-fous) pour le §7.3 ;
        # le backend `toy` est toujours valide -> json_valid_rate = 1.0.
        raw_valid = raw.get("raw_json_valid")
        if raw_valid is None:
            raw_valid = validate_prediction(raw)[0]
        safe = apply_safety_guardrails(dict(raw))
        rows.append(
            {
                "case_id": case["case_id"],
                "label": case["label"],
                "quality": case.get("quality", ""),
                "predicted_class": safe["predicted_class"],
                "confidence": safe["confidence"],
                "json_valid": bool(raw_valid),
                "warning": safe.get("warning", ""),
                "uncertainty_warning": safe.get("uncertainty_warning", ""),
                "hallucination": bool(raw.get("hallucination", False)),
                "latency_ms": safe.get("latency_ms", 0),
                "guardrail_errors": ";".join(safe.get("guardrail_errors", [])),
            }
        )
        insert_run(db_path, case["case_id"], str(image_path), safe)
    metrics = summarize_metrics(rows)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["toy", "baseline", "improved"], default="toy")
    parser.add_argument("--backend", choices=["toy", "noisy", "vlm", "classifier"], default="toy")
    parser.add_argument("--split", choices=["all", "smoke", "final"], default="all")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "outputs")
    parser.add_argument("--db-path", type=Path, default=ROOT / "medical_ai_evidence.sqlite")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["baseline", "improved"] if args.mode == "toy" else [args.mode]
    summary = []
    for mode in modes:
        rows, metrics = run(mode, args.backend, args.split, args.db_path, args.seed)
        write_csv(out_dir / f"{mode}_predictions.csv", rows)
        (out_dir / f"{mode}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary.append({"mode": mode, "backend": args.backend, "split": args.split, **metrics})
    write_csv(out_dir / "before_after_summary.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
