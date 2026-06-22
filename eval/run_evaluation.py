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
from src.metrics import classify_error, confusion_counts, summarize_metrics
from src.database import insert_run, init_db


def read_cases(path: Path, split: str | None = None) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        cases = list(csv.DictReader(f))
    if split:
        cases = [c for c in cases if c.get('split') == split]
    return cases


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def run(mode: str, db_path: Path, *, backend: str, cases_path: Path, split: str | None) -> tuple[list[dict], dict]:
    cases = read_cases(cases_path, split=split)
    rows = []
    init_db(db_path)
    for case in cases:
        image_path = ROOT / case['image_path']
        pred = apply_safety_guardrails(predict(image_path, mode=mode, backend=backend))
        valid, errors = validate_prediction(pred)
        row = {
            'case_id': case['case_id'],
            'label': case['label'],
            'predicted_class': pred['predicted_class'],
            'confidence': pred['confidence'],
            'json_valid': valid,
            'error_type': classify_error(case['label'], pred['predicted_class'], valid),
            'warning': pred.get('warning', ''),
            'latency_ms': pred.get('latency_ms', 0),
            'guardrail_errors': ';'.join(errors),
        }
        rows.append(row)
        insert_run(db_path, case['case_id'], str(image_path), pred)
    metrics = summarize_metrics(rows)
    metrics['confusion'] = confusion_counts([r['label'] for r in rows], [r['predicted_class'] for r in rows])
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the cautious CXR assistant")
    parser.add_argument('--mode', choices=['toy', 'baseline', 'improved'], default='toy',
                        help="'toy' compares baseline vs improved prompts")
    parser.add_argument('--backend', choices=['toy', 'vlm', 'classifier'], default='toy')
    parser.add_argument('--cases', type=Path, default=ROOT / 'data' / 'synthetic_cases.csv')
    parser.add_argument('--split', default=None, help="optional split filter (smoke|dev|final)")
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'eval' / 'outputs')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'medical_ai_evidence.sqlite')
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ['baseline', 'improved'] if args.mode == 'toy' else [args.mode]
    summary = []
    for mode in modes:
        rows, metrics = run(mode, args.db_path, backend=args.backend,
                            cases_path=args.cases, split=args.split)
        write_csv(out_dir / f'{mode}_predictions.csv', rows)
        (out_dir / f'{mode}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        flat = {k: v for k, v in metrics.items() if k != 'confusion'}
        summary.append({'mode': mode, 'backend': args.backend, **flat})
    write_csv(out_dir / 'before_after_summary.csv', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
