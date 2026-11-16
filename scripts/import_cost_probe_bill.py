"""Offline import of explicitly mapped official probe bills; no API calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vcrb.cost_probe_report import import_billing_csv, render_report, validate_billing, write_billing_template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--csv", type=Path, help="Explicit per-model official bill mapping CSV")
    parser.add_argument("--billing-json", type=Path)
    parser.add_argument("--complete-model", action="append", default=[])
    parser.add_argument("--template", type=Path, help="Write an empty mapping template only")
    parser.add_argument("--save-billing", type=Path, help="Private validated billing JSON output")
    parser.add_argument("--output", type=Path, help="Sanitized standalone HTML")
    parser.add_argument("--recognition", type=Path, help="Reviewed C benchmark summary for the briefing")
    args = parser.parse_args()
    if args.template:
        if any((args.csv, args.billing_json, args.save_billing, args.output)):
            parser.error("--template is a separate operation")
        write_billing_template(args.template)
        return
    if not args.manifest or not args.output or bool(args.csv) == bool(args.billing_json):
        parser.error("Provide --manifest, --output and exactly one of --csv/--billing-json")
    if args.billing_json and args.complete_model:
        parser.error("JSON imports declare complete_models inside the evidence file")
    billing = import_billing_csv(args.manifest, args.csv, complete_models=args.complete_model) if args.csv else validate_billing(args.manifest, args.billing_json)
    if args.save_billing:
        args.save_billing.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.save_billing.parent, prefix=".private-billing-", delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(json.dumps(billing, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.save_billing)
    recognition = args.recognition
    default_scores = ROOT / 'reports/2026-11-15-c-search/summary.json'
    if recognition is None and default_scores.is_file():
        candidate = json.loads(default_scores.read_text())
        if candidate.get('dataset_hash') == json.loads(args.manifest.read_text()).get('dataset_hash'):
            recognition = candidate
    summary = render_report(args.manifest, args.output, billing, recognition=recognition)
    print(json.dumps({"models": len(summary["models"]), "officially_confirmed": sum(row["billing_complete"] for row in summary["models"]), "report": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
