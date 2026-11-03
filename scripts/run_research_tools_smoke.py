"""Private local spectroscopy -> PPTX smoke test, without uploading raw data.

Usage:
    python scripts/run_research_tools_smoke.py ^
      --manifest "C:\Data\smoke_manifest.json" ^
      --output "C:\Data\out" ^
      --template "C:\Templates\Shuo_Hung_Research_Presentation_Template.pptx"

All file paths in the manifest can be absolute or relative to the manifest.
This script does not invoke Ollama and never identifies phases automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.research_tools.report_pipeline import analyze_spectra_to_deck


def _read_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("measurements"), list):
        raise ValueError("JSON manifest must contain a 'measurements' array")
    rows = []
    for i, entry in enumerate(data["measurements"], start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"measurements[{i}] must be an object")
        missing = [name for name in ("path", "technique", "sample_id", "min_prominence")
                   if name not in entry]
        if missing:
            raise ValueError(f"measurements[{i}] missing: {', '.join(missing)}")
        entry = dict(entry)
        source = Path(entry["path"])
        if not source.is_absolute():
            source = path.parent / source
        if not source.is_file():
            raise FileNotFoundError(f"Measurement {i}: {source}")
        entry["path"] = str(source.resolve())
        if entry["technique"].lower() == "ftir":
            if entry.get("signal_mode") not in {"absorbance", "transmittance"}:
                raise ValueError(f"Measurement {i}: specify FTIR absorbance or transmittance")
        if not entry["sample_id"].strip():
            raise ValueError(f"Measurement {i}: sample_id cannot be blank")
        rows.append(entry)
    if not rows:
        raise ValueError("Manifest needs at least one measurement")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--template", type=Path, help="Optional private .pptx template")
    parser.add_argument("--title", default="Research spectroscopy data")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = args.manifest.expanduser().resolve()
    rows = _read_manifest(manifest)
    if args.template and not args.template.is_file():
        raise FileNotFoundError(f"Template not found: {args.template}")
    result = analyze_spectra_to_deck(
        measurements=rows,
        output_dir=args.output.expanduser(),
        deck_title=args.title,
        template_path=args.template,
        create_pptx=not args.analysis_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Scientific limitation: candidate peaks are NOT validated phase assignments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
