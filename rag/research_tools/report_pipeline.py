"""File-based workflow connecting raw spectrum data with research slide output.

No AI is called automatically. A Research Agent can select the workflow
through an explicit, allowlisted tool mapping after validating inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .spectroscopy import load_spectrum, analyze_spectrum, save_plot, compare_spectra
from .presentation import DeckSpec, SlideSpec, export_research_deck


def analyze_spectra_to_deck(
    measurements: Sequence[dict],
    output_dir: str | Path,
    *,
    deck_title: str = "Spectroscopy data",
    template_path: str | Path | None = None,
    create_pptx: bool = True,
) -> dict:
    """Each measurement needs path, technique, sample_id, min_prominence.

    FTIR additionally needs explicit signal_mode. Returns machine-readable
    JSON and optional editable PPTX with source-path traceability.
    """
    if not measurements:
        raise ValueError("At least one measurement is required")
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    slides = []
    for idx, row in enumerate(measurements, start=1):
        spec = load_spectrum(
            row["path"], row["technique"], row["sample_id"],
            signal_mode=row.get("signal_mode", "intensity"),
            x_unit=row.get("x_unit"),
        )
        result = analyze_spectrum(
            spec, min_prominence=float(row["min_prominence"]),
            min_distance_points=int(row.get("min_distance_points", 3)),
        )
        plot_path = dest / f"spectrum_{idx:03d}.png"
        save_plot(spec, result, plot_path)
        results.append(result)
        slides.append(SlideSpec(
            title=f"{spec.sample_id} {spec.technique.upper()} data",
            bullets=(
                f"Peak screening: {len(result.peaks)} candidate peaks",
                "Phase assignment and purity require independent confirmation",
            ),
            figure_paths=(str(plot_path),),
            evidence_refs=(str(row["path"]),),
        ))
    summary = {
        "measurements": [r.as_dict() for r in results],
        "comparisons": {
            tech: compare_spectra([r for r in results if r.technique == tech])
            for tech in sorted({r.technique for r in results})
        },
        "not_estimated": ["Fe3O4/Fe2O3 percentages", "absolute phase purity"],
    }
    json_path = dest / "analysis.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pptx_path = None
    if create_pptx:
        pptx_path = export_research_deck(
            DeckSpec(title=deck_title, slides=tuple(slides),
                     future_work=("Validate phase assignments against standards and complementary measurements",)),
            dest / "research_report.pptx", template_path=template_path
        )
    return {"json": str(json_path), "pptx": str(pptx_path) if pptx_path else None,
            "plots": [p for slide in slides for p in slide.figure_paths]}
