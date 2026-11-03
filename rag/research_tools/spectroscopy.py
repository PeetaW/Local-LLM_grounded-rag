"""Deterministic first-pass XRD, Raman, and FTIR spectrum preprocessing.

No automatic mineral/phase identification or quantitative phase proportions:
those require instrument metadata, calibration, references, and validated fitting.
Input: CSV/TSV/TXT/ASC two-column X,Y files (header/comments tolerated).
Raw x and y remain available. In FTIR transmittance mode peaks are minima;
in absorbance/Raman/XRD mode peaks are maxima.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

SUPPORTED = {"xrd", "raman", "ftir"}
TRANSMITTANCE = "transmittance"
ABSORBANCE = "absorbance"
INTENSITY = "intensity"


@dataclass(frozen=True)
class Spectrum:
    sample_id: str
    technique: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    x_unit: str
    signal_mode: str = INTENSITY
    source_path: str = ""

    def __post_init__(self):
        if self.technique.lower() not in SUPPORTED:
            raise ValueError("Unsupported technique")
        if len(self.x) != len(self.y) or len(self.x) < 3:
            raise ValueError("Spectrum needs >=3 paired numeric data points")
        if not all(math.isfinite(z) for z in self.x + self.y):
            raise ValueError("Non-finite spectrum values")
        if len(set(self.x)) != len(self.x):
            raise ValueError("Duplicate x positions: aggregate replicates explicitly")
        if self.signal_mode not in {INTENSITY, ABSORBANCE, TRANSMITTANCE}:
            raise ValueError("Unknown signal mode")


@dataclass(frozen=True)
class Peak:
    x: float
    signal: float
    prominence: float
    point_index: int


@dataclass(frozen=True)
class SpectrumResult:
    sample_id: str
    technique: str
    x_unit: str
    signal_mode: str
    peaks: tuple[Peak, ...]
    warnings: tuple[str, ...]
    source_path: str
    point_count: int

    def as_dict(self):
        return {
            **{k: v for k, v in asdict(self).items() if k != "peaks"},
            "peaks": [asdict(p) for p in self.peaks],
        }


def _parse_rows(path: Path):
    # Universal text decoding: avoid treating binary SIF as plain text.
    if path.suffix.lower() not in {".csv", ".tsv", ".txt", ".asc", ".dat"}:
        raise ValueError("Unsupported file type; convert vendor binary files (e.g. .sif) first")
    raw = path.read_text(encoding="utf-8-sig", errors="strict")
    if "\x00" in raw:
        raise ValueError("Binary input not supported")
    pairs = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", ";", "//")):
            continue
        # Delimited formats and whitespace formats, including tab-separated ASC
        parts = [v.strip().strip('"') for v in re.split(r"[,;\t]+|\s{2,}", s) if v.strip()]
        if len(parts) < 2:
            parts = s.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 3:
        raise ValueError("Fewer than 3 valid X,Y rows: inspect delimiter/encoding/headers")
    pairs.sort(key=lambda p: p[0])
    return pairs


def load_spectrum(
    path: str | Path,
    technique: str,
    sample_id: str,
    signal_mode: str = INTENSITY,
    x_unit: Optional[str] = None,
) -> Spectrum:
    """Sample ID is required: separate sample IDs and spots must not be silently pooled."""
    technique = technique.lower().strip()
    if not sample_id.strip():
        raise ValueError("sample_id is required")
    if technique not in SUPPORTED:
        raise ValueError("Technique must be XRD, Raman, or FTIR")
    if technique == "ftir" and signal_mode not in {TRANSMITTANCE, ABSORBANCE}:
        raise ValueError("FTIR requires explicit absorbance/transmittance signal mode")
    unit = x_unit or ("2theta_deg" if technique == "xrd" else "cm^-1")
    pairs = _parse_rows(Path(path))
    return Spectrum(
        sample_id=sample_id,
        technique=technique,
        x=tuple(p[0] for p in pairs),
        y=tuple(p[1] for p in pairs),
        x_unit=unit,
        signal_mode=signal_mode,
        source_path=str(path),
    )


def analyze_spectrum(
    spectrum: Spectrum,
    min_prominence: float,
    min_distance_points: int = 3,
    max_peaks: int = 30,
) -> SpectrumResult:
    """Find local extrema using original intensities; no synthetic chemical labels.

    Simple local prominence = depth below/above nearest valleys/peaks within
    the gap to neighboring extrema. This is screening, NOT validated fitting.
    """
    if min_prominence <= 0 or not math.isfinite(min_prominence):
        raise ValueError("Choose a positive absolute prominence in signal units")
    if min_distance_points < 1 or max_peaks < 1:
        raise ValueError("min_distance_points and max_peaks must be positive")
    y = spectrum.y
    trans = spectrum.signal_mode == TRANSMITTANCE
    directed = tuple(-v if trans else v for v in y)
    n = len(directed)
    candidates = []
    for i in range(1, n - 1):
        if not (directed[i] > directed[i-1] and directed[i] >= directed[i+1]):
            continue
        # Use adjacent local minima or endpoints; conservative for broad peaks.
        left = i - 1
        while left > 0 and directed[left - 1] <= directed[left]:
            left -= 1
        right = i + 1
        while right < n - 1 and directed[right + 1] <= directed[right]:
            right += 1
        prom = directed[i] - max(directed[left], directed[right])
        if prom >= min_prominence:
            candidates.append(Peak(spectrum.x[i], y[i], prom, i))
    # Strongest extrema first; deterministic spacing suppression.
    chosen = []
    for p in sorted(candidates, key=lambda p: (-p.prominence, p.point_index)):
        if all(abs(p.point_index - q.point_index) >= min_distance_points for q in chosen):
            chosen.append(p)
        if len(chosen) == max_peaks:
            break
    chosen.sort(key=lambda p: p.x)
    warnings = [
        "Exploratory peak positions only; not calibrated phase/mineral identification.",
        "No baseline correction, peak fitting, instrument broadening correction or uncertainty estimation.",
        "Peak intensities alone do not determine Fe3O4/Fe2O3 mass fractions.",
    ]
    if spectrum.technique == "xrd":
        warnings.append("XRD phase identification requires reference patterns and instrument wavelength; quantification needs validated full-pattern fitting.")
    elif spectrum.technique == "raman":
        warnings.append("Raman assignments need excitation wavelength, laser power and reference spectra; beware laser-induced iron-oxide changes.")
    else:
        warnings.append("FTIR assignments need a reference spectrum and measurement mode; ATR/transmission intensities are not interchangeable.")
    return SpectrumResult(
        sample_id=spectrum.sample_id,
        technique=spectrum.technique,
        x_unit=spectrum.x_unit,
        signal_mode=spectrum.signal_mode,
        peaks=tuple(chosen),
        warnings=tuple(warnings),
        source_path=spectrum.source_path,
        point_count=len(y),
    )


def compare_spectra(results: list[SpectrumResult]) -> dict:
    """Compare measurements without pretending separate sample IDs are replicates."""
    if not results:
        raise ValueError("No results")
    techniques = {r.technique for r in results}
    if len(techniques) != 1:
        raise ValueError("Compare each technique separately")
    return {
        "technique": results[0].technique,
        "sample_ids": sorted({r.sample_id for r in results}),
        "measurement_count": len(results),
        "peaks_by_measurement": [
            {"sample_id": r.sample_id, "source_path": r.source_path, "peak_positions": [p.x for p in r.peaks]}
            for r in results
        ],
        "warning": "Distinct sample IDs are independent samples; multiple measurement positions are not automatically biological or synthesis replicates.",
    }


def save_plot(spectrum: Spectrum, result: SpectrumResult, path: str | Path):
    """Optional matplotlib plot; use in a trusted local environment."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spectrum.x, spectrum.y, linewidth=1.1, color="black")
    for p in result.peaks:
        ax.axvline(p.x, alpha=0.18, color="red")
    ax.set_xlabel(f"{spectrum.technique.upper()} ({spectrum.x_unit})")
    ax.set_ylabel(spectrum.signal_mode)
    ax.set_title(f"{spectrum.sample_id} — {spectrum.technique.upper()}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
