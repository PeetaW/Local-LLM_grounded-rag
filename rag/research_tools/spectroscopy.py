"""Conservative, deterministic XRD / Raman / FTIR numerical preprocessing.

This is screening, not phase identification, purity determination, or chemical
assignment. Experimental source files are never bundled into this repository.
SciPy is optional for the rest of RAG, but required when analyzing spectra.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
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
    measurement_id: str = ""

    def __post_init__(self):
        if self.technique not in SUPPORTED:
            raise ValueError("Unsupported spectroscopy technique")
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if len(self.x) != len(self.y) or len(self.x) < 3:
            raise ValueError("Require at least three paired X/Y values")
        if not all(math.isfinite(v) for v in self.x + self.y):
            raise ValueError("Spectrum contains non-finite numerical values")
        if any(b <= a for a, b in zip(self.x, self.x[1:])):
            raise ValueError("X positions must be strictly increasing")
        if self.signal_mode not in {INTENSITY, ABSORBANCE, TRANSMITTANCE}:
            raise ValueError("Unsupported signal mode")
        if self.technique == "ftir" and self.signal_mode == INTENSITY:
            raise ValueError("FTIR requires explicit absorbance or transmittance mode")


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
    measurement_id: str = ""

    def as_dict(self) -> dict:
        obj = asdict(self)
        obj["peaks"] = [asdict(p) for p in self.peaks]
        return obj


def _parse_rows(path: Path) -> list[tuple[float, float]]:
    if path.suffix.lower() not in {".csv", ".tsv", ".txt", ".asc", ".dat"}:
        raise ValueError("Only text XY files supported; export binary .sif or vendor formats first")
    raw = path.read_text(encoding="utf-8-sig", errors="strict")
    if "\x00" in raw:
        raise ValueError("Binary file detected; convert instrument export first")
    # Bruker RAW4.00 text export contains [Data] after configuration sections.
    # Do not interpret unrelated metadata values as spectral XY pairs.
    if "[RawHeader]" in raw and "[Data]" in raw:
        raw = raw.split("[Data]", 1)[1]
    pairs = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", ";", "//", "[")):
            continue
        # Numeric decimal separator is a period, not a locale-specific comma.
        parts = [p.strip().strip('"') for p in re.split(r"[,;\t]+|\s{2,}", s) if p.strip()]
        if len(parts) < 2:
            parts = s.split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 3:
        raise ValueError("Fewer than three numeric XY pairs; inspect file export")
    pairs.sort(key=lambda item: item[0])
    if any(b[0] <= a[0] for a, b in zip(pairs, pairs[1:])):
        raise ValueError("Duplicate X positions: do not silently average measurements")
    return pairs


def load_spectrum(
    path: str | Path, technique: str, sample_id: str,
    signal_mode: str = INTENSITY, x_unit: Optional[str] = None,
    measurement_id: str = "",
) -> Spectrum:
    technique = technique.strip().lower()
    if technique not in SUPPORTED:
        raise ValueError("Technique must be xrd, raman, or ftir")
    if technique == "ftir" and signal_mode not in {TRANSMITTANCE, ABSORBANCE}:
        raise ValueError("Explicit FTIR signal_mode=absorbance or transmittance required")
    pairs = _parse_rows(Path(path))
    return Spectrum(
        sample_id=sample_id, technique=technique,
        x=tuple(p[0] for p in pairs), y=tuple(p[1] for p in pairs),
        x_unit=x_unit or ("2theta_deg" if technique == "xrd" else "cm^-1"),
        signal_mode=signal_mode, source_path=str(path),
        measurement_id=measurement_id,
    )


def _smooth_directed(spectrum: Spectrum, smoothing_window_points: int) -> tuple[list[float], list[str]]:
    """Smooth intensity using Savitzky-Golay; never overwrite raw measurements."""
    try:
        import numpy as np
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise RuntimeError("Spectral peak screening needs scipy and numpy (optional research-tools extras)") from exc

    y = np.asarray(spectrum.y, dtype=float)
    directed = -y if spectrum.signal_mode == TRANSMITTANCE else y
    warnings = []
    x = np.asarray(spectrum.x, dtype=float)
    dx = np.diff(x)
    median_dx = float(np.median(dx))
    if median_dx <= 0:
        raise ValueError("Invalid X increment")
    if np.max(dx) > 3 * median_dx or np.min(dx) < median_dx / 3:
        warnings.append("Irregular X spacing: point-window smoothing/distance may distort interpretation")
    if smoothing_window_points < 1:
        raise ValueError("smoothing_window_points must be positive")
    if smoothing_window_points == 1 or len(directed) < 7:
        return directed.tolist(), warnings
    window = min(smoothing_window_points, len(directed) if len(directed) % 2 else len(directed) - 1)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return directed.tolist(), warnings
    smooth = savgol_filter(directed, window_length=window, polyorder=min(3, window - 2))
    return smooth.tolist(), warnings


def analyze_spectrum(
    spectrum: Spectrum,
    min_prominence: float,
    min_distance_points: Optional[int] = None,
    max_peaks: int = 30,
    smoothing_window_points: int = 15,
    min_distance_x: Optional[float] = None,
) -> SpectrumResult:
    """Screen local maxima (absorbance / intensity) or minima (%T).

    Peak heights are relative to a Savitzky-Golay-smoothed baseline, with
    SciPy prominence, NOT the immediate neighbor difference. No background
    correction, calibration, profile fitting, or chemical assignment is done.
    """
    if min_prominence <= 0 or not math.isfinite(min_prominence):
        raise ValueError("min_prominence must be positive in the original signal units")
    if max_peaks < 1:
        raise ValueError("max_peaks must be at least 1")
    smoothed, cautions = _smooth_directed(spectrum, smoothing_window_points)
    import numpy as np
    from scipy.signal import find_peaks
    step = float(np.median(np.diff(spectrum.x)))
    # Laboratory-scale default separation avoids dozens of noise extrema.
    default_spacing = {"xrd": 0.4, "raman": 15.0, "ftir": 15.0}[spectrum.technique]
    if min_distance_points is not None:
        if min_distance_points < 1:
            raise ValueError("min_distance_points must be positive")
        distance_points = min_distance_points
    else:
        spacing = default_spacing if min_distance_x is None else min_distance_x
        if spacing <= 0 or not math.isfinite(spacing):
            raise ValueError("min_distance_x must be positive")
        distance_points = max(1, round(spacing / step))
    indices, info = find_peaks(
        smoothed, prominence=min_prominence, distance=distance_points
    )
    peaks = [
        Peak(float(spectrum.x[int(idx)]), float(spectrum.y[int(idx)]),
             float(prom), int(idx))
        for idx, prom in zip(indices, info["prominences"])
    ]
    # Restrict to strongest candidate peaks, then present in axis order.
    peaks = sorted(peaks, key=lambda p: (-p.prominence, p.point_index))[:max_peaks]
    peaks.sort(key=lambda p: p.x)
    warnings = [
        "Candidate extrema only; baseline, instrument calibration and uncertainty require validation.",
        "Peak intensities do not give Fe3O4/Fe2O3 mass fractions or phase purity.",
    ]
    warnings.extend(cautions)
    if spectrum.technique == "xrd":
        warnings.append("XRD phase validation requires measured wavelength, calibrated references and full-pattern fitting.")
    elif spectrum.technique == "raman":
        warnings.append("Raman identification requires laser wavelength/power, references and laser-heating checks.")
    else:
        warnings.append("FTIR mode and ATR/transmission geometry must be checked; %T minima are treated as peaks.")
    return SpectrumResult(
        sample_id=spectrum.sample_id, technique=spectrum.technique,
        x_unit=spectrum.x_unit, signal_mode=spectrum.signal_mode,
        peaks=tuple(peaks), warnings=tuple(warnings),
        source_path=spectrum.source_path, point_count=len(spectrum.y),
        measurement_id=spectrum.measurement_id,
    )


def compare_spectra(results: list[SpectrumResult]) -> dict:
    """Maintain independent sample identities and measurement-level provenance."""
    if not results:
        raise ValueError("No spectral results")
    if len({r.technique for r in results}) != 1:
        raise ValueError("Compare one technique at a time")
    return {
        "technique": results[0].technique,
        "sample_ids": sorted({r.sample_id for r in results}),
        "measurement_count": len(results),
        "peaks_by_measurement": [
            {
                "sample_id": r.sample_id,
                "measurement_id": r.measurement_id,
                "source_path": r.source_path,
                "peak_positions": [p.x for p in r.peaks],
            }
            for r in results
        ],
        "warning": "Measurement spots are not independent synthesis batches; do not silently pool or perform unjustified significance tests.",
    }


def save_plot(spectrum: Spectrum, result: SpectrumResult, path: str | Path):
    """Render raw trace with detected peak markers; no automatic assignments."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spectrum.x, spectrum.y, linewidth=1.1, color="black")
    for peak in result.peaks:
        ax.axvline(peak.x, alpha=0.22, color="red")
    if spectrum.technique == "ftir":
        ax.invert_xaxis()  # Standard convention: high cm^-1 at left
    ax.set_xlabel(f"{spectrum.technique.upper()} ({spectrum.x_unit})")
    ax.set_ylabel(spectrum.signal_mode)
    ax.set_title(f"{spectrum.sample_id} / {spectrum.measurement_id or 'measurement'} — {spectrum.technique.upper()}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
