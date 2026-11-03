"""Scientific review checklists, not impersonations of named researchers.

Expert roles provide independent questions. They are not evidence or a
substitute for reference data, instrument metadata, or experimental controls.
"""
from __future__ import annotations

PROFILES = {
    "xrd": (
        "Check radiation wavelength, calibration, background, peak overlap, reference "
        "patterns, Rietveld refinement and uncertainties; no phase percentages from peak heights."
    ),
    "ftir": (
        "Check absorbance versus transmittance, ATR versus transmission, baseline, "
        "water/CO2 contamination, reference bands and signal-to-noise."
    ),
    "raman": (
        "Check laser wavelength, irradiance, focus, heating/transformation effects, "
        "baseline, fluorescence and phase-standard reference spectra."
    ),
    "presentation": (
        "Separate measured observations, tentative hypotheses and verified conclusions. "
        "Preserve the user's original slide master and figure/source provenance."
    ),
    "literature": (
        "Separate direct quoted evidence from inference. Check opposing studies, "
        "measurement context, limitations and whether each conclusion is supported."
    ),
}


def profiles_for(techniques: list[str] | tuple[str, ...], presentation: bool = False) -> tuple[str, ...]:
    keys = sorted({str(s).lower() for s in techniques if str(s).lower() in PROFILES})
    if presentation:
        keys.append("presentation")
    if not keys:
        keys = ["literature"]
    return tuple(PROFILES[k] for k in dict.fromkeys(keys))
