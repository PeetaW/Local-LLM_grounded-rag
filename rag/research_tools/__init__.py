"""Local, opt-in scientific file tools. Imports do not run analyses."""
from .spectroscopy import Spectrum, SpectrumResult, load_spectrum, analyze_spectrum, compare_spectra
__all__ = ["Spectrum", "SpectrumResult", "load_spectrum", "analyze_spectrum", "compare_spectra"]
