"""Create slide specifications from checked, structured research discussion.

A language model may suggest topics, but this deterministic renderer accepts
only supplied evidence-bearing findings. This prevents claiming that an LLM's
interpretation of XRD/Raman/FTIR is a verified phase identification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .presentation import DeckSpec, SlideSpec


@dataclass(frozen=True)
class ResearchFinding:
    topic: str
    observation: str
    interpretation: str = ""
    evidence_refs: tuple[str, ...] = ()
    status: str = "observed"  # observed | tentative | validated
    figure_paths: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.topic.strip() or not self.observation.strip():
            raise ValueError("Finding requires topic and observation")
        if self.status not in {"observed", "tentative", "validated"}:
            raise ValueError("Unsupported research finding status")
        if self.status == "validated" and not self.evidence_refs:
            raise ValueError("Validated findings need traceable evidence_refs")


def plan_deck_from_discussion(
    title: str,
    findings: Sequence[ResearchFinding],
    *,
    next_steps: Sequence[str] = (),
    unresolved_questions: Sequence[str] = (),
    include_supplement: bool = False,
) -> DeckSpec:
    """Convert researched conclusions to editable slides; never invent evidence."""
    if not findings:
        raise ValueError("At least one structured finding required")
    slides = []
    for item in findings:
        bullets = [f"Observation: {item.observation}"]
        if item.interpretation:
            label = "Tentative interpretation" if item.status != "validated" else "Interpretation"
            bullets.append(f"{label}: {item.interpretation}")
        if not item.evidence_refs:
            bullets.append("Evidence status: reference/data source not supplied")
        slides.append(SlideSpec(
            title=item.topic,
            bullets=tuple(bullets),
            figure_paths=item.figure_paths,
            evidence_refs=item.evidence_refs,
        ))
    conclusions = tuple(
        f"{f.topic}: {f.interpretation or f.observation}"
        for f in findings if f.status == "validated"
    )
    if unresolved_questions:
        slides.append(SlideSpec("Current questions", tuple(unresolved_questions)))
    if include_supplement:
        slides.append(SlideSpec("Supplement data"))
    return DeckSpec(
        title=title,
        slides=tuple(slides),
        conclusion=conclusions,
        future_work=tuple(next_steps),
    )
