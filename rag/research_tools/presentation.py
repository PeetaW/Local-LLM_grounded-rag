"""Research slide export with editable PowerPoint text and graphs.

Use a local .pptx template for exact inherited master/layout preservation.
If no template is supplied, create a neutral fallback styled using the user's
saved Research Presentation Style v1. No claims or figures are invented.
Requires optional python-pptx and matplotlib packages, both open-source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


STYLE = {
    "name": "Shuo-Hung Research Presentation Style v1",
    "slide_width_inches": 13.333,
    "slide_height_inches": 7.5,
    "font": "Times New Roman",
    "technical_font": "Arial",
    "text_color": (0, 0, 0),
    "highlight_color": (180, 0, 0),
    "cover_title_rect": (0.6, 1.23, 10, 2.61),
    "cover_presenter_rect": (0.6, 3.94, 10, 1.81),
    "content_title_rect": (1.53, 0.15, 10.89, 0.74),
    "page_number_rect": (9.42, 6.95, 3, 0.4),
    "presenter": "Shuo-Hung, Wang",
}


@dataclass(frozen=True)
class SlideSpec:
    title: str
    bullets: tuple[str, ...] = ()
    figure_paths: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self):
        if not self.title.strip():
            raise ValueError("Slide title cannot be empty")
        if len(self.figure_paths) > 2:
            raise ValueError("Split more than two figures across slides")


@dataclass(frozen=True)
class DeckSpec:
    title: str
    slides: tuple[SlideSpec, ...]
    presenter: str = STYLE["presenter"]
    conclusion: tuple[str, ...] = ()
    future_work: tuple[str, ...] = ()
    language: str = "en"

    def __post_init__(self):
        if not self.title.strip():
            raise ValueError("Deck title cannot be empty")


def _add_text(slide, rect, text, font_size, *, bold=False):
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    x, y, w, h = rect
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    for idx, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = STYLE["font"]
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor(*STYLE["text_color"])
    return shape


def _blank_layout(presentation):
    # Template layout may have arbitrary names. Use the layout with least
    # placeholders; preserve master through template rather than rebuilding it.
    return min(presentation.slide_layouts, key=lambda layout: len(layout.placeholders))


def _slide_content(slide, spec: SlideSpec, number: int):
    _add_text(slide, STYLE["content_title_rect"], spec.title, 30, bold=True)
    if spec.figure_paths:
        from pptx.util import Inches
        pictures = tuple(Path(p) for p in spec.figure_paths)
        for picture in pictures:
            if not picture.is_file():
                raise FileNotFoundError(str(picture))
        fig_width = 5.5 if len(pictures) == 2 else 8.4
        first_x = 0.7 if len(pictures) == 2 else 2.4
        for j, picture in enumerate(pictures):
            slide.shapes.add_picture(str(picture), Inches(first_x + j * 6.0), Inches(1.45), width=Inches(fig_width))
        if spec.bullets:
            _add_text(slide, (0.9, 5.65, 11.5, 1.05), "\n".join(spec.bullets), 17)
    elif spec.bullets:
        _add_text(slide, (1.05, 1.45, 11.0, 4.85), "\n".join(spec.bullets), 22)
    _add_text(slide, STYLE["page_number_rect"], str(number), 13)
    # Evidence refs remain visible to reviewers, not silently fabricated.
    if spec.evidence_refs:
        _add_text(slide, (0.9, 6.4, 11.5, 0.45), "Sources: " + "; ".join(spec.evidence_refs), 11)


def export_research_deck(
    deck: DeckSpec,
    output_path: str | Path,
    template_path: str | Path | None = None,
) -> Path:
    from pptx import Presentation
    from pptx.util import Inches
    from pptx.dml.color import RGBColor

    if template_path:
        source = Path(template_path)
        if not source.is_file():
            raise FileNotFoundError(f"Missing master template: {source}")
        prs = Presentation(str(source))
        # Actual user-owned template is a full sample deck, not blank master.
        # Preserve its theme/master/layouts; remove example pages.
        for slide_id in list(prs.slides._sldIdLst):
            prs.part.drop_rel(slide_id.rId)
            prs.slides._sldIdLst.remove(slide_id)
    else:
        prs = Presentation()
        prs.slide_width = Inches(STYLE["slide_width_inches"])
        prs.slide_height = Inches(STYLE["slide_height_inches"])

    layout = _blank_layout(prs)
    cover = prs.slides.add_slide(layout)
    _add_text(cover, STYLE["cover_title_rect"], deck.title, 38, bold=True)
    _add_text(cover, STYLE["cover_presenter_rect"], f"Presenter: {deck.presenter}", 28)
    _add_text(cover, STYLE["page_number_rect"], "1", 13)

    for idx, spec in enumerate(deck.slides, start=2):
        slide = prs.slides.add_slide(layout)
        _slide_content(slide, spec, idx)

    if deck.conclusion or deck.future_work:
        slide = prs.slides.add_slide(layout)
        lines = tuple(deck.conclusion) + tuple("Future work: " + v for v in deck.future_work)
        _slide_content(slide, SlideSpec("Conclusions & future work", lines), len(prs.slides) + 1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    return output
