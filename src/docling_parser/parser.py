"""Parse PDFs with docling and return word-level bounding boxes.

Docling exposes three cell granularities per page on ``parsed_page``:
``char_cells``, ``word_cells`` and ``textline_cells``. Digital (born-PDF)
text comes with real ``word_cells``; OCR engines only emit text *lines*, so
for OCR'd regions the words are cut out of the line box proportionally to
their character offsets and flagged with ``estimated=True``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.page import TextCell
from PIL.Image import Image

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class WordBox:
    """One word and its box, in PDF points with a top-left origin."""

    page_no: int
    text: str
    left: float
    top: float
    right: float
    bottom: float
    confidence: float = 1.0
    from_ocr: bool = False
    estimated: bool = False

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def scaled(self, scale: float) -> tuple[float, float, float, float]:
        """Box as ``(left, top, right, bottom)`` in pixels of a rendered page."""
        return (
            self.left * scale,
            self.top * scale,
            self.right * scale,
            self.bottom * scale,
        )

    def as_dict(self) -> dict:
        return {
            "page_no": self.page_no,
            "text": self.text,
            "bbox": [self.left, self.top, self.right, self.bottom],
            "confidence": self.confidence,
            "from_ocr": self.from_ocr,
            "estimated": self.estimated,
        }


@dataclass
class PageWords:
    """A single page: its size, its rendering and its words."""

    page_no: int
    width: float
    height: float
    words: list[WordBox] = field(default_factory=list)
    image: Image | None = None

    @property
    def image_scale(self) -> float:
        """Pixels per PDF point of :attr:`image` (1.0 when no image)."""
        if self.image is None:
            return 1.0
        return self.image.width / self.width


@dataclass
class ParsedDocument:
    """Result of :func:`parse_pdf`."""

    document: DoclingDocument
    pages: list[PageWords]
    conversion: ConversionResult

    @property
    def words(self) -> Iterator[WordBox]:
        for page in self.pages:
            yield from page.words

    def as_dict(self) -> dict:
        return {
            "name": self.document.name,
            "pages": [
                {
                    "page_no": page.page_no,
                    "width": page.width,
                    "height": page.height,
                    "words": [w.as_dict() for w in page.words],
                }
                for page in self.pages
            ],
        }


def set_full_page_ocr(ocr_options) -> None:
    """Switch ``ocr_options`` to OCR the whole page, on any docling version.

    ``OcrMode`` only exists from docling 2.116.0; before that the same thing was
    a boolean, which newer versions keep as a deprecated alias.
    """
    try:
        from docling.datamodel.pipeline_options import OcrMode
    except ImportError:  # docling < 2.116
        ocr_options.force_full_page_ocr = True
    else:
        ocr_options.mode = OcrMode.FULL_PAGE


def word_level_ocr_options(engine: str = "easyocr", **kwargs):
    """Options for an OCR engine configured to return one box per word.

    ``engine`` is ``"easyocr"`` (EasyOCR with box merging disabled) or
    ``"rapidocr"`` (RapidOCR with ``return_word_box``). Extra keyword arguments
    go to the options class, e.g. ``lang=["en"]``.
    """
    if engine == "easyocr":
        from docling_parser.easyocr_words import WordLevelEasyOcrOptions

        return WordLevelEasyOcrOptions(**kwargs)
    if engine == "rapidocr":
        from docling_parser.rapidocr_words import WordLevelRapidOcrOptions

        return WordLevelRapidOcrOptions(**kwargs)
    raise ValueError(
        f"unknown ocr_engine {engine!r}, expected 'easyocr' or 'rapidocr'"
    )


def parse_pdf(
    source: str | Path,
    *,
    ocr: bool = True,
    ocr_engine: str = "easyocr",
    ocr_options=None,
    word_level_ocr: bool = True,
    force_full_page_ocr: bool = False,
    image_scale: float = 2.0,
    generate_page_images: bool = True,
    page_range: tuple[int, int] | None = None,
    do_table_structure: bool = True,
) -> ParsedDocument:
    """Convert ``source`` with docling and collect word-level boxes.

    Args:
        source: path or URL of the PDF (anything docling accepts).
        ocr: run OCR.
        ocr_engine: ``"easyocr"`` or ``"rapidocr"``. Ignored when
            ``ocr_options`` is given, or when ``word_level_ocr`` is False (then
            docling auto-selects).
        ocr_options: explicit docling OCR options, e.g. ``EasyOcrOptions()``
            or ``TesseractCliOcrOptions()``. Overrides the two settings above.
        word_level_ocr: ask the engine for one *measured* box per word instead
            of per line. Set to False to let docling auto-select the engine;
            OCR boxes are then lines, cut into ``estimated`` words.
        force_full_page_ocr: OCR the whole page even when it has digital text.
            What you want for PDFs whose embedded text layer is broken.
        image_scale: page rendering scale (2.0 ~= 144 DPI) used for the images
            returned on each page.
        page_range: 1-based inclusive ``(first, last)`` page range.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = ocr
    pipeline_options.do_table_structure = do_table_structure
    # Required: without it docling drops the per-page cell layers we need.
    pipeline_options.generate_parsed_pages = True
    pipeline_options.generate_page_images = generate_page_images
    pipeline_options.images_scale = image_scale

    if ocr_options is not None:
        pipeline_options.ocr_options = ocr_options
    elif ocr and word_level_ocr:
        pipeline_options.ocr_options = word_level_ocr_options(ocr_engine)
    if force_full_page_ocr:
        set_full_page_ocr(pipeline_options.ocr_options)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    kwargs = {"page_range": page_range} if page_range else {}
    result = converter.convert(str(source), **kwargs)

    pages = [_collect_page(page) for page in result.pages]
    return ParsedDocument(document=result.document, pages=pages, conversion=result)


def _collect_page(page) -> PageWords:
    height = page.size.height if page.size else 0.0
    width = page.size.width if page.size else 0.0
    page_no = page.page_no  # 1-based, matches DoclingDocument.pages keys

    words: list[WordBox] = []
    parsed = page.parsed_page
    if parsed is not None:
        # 1. Word boxes: from the PDF text layer, plus whatever a word-level OCR
        #    engine published there. A cell holding more than one token still
        #    gets split (word-level OCR occasionally merges two words).
        for cell in parsed.word_cells:
            words.extend(_split_line(cell, page_no, height))
        # 2. Line-level OCR: cut the lines covering areas without word boxes
        #    (scanned pages, or a figure inside a digital page) into words.
        for cell in parsed.textline_cells:
            if not cell.from_ocr:
                continue
            line = _top_left_bbox(cell, height)
            if any(_covered(line, w) for w in words):
                continue
            words.extend(_split_line(cell, page_no, height))

    return PageWords(
        page_no=page_no,
        width=width,
        height=height,
        words=words,
        image=page.image,
    )


def _top_left_bbox(cell: TextCell, page_height: float):
    bbox = cell.rect.to_bounding_box()
    if bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
        bbox = bbox.to_top_left_origin(page_height)
    return bbox


def _covered(line, word: WordBox) -> bool:
    """True if ``word`` sits (mostly) inside the line box ``line``."""
    overlap_x = min(line.r, word.right) - max(line.l, word.left)
    overlap_y = min(line.b, word.bottom) - max(line.t, word.top)
    if overlap_x <= 0 or overlap_y <= 0:
        return False
    area = max(word.width * word.height, 1e-6)
    return (overlap_x * overlap_y) / area > 0.5


def _word_from_cell(cell: TextCell, page_no: int, page_height: float) -> WordBox:
    bbox = _top_left_bbox(cell, page_height)
    return WordBox(
        page_no=page_no,
        text=cell.text.strip(),
        left=bbox.l,
        top=bbox.t,
        right=bbox.r,
        bottom=bbox.b,
        confidence=cell.confidence,
        from_ocr=cell.from_ocr,
        estimated=False,
    )


def _split_line(cell: TextCell, page_no: int, page_height: float) -> list[WordBox]:
    """Slice a text line into words, proportionally to character offsets.

    A cell holding a single token is already a word box (that is what the
    word-level EasyOCR engine returns), so it is kept as measured.
    """
    text = cell.text
    if not text.strip():
        return []

    matches = list(_WORD_RE.finditer(text))
    if len(matches) == 1:
        return [_word_from_cell(cell, page_no, page_height)]

    bbox = _top_left_bbox(cell, page_height)
    n_chars = len(text)
    span = bbox.r - bbox.l

    words: list[WordBox] = []
    for match in matches:
        start, end = match.span()
        words.append(
            WordBox(
                page_no=page_no,
                text=match.group(),
                left=bbox.l + span * start / n_chars,
                top=bbox.t,
                right=bbox.l + span * end / n_chars,
                bottom=bbox.b,
                confidence=cell.confidence,
                from_ocr=cell.from_ocr,
                estimated=True,
            )
        )
    return words
