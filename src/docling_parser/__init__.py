"""Minimal PDF parsing with word-level bounding boxes, powered by docling."""

from docling_parser.easyocr_words import WordLevelEasyOcrOptions
from docling_parser.parser import (
    PageWords,
    ParsedDocument,
    WordBox,
    parse_pdf,
    set_full_page_ocr,
    word_level_ocr_options,
)
from docling_parser.rapidocr_words import WordLevelRapidOcrOptions
from docling_parser.viz import draw_word_boxes

__all__ = [
    "PageWords",
    "ParsedDocument",
    "WordBox",
    "WordLevelEasyOcrOptions",
    "WordLevelRapidOcrOptions",
    "draw_word_boxes",
    "parse_pdf",
    "set_full_page_ocr",
    "word_level_ocr_options",
]
