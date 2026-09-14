"""Shared behaviour for the word-level OCR engines.

docling's OCR post-processing writes every OCR cell to
``parsed_page.textline_cells`` and never to ``word_cells``
(``docling/models/base_ocr_model.py:385``), because for its built-in engines
the cells really are lines — in ``OcrMode.FULL_PAGE`` it even clears
``word_cells`` on purpose.

The engines in this package return one cell per *word*, so ``word_cells`` is
where a caller would expect to find them. This mixin copies them there after
docling's own post-processing, leaving ``textline_cells`` untouched.
"""

from __future__ import annotations

from typing import Any


class PublishWordCellsMixin:
    """Mirror OCR word cells into ``parsed_page.word_cells``."""

    def post_process_cells(self, ocr_cells, page, conv_res, priority=None) -> None:
        super().post_process_cells(ocr_cells, page, conv_res, priority)  # type: ignore[misc]

        parsed: Any = page.parsed_page
        if parsed is None:
            return

        # Word boxes read from the PDF text layer survive outside FULL_PAGE mode
        # and stay authoritative; the OCR words are appended to them.
        native = [cell for cell in parsed.word_cells if not cell.from_ocr]
        ocr_words = [cell for cell in parsed.textline_cells if cell.from_ocr]

        parsed.word_cells = native + ocr_words
        parsed.has_words = bool(parsed.word_cells)


__all__ = ["PublishWordCellsMixin"]
