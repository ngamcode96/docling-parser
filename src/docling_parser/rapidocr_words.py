"""Word-level RapidOCR engine, plugged into docling's OCR factory.

docling's ``RapidOcrModel`` calls ``self.reader(im, use_det=..., use_cls=...,
use_rec=...)`` (``docling/models/stages/ocr/rapid_ocr_model.py:518``) and then
reads ``result.boxes`` / ``.txts`` / ``.scores``, which are *text lines*:
RapidOCR's DBNet detector is trained on line regions, so unlike EasyOCR there is
no grouping step to switch off.

RapidOCR does expose word boxes through ``return_word_box=True``: the
recognizer splits each line using the CTC alignment of its own output and fills
``result.word_results`` with ``(text, score, quad)`` per word. This module wraps
the reader so that flag is set and the word results are flattened back into
``boxes``/``txts``/``scores`` — docling's own cell-building loop then emits one
``TextCell`` per word, untouched.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Type

import numpy

from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.factories import get_ocr_factory
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling_parser.word_cells import PublishWordCellsMixin

_PLUGIN_NAME = "docling_parser"


class WordLevelRapidOcrOptions(RapidOcrOptions):
    """``RapidOcrOptions`` that asks RapidOCR for per-word boxes."""

    kind: ClassVar[Literal["rapidocr_words"]] = "rapidocr_words"  # type: ignore[assignment]

    single_char_box: bool = False
    """Split down to single characters instead of words."""


class _WordLevelReader:
    """Wraps a ``RapidOCR`` instance so it returns words instead of lines."""

    def __init__(self, reader: Any, single_char_box: bool = False) -> None:
        self._reader = reader
        self._single_char_box = single_char_box

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reader, name)

    def __call__(self, img, **kwargs) -> Any:
        result = self._reader(
            img,
            return_word_box=True,
            return_single_char_box=self._single_char_box,
            **kwargs,
        )
        if result is None or result.boxes is None or not result.word_results:
            return result

        boxes, txts, scores = [], [], []
        for line in result.word_results:
            for word in line or ():
                # (text, score, quad) — any of them can come back empty
                if not word or len(word) < 3 or word[2] is None:
                    continue
                text = word[0]
                if text is None or not str(text).strip():
                    continue
                boxes.append(word[2])
                txts.append(str(text))
                scores.append(float(word[1]) if word[1] is not None else 1.0)

        if not boxes:
            return result

        result.boxes = numpy.array(boxes)
        result.txts = tuple(txts)
        result.scores = tuple(scores)
        return result


class WordLevelRapidOcrModel(PublishWordCellsMixin, RapidOcrModel):
    """``RapidOcrModel`` whose reader yields word boxes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if not self.enabled:
            return

        options: WordLevelRapidOcrOptions = self.options  # type: ignore[assignment]
        self.reader = _WordLevelReader(
            self.reader, single_char_box=options.single_char_box
        )

    @classmethod
    def get_options_type(cls) -> Type[RapidOcrOptions]:
        return WordLevelRapidOcrOptions


def register() -> None:
    """Make ``WordLevelRapidOcrOptions`` usable by docling's pipeline."""
    for external in (False, True):
        factory = get_ocr_factory(allow_external_plugins=external)
        if WordLevelRapidOcrOptions not in factory.classes:
            factory.register(
                WordLevelRapidOcrModel, _PLUGIN_NAME, WordLevelRapidOcrModel.__module__
            )


register()


__all__ = [
    "WordLevelRapidOcrModel",
    "WordLevelRapidOcrOptions",
    "register",
]
