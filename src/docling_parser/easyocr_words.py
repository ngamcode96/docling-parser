"""Word-level EasyOCR engine, plugged into docling's OCR factory.

docling's built-in ``EasyOcrModel`` calls ``reader.readtext(im)`` with no
keyword arguments (``docling/models/stages/ocr/easyocr_model.py:219``), so
EasyOCR's default ``width_ths=0.5`` applies and horizontally adjacent
detections get merged into text *lines*. EasyOCR itself detects words: with
``width_ths=0.0`` the merge step is disabled and every box is a word, with a
box measured by the detector rather than estimated from character offsets.

This module subclasses the docling model, forwards the detection thresholds,
and registers it under the kind ``"easyocr_words"``.
"""

from __future__ import annotations

from functools import partial
from typing import ClassVar, Literal, Type

from docling.datamodel.pipeline_options import EasyOcrOptions
from docling.models.factories import get_ocr_factory
from docling.models.stages.ocr.easyocr_model import EasyOcrModel
from docling_parser.word_cells import PublishWordCellsMixin

_PLUGIN_NAME = "docling_parser"


class WordLevelEasyOcrOptions(EasyOcrOptions):
    """``EasyOcrOptions`` plus EasyOCR's box-grouping thresholds.

    The defaults are EasyOCR's own, except ``width_ths``, which is what turns
    line boxes into word boxes. Raise it towards 0.5 to merge words back into
    lines.
    """

    kind: ClassVar[Literal["easyocr_words"]] = "easyocr_words"  # type: ignore[assignment]

    # Per-word confidences run lower than per-line ones (a lone "a" or "1.0"
    # scores ~0.3), so docling's 0.5 cutoff would silently drop short words.
    confidence_threshold: float = 0.1

    width_ths: float = 0.0
    ycenter_ths: float = 0.5
    height_ths: float = 0.5
    slope_ths: float = 0.1

    # EasyOCR's own default is 0.1, which pads every box outwards. 0.05 trades a
    # little recognition accuracy for noticeably tighter boxes; 0.0 is tighter
    # still, 0.1 recognizes best. See the table in the README.
    add_margin: float = 0.05


class WordLevelEasyOcrModel(PublishWordCellsMixin, EasyOcrModel):
    """``EasyOcrModel`` that forwards the grouping thresholds to ``readtext``."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if not self.enabled:
            return

        options: WordLevelEasyOcrOptions = self.options  # type: ignore[assignment]
        self.reader.readtext = partial(
            self.reader.readtext,
            width_ths=options.width_ths,
            ycenter_ths=options.ycenter_ths,
            height_ths=options.height_ths,
            slope_ths=options.slope_ths,
            add_margin=options.add_margin,
        )

    @classmethod
    def get_options_type(cls) -> Type[EasyOcrOptions]:
        return WordLevelEasyOcrOptions


def register() -> None:
    """Make ``WordLevelEasyOcrOptions`` usable by docling's pipeline.

    Registration is done directly on the factory instead of via a setuptools
    entry point, because docling skips entry points from modules outside the
    ``docling.`` namespace unless ``allow_external_plugins=True``.
    """
    for external in (False, True):
        factory = get_ocr_factory(allow_external_plugins=external)
        if WordLevelEasyOcrOptions not in factory.classes:
            factory.register(
                WordLevelEasyOcrModel, _PLUGIN_NAME, WordLevelEasyOcrModel.__module__
            )


register()


__all__ = [
    "WordLevelEasyOcrModel",
    "WordLevelEasyOcrOptions",
    "register",
]
