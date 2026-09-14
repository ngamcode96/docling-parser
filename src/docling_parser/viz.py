"""Draw word boxes on the page images docling renders."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from docling_parser.parser import PageWords

_NATIVE = "#1a7f37"  # word box read from the PDF text layer
_OCR = "#d1242f"  # word box coming from (or estimated within) OCR


def draw_word_boxes(
    page: PageWords,
    *,
    show_text: bool = False,
    width: int = 1,
    fill_alpha: int = 40,
) -> Image.Image:
    """Return a copy of the page image with one rectangle per word."""
    if page.image is None:
        raise ValueError(
            f"page {page.page_no} has no image; "
            "parse with generate_page_images=True"
        )

    base = page.image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale = page.image_scale
    font = ImageFont.load_default()

    for word in page.words:
        left, top, right, bottom = word.scaled(scale)
        color = _OCR if (word.from_ocr or word.estimated) else _NATIVE
        rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
        draw.rectangle(
            (left, top, right, bottom),
            outline=(*rgb, 255),
            fill=(*rgb, fill_alpha),
            width=width,
        )
        if show_text:
            draw.text((left, max(0.0, top - 9)), word.text, fill=(*rgb, 255), font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")
