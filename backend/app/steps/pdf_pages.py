from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import fitz
import numpy as np


def pixmap_to_bgr(pixmap: fitz.Pixmap) -> np.ndarray:
    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    if pixmap.n == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    if pixmap.n == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)


def render_page_bgr(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        clip=page.cropbox,
        alpha=False,
        annots=True,
    )
    return pixmap_to_bgr(pixmap)


def iter_pdf_pages(pdf_path: Path, dpi: int) -> Iterator[tuple[int, int, np.ndarray]]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Could not read PDF: {pdf_path}")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    with fitz.open(pdf_path) as document:
        page_count = document.page_count
        for index, page in enumerate(document, start=1):
            yield index, page_count, render_page_bgr(page, dpi)
