from __future__ import annotations

import cv2
import numpy as np


def binarize_image(
    image: np.ndarray,
    block_size: int = 51,
    c: int = 22,
) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary[gray > max(0, int(otsu_val) - 8)] = 255
    return binary


def writing_ink(binary: np.ndarray) -> np.ndarray:
    """Ink left over once the printed ruling of the paper is removed.

    Returned as a 0/1 mask.  An opening keeps only strokes that run for a
    tenth of the page, which handwriting never does but a ruled line or a
    margin always does.  Both directions are cleaned, so the result does not
    depend on which way round the page happens to be.
    """
    ink = (binary < 128).astype(np.uint8)
    height, width = ink.shape[:2]
    for size, pad in (
        ((max(25, width // 10), 1), (1, 3)),
        ((1, max(25, height // 10)), (3, 1)),
    ):
        rules = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size)
        )
        thick = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, pad))
        ink = cv2.subtract(ink, thick)
    return ink
