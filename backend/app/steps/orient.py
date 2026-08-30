"""Detect the orientation of the writing on a page and rotate it upright.

Answer sheets are sometimes scanned sideways or upside down.  CRAFT and TrOCR
both assume horizontal, right-side-up lines, so this step picks one of the
four cardinal rotations before the rest of the pipeline sees the page.

The decision is made in two independent stages on a downscaled binary copy:

1. *Axis* - which way do the writing lines run?  Lines of text make the ink
   profile across them stripe between dense and empty, so whichever of the
   two profiles varies more names the axis.  That narrows the choice to
   0°/180° or 90°/270°.
2. *Direction* - which end is up?  Latin script is top-heavy: ascenders
   (b, d, h, k, l and every capital) are taller and far more common than
   descenders (g, p, q, y).  Two consequences are scored per text line - the
   densest row sits below the line's middle, and the sharpest edge of the
   line is its lower one, because letters stop dead on the baseline while
   their tops are ragged.  Every line votes, and the rotation with the
   stronger vote wins; a near-tie leaves the page as it is.

Ruled lines are subtracted before any of this, otherwise the rules themselves
dominate every row profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.steps.binarize import binarize_image, writing_ink

_ANALYZE_LONG_SIDE = 900
_MIN_AXIS_RATIO = 1.12
# Two directions this close apart are a coin toss, so the page is left as is.
_MIN_DIRECTION_MARGIN = 0.05
_MIN_TEXT_LINES = 3
_MIN_LINE_HEIGHT = 8
_MIN_LINE_INK = 25
_SMOOTH_WINDOW = 3
_MIN_BLOBS = 8


@dataclass(frozen=True)
class Orientation:
    """Degrees of clockwise rotation needed to make the page upright."""

    degrees: int
    axis_ratio: float
    direction_margin: float

    @property
    def rotated(self) -> bool:
        return self.degrees != 0


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {degrees}")


def unrotate_point(
    x: float,
    y: float,
    degrees: int,
    orig_w: int,
    orig_h: int,
) -> tuple[float, float]:
    """Map a point from the upright image back onto the original render."""
    if degrees == 0:
        return x, y
    if degrees == 90:
        return y, orig_h - 1 - x
    if degrees == 180:
        return orig_w - 1 - x, orig_h - 1 - y
    if degrees == 270:
        return orig_w - 1 - y, x
    raise ValueError(f"Unsupported rotation: {degrees}")


def unrotate_rect(
    rect: tuple[int, int, int, int],
    degrees: int,
    orig_w: int,
    orig_h: int,
) -> tuple[float, float, float, float]:
    """Map ``(x, y, w, h)`` from the upright image back to original pixels."""
    x, y, width, height = rect
    corners = (
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    )
    mapped = [unrotate_point(px, py, degrees, orig_w, orig_h) for px, py in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return min(xs), min(ys), max(xs), max(ys)


def _downscale(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= _ANALYZE_LONG_SIDE:
        return image
    scale = _ANALYZE_LONG_SIDE / long_side
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _ink_mask(binary: np.ndarray) -> np.ndarray:
    """Ink as a 0/1 byte image."""
    return (binary < 128).astype(np.uint8)


def _banding_score(ink: np.ndarray) -> float:
    """How strongly ink forms horizontal bands (typical of upright lines)."""
    if ink.size == 0:
        return 0.0
    rows = ink.sum(axis=1).astype(np.float64)
    mean = float(rows.mean())
    if mean < 1.0:
        return 0.0
    return float(rows.std() / mean)


def _usable(strip: np.ndarray) -> bool:
    return strip.shape[0] >= _MIN_LINE_HEIGHT and float(strip.sum()) >= _MIN_LINE_INK


def _text_boxes(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of ink blobs that look like letters or short words."""
    height, width = ink.shape
    area = height * width
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, num):
        x, y, w, h, blob_area = stats[index]
        if blob_area < 12 or blob_area > 0.03 * area:
            continue
        if w < 2 or h < 2:
            continue
        if w > 0.45 * width or h > 0.45 * height:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def _cluster_rows(
    boxes: list[tuple[int, int, int, int]],
) -> list[list[tuple[int, int, int, int]]]:
    """Group blobs that share a vertical centre into text lines."""
    if not boxes:
        return []
    median_h = float(np.median([h for _x, _y, _w, h in boxes]))
    tol = max(6.0, median_h * 0.7)
    rows: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda item: item[1]):
        _x, y, _w, h = box
        mid = y + h / 2
        for row in rows:
            _rx, ry, _rw, rh = row[0]
            if abs(mid - (ry + rh / 2)) <= tol:
                row.append(box)
                break
        else:
            rows.append([box])
    return [row for row in rows if len(row) >= 2]


def _blob_line_strips(writing: np.ndarray) -> list[np.ndarray]:
    """Line images cut out around clusters of letter-sized blobs."""
    strips: list[np.ndarray] = []
    for row in _cluster_rows(_text_boxes(writing)):
        top = min(y for _x, y, _w, _h in row)
        bottom = max(y + h for _x, y, _w, h in row)
        left = min(x for x, _y, _w, _h in row)
        right = max(x + w for x, _y, w, _h in row)
        strip = writing[top:bottom, left:right]
        if _usable(strip):
            strips.append(strip)
    return strips


def _profile_line_strips(writing: np.ndarray) -> list[np.ndarray]:
    """Line images cut at the empty rows of the page ink profile."""
    profile = writing.sum(axis=1).astype(np.float64)
    if profile.max() <= 0:
        return []
    inked = profile > 0.08 * profile.max()

    strips: list[np.ndarray] = []
    start: int | None = None
    for y, flag in enumerate(inked):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            strips.append(writing[start:y])
            start = None
    if start is not None:
        strips.append(writing[start:])
    return [strip for strip in strips if _usable(strip)]


def _line_profile(strip: np.ndarray) -> np.ndarray:
    """Ink per row of a text line, lightly smoothed."""
    profile = strip.sum(axis=1).astype(np.float64)
    if len(profile) >= _SMOOTH_WINDOW:
        window = np.ones(_SMOOTH_WINDOW) / _SMOOTH_WINDOW
        profile = np.convolve(profile, window, mode="same")
    return profile


def _peak_offset(profile: np.ndarray) -> float:
    """Where the densest row of a line sits: -0.5 is its top, +0.5 its bottom.

    The x-height band is the densest part of a line and lies just above the
    baseline, so on upright text it falls below the middle.
    """
    return int(np.argmax(profile)) / len(profile) - 0.5


def _edge_offset(profile: np.ndarray) -> float:
    """Positive when the line's sharpest edge is its lower one.

    Letters rest exactly on the baseline, so ink stops abruptly there, while
    the upper edge is ragged because ascenders vary in height.
    """
    steps = np.diff(profile)
    if steps.size == 0:
        return 0.0
    fall = abs(float(steps.min()))
    rise = abs(float(steps.max()))
    total = fall + rise
    if total <= 0:
        return 0.0
    return (fall - rise) / total


def _vote(offsets: list[float]) -> float:
    if not offsets:
        return 0.0
    below = sum(1 for offset in offsets if offset > 0)
    above = sum(1 for offset in offsets if offset < 0)
    return (below - above) / len(offsets)


def _upright_confidence(binary: np.ndarray) -> float:
    """Net vote in [-1, 1] for this orientation being right-side up.

    Both line segmentations are pooled: blob clustering copes with sparse or
    slanted writing, the ink profile copes with lines that run together.
    """
    writing = writing_ink(binary)
    strips = _blob_line_strips(writing) + _profile_line_strips(writing)
    if len(strips) < _MIN_TEXT_LINES:
        return 0.0
    profiles = [_line_profile(strip) for strip in strips]
    peak = _vote([_peak_offset(profile) for profile in profiles])
    edge = _vote([_edge_offset(profile) for profile in profiles])
    return (peak + edge) / 2.0


def _text_axis(ink: np.ndarray) -> tuple[bool, float]:
    """Whether the writing runs vertically, and how clear-cut that call is."""
    horizontal_band = _banding_score(ink)
    vertical_band = _banding_score(np.rot90(ink, k=1))
    if vertical_band > horizontal_band * _MIN_AXIS_RATIO:
        return True, vertical_band / max(horizontal_band, 1e-6)
    return False, horizontal_band / max(vertical_band, 1e-6)


def estimate_orientation(image: np.ndarray) -> Orientation:
    """Return the clockwise rotation that makes handwriting read left-to-right."""
    sample = binarize_image(_downscale(image))
    ink = _ink_mask(sample)
    if len(_text_boxes(ink)) < _MIN_BLOBS:
        return Orientation(0, 1.0, 0.0)

    vertical_axis, axis_ratio = _text_axis(ink)
    upright, upside_down = (90, 270) if vertical_axis else (0, 180)

    keep = _upright_confidence(rotate_image(sample, upright))
    flip = _upright_confidence(rotate_image(sample, upside_down))
    margin = abs(keep - flip)
    if flip - keep > _MIN_DIRECTION_MARGIN:
        return Orientation(upside_down, axis_ratio, margin)
    return Orientation(upright, axis_ratio, margin)


def upright_page(image: np.ndarray) -> tuple[np.ndarray, Orientation]:
    """Rotate ``image`` so text runs horizontally, right-side up."""
    orientation = estimate_orientation(image)
    return rotate_image(image, orientation.degrees), orientation
