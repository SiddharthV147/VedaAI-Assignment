from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models.vgg as vgg_module

if not hasattr(vgg_module, "model_urls"):
    vgg_module.model_urls = {
        "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
        "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
        "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
        "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
        "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
        "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
    }

from craft_text_detector import load_craftnet_model, load_refinenet_model
import craft_text_detector.craft_utils as craft_utils
import craft_text_detector.image_utils as image_utils
import craft_text_detector.torch_utils as torch_utils

from app.config import use_cuda
from app.steps.binarize import writing_ink

CRAFT_WEIGHT = "craft_mlt_25k.pth"
REFINER_WEIGHT = "craft_refiner_CTW1500.pth"


def _model_on_cuda(net) -> bool:
    try:
        return next(net.parameters()).is_cuda
    except StopIteration:
        return False


def _scale_boxes(boxes, ratio_w, ratio_h, ratio_net=2):
    scaled = []
    for box in boxes:
        if box is None:
            continue
        scaled.append(
            np.asarray(box, dtype=np.float32) * (ratio_w * ratio_net, ratio_h * ratio_net)
        )
    return scaled


def _box_to_rect(box: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.asarray(box, dtype=np.int32).reshape(-1, 2)
    x, y, width, height = cv2.boundingRect(points)
    if width < 4 or height < 4:
        return None
    return x, y, width, height


def _rect_to_box(rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = rect
    return np.array(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float32,
    )


def _inked_row_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive-exclusive (top, bottom) runs of rows that still hold ink."""
    if mask.size == 0:
        return []
    profile = mask.sum(axis=1).astype(np.float64)
    if len(profile) >= 5:
        profile = np.convolve(profile, np.ones(3) / 3.0, mode="same")
    peak = float(profile.max())
    if peak <= 0:
        return []
    on = profile > max(2.0, 0.06 * peak)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for y, flag in enumerate(on):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, len(on)))
    return runs


def _text_row_bands(ink: np.ndarray) -> list[tuple[int, int]]:
    """Horizontal bands of writing, broken at empty ruled rows.

    Ticks above ``(A)`` and dots on ``i`` are short ink runs.  They are
    attached to the nearest real line.  A skipped notebook row between
    answers is taller than a character and stays as a split.

    Bands are computed from the crop that is passed in, not the whole
    page: another column of writing would otherwise fill the blank rows.
    """
    runs = _inked_row_runs(ink)
    if not runs:
        return []
    heights = [bottom - top for top, bottom in runs]
    tallest = max(heights)
    substantial = [h for h in heights if h >= max(12, 0.25 * tallest)]
    line_h = float(np.median(substantial or heights))

    # Close broken strokes inside one letter (not ticks, not blank rows).
    close = max(4, int(line_h * 0.12))
    merged: list[list[int]] = [list(runs[0])]
    for top, bottom in runs[1:]:
        if top - merged[-1][1] <= close:
            merged[-1][1] = bottom
        else:
            merged.append([top, bottom])

    text: list[list[int]] = []
    fragments: list[list[int]] = []
    min_text_h = max(14, int(0.35 * line_h))
    for top, bottom in merged:
        if bottom - top >= min_text_h:
            text.append([top, bottom])
        else:
            fragments.append([top, bottom])

    # Prefer the line below a fragment: ticks sit above ``(A)``.
    attach_lim = int(line_h * 0.85)
    for ft, fb in fragments:
        if not text:
            continue
        best_i = None
        best_d = 10**9
        for i, (tt, tb) in enumerate(text):
            if fb <= tt:
                dist = tt - fb
            elif ft >= tb:
                dist = ft - tb
            else:
                best_i, best_d = i, 0
                break
            if dist < best_d:
                best_i, best_d = i, dist
        if best_i is not None and best_d <= attach_lim:
            text[best_i][0] = min(text[best_i][0], ft)
            text[best_i][1] = max(text[best_i][1], fb)

    if not text:
        return []
    text.sort()
    out = [text[0]]
    join = max(6, int(line_h * 0.20))
    for top, bottom in text[1:]:
        if top - out[-1][1] <= join:
            out[-1][1] = max(out[-1][1], bottom)
        else:
            out.append([top, bottom])
    return [(top, bottom) for top, bottom in out]


def split_boxes_by_blank_rows(
    boxes: list[np.ndarray],
    ink: np.ndarray,
) -> list[np.ndarray]:
    """Cut a box that covers several writing lines at the empty rows between them."""
    if ink.size == 0 or not boxes:
        return boxes

    split: list[np.ndarray] = []
    for box in boxes:
        rect = _box_to_rect(box)
        if rect is None:
            continue
        x, y, width, height = rect
        local = _text_row_bands(ink[y : y + height, x : x + width])
        if len(local) <= 1:
            split.append(box)
            continue
        for top, bottom in local:
            ny, nb = y + top, y + bottom
            if nb - ny < 4:
                continue
            patch = ink[ny:nb, x : x + width]
            if patch.size == 0 or int(patch.sum()) < 8:
                continue
            split.append(_rect_to_box((x, ny, width, nb - ny)))
    return split or boxes


def merge_boxes_into_lines(
    boxes: list[np.ndarray],
    gap_ratio: float = 2.2,
    overlap_ratio: float = 0.4,
) -> list[np.ndarray]:
    rects = [rect for rect in (_box_to_rect(box) for box in boxes) if rect is not None]
    if not rects:
        return []

    median_h = float(np.median([height for _, _, _, height in rects]))
    max_gap = max(12.0, median_h * gap_ratio)

    lines: list[list[tuple[int, int, int, int]]] = []
    for rect in sorted(rects, key=lambda item: (item[1], item[0])):
        _, y, _, height = rect
        placed = False
        for line in lines:
            _rx, ref_y, _rw, ref_h = line[0]
            overlap = min(ref_y + ref_h, y + height) - max(ref_y, y)
            if overlap >= overlap_ratio * min(ref_h, height):
                line.append(rect)
                placed = True
                break
        if not placed:
            lines.append([rect])

    merged: list[np.ndarray] = []
    for line in lines:
        line.sort(key=lambda item: item[0])
        current = list(line[0])
        for nxt in line[1:]:
            gap = nxt[0] - (current[0] + current[2])
            if gap <= max_gap:
                x1 = min(current[0], nxt[0])
                y1 = min(current[1], nxt[1])
                x2 = max(current[0] + current[2], nxt[0] + nxt[2])
                y2 = max(current[1] + current[3], nxt[1] + nxt[3])
                current = [x1, y1, x2 - x1, y2 - y1]
            else:
                merged.append(
                    _rect_to_box((current[0], current[1], current[2], current[3]))
                )
                current = list(nxt)
        merged.append(_rect_to_box((current[0], current[1], current[2], current[3])))
    return merged


def drop_blank_boxes(
    boxes: list[np.ndarray],
    ink: np.ndarray,
    min_ink_ratio: float,
) -> list[np.ndarray]:
    """Discard boxes holding too little ink to be writing.

    CRAFT answers on blank ruled paper now and then, and the recogniser turns
    those empty crops into invented text, so they are cheaper to drop here.
    """
    if min_ink_ratio <= 0:
        return boxes
    kept: list[np.ndarray] = []
    for box in boxes:
        rect = _box_to_rect(box)
        if rect is None:
            continue
        x, y, width, height = rect
        patch = ink[max(0, y) : y + height, max(0, x) : x + width]
        if patch.size and float(patch.sum()) / patch.size >= min_ink_ratio:
            kept.append(box)
    return kept


def predict_boxes(
    image: np.ndarray,
    craft_net,
    refine_net,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
    long_size: int,
    min_ink_ratio: float = 0.0,
) -> list[np.ndarray]:
    image = image_utils.read_image(image)
    img_resized, target_ratio, _ = image_utils.resize_aspect_ratio(
        image, long_size, interpolation=cv2.INTER_CUBIC
    )
    ratio = 1 / target_ratio

    x = image_utils.normalizeMeanVariance(img_resized)
    x = torch_utils.from_numpy(x).permute(2, 0, 1)
    x = torch_utils.Variable(x.unsqueeze(0))
    if _model_on_cuda(craft_net):
        x = x.cuda()
    with torch_utils.no_grad():
        y, feature = craft_net(x)
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    if refine_net is not None:
        with torch_utils.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    boxes, _ = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly=False
    )
    boxes = _scale_boxes(boxes, ratio, ratio)

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ink = writing_ink(gray)
    # Blanks are dropped before merging too, so an empty box cannot stretch a
    # real line box across the page.
    boxes = drop_blank_boxes(boxes, ink, min_ink_ratio)
    boxes = split_boxes_by_blank_rows(boxes, ink)
    boxes = merge_boxes_into_lines(boxes)
    # Merge can re-stack two lines if their boxes overlapped; cut again.
    boxes = split_boxes_by_blank_rows(boxes, ink)
    return drop_blank_boxes(boxes, ink, min_ink_ratio)


def require_weight(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing weight file: {path}\n"
            "Download weights with: python download_models.py -o models"
        )
    return path


def load_detector(models_dir: Path):
    craft_path = require_weight(models_dir / CRAFT_WEIGHT)
    refiner_path = require_weight(models_dir / REFINER_WEIGHT)
    cuda = use_cuda()
    craft_net = load_craftnet_model(cuda=cuda, weight_path=str(craft_path))
    refine_net = load_refinenet_model(cuda=cuda, weight_path=str(refiner_path))
    return craft_net, refine_net


def detect_boxes(
    image: np.ndarray,
    models_dir: Path,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
    long_size: int,
    min_ink_ratio: float = 0.0,
    craft_net=None,
    refine_net=None,
) -> list[np.ndarray]:
    if craft_net is None or refine_net is None:
        craft_net, refine_net = load_detector(models_dir)
    return predict_boxes(
        image,
        craft_net,
        refine_net,
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        long_size=long_size,
        min_ink_ratio=min_ink_ratio,
    )


def draw_bounding_boxes(image: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        points = np.asarray(box, dtype=np.int32).reshape(-1, 2)
        x, y, width, height = cv2.boundingRect(points)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 0, 255), 2)
    return annotated


def sort_boxes_reading_order(
    boxes: list[np.ndarray],
) -> list[tuple[int, int, int, int]]:
    items: list[tuple[int, int, int, int]] = []
    for box in boxes:
        rect = _box_to_rect(box)
        if rect is not None:
            items.append(rect)
    if not items:
        return []

    median_h = float(np.median([h for _, _, _, h in items]))
    row_tol = max(10.0, median_h * 0.6)
    items.sort(key=lambda item: item[1])

    lines: list[list[tuple[int, int, int, int]]] = []
    for item in items:
        if not lines or abs(item[1] - lines[-1][0][1]) > row_tol:
            lines.append([item])
        else:
            lines[-1].append(item)

    ordered: list[tuple[int, int, int, int]] = []
    for line in lines:
        line.sort(key=lambda item: item[0])
        ordered.extend(line)
    return ordered


def crop_region(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    pad: int = 6,
) -> np.ndarray:
    x, y, width, height = rect
    h, w = image.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + width + pad)
    y1 = min(h, y + height + pad)
    return image[y0:y1, x0:x1]
