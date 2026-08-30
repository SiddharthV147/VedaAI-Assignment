"""Map TrOCR lines + CRAFT boxes onto the nested segment tree the UI expects."""

from __future__ import annotations

from collections import defaultdict

from app.steps.ocr_text import Line, Page, clean_line
from app.steps.questions import analyze_pages

BBox = tuple[float, float, float, float]


def pages_from_records(records: list[dict]) -> list[Page]:
    by_page: dict[int, list[Line]] = {}
    for index, record in enumerate(records):
        text = clean_line(record.get("text") or "")
        if not text:
            continue
        by_page.setdefault(record["page"], []).append(
            Line(page=record["page"], index=index, text=text)
        )
    return [Page(number=number, lines=tuple(by_page[number])) for number in sorted(by_page)]


def _merge_page_boxes(boxes: list[BBox], gap: float = 100.0) -> list[BBox]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda box: box[1])
    merged: list[list[float]] = [list(ordered[0])]
    for box in ordered[1:]:
        current = merged[-1]
        if box[1] - current[3] < gap:
            current[0] = min(current[0], box[0])
            current[1] = min(current[1], box[1])
            current[2] = max(current[2], box[2])
            current[3] = max(current[3], box[3])
        else:
            merged.append(list(box))
    return [tuple(item) for item in merged]


def _coords_from_records(records: list[dict]) -> list[dict]:
    by_page: dict[int, list[BBox]] = defaultdict(list)
    for record in records:
        bbox = record.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        by_page[record["page"]].append(tuple(bbox))
    coords: list[dict] = []
    for page in sorted(by_page):
        for bbox in _merge_page_boxes(by_page[page]):
            coords.append({"page": page, "bbox": [float(v) for v in bbox]})
    return coords


def map_answer_segments(records: list[dict]) -> dict:
    """Build ``{Section → Question → {_coords, text_lines, page, pages}}``."""
    pages = pages_from_records(records)
    marks = analyze_pages(pages).marks
    if not marks:
        return {}

    anchored = [mark for mark in marks if mark.line_index is not None]
    anchored.sort(key=lambda mark: mark.line_index or 0)

    assigned: set[int] = set()
    nested: dict = {}

    for position, mark in enumerate(anchored):
        start = mark.line_index or 0
        end = (
            anchored[position + 1].line_index
            if position + 1 < len(anchored) and anchored[position + 1].line_index is not None
            else len(records)
        )
        slice_records = records[start:end]
        assigned.update(range(start, end))
        _store_segment(nested, mark, slice_records)

    leftovers = [record for index, record in enumerate(records) if index not in assigned]
    for mark in marks:
        if mark.line_index is not None:
            continue
        page_records = [record for record in leftovers if record["page"] == mark.page]
        if page_records:
            _store_segment(nested, mark, page_records)

    return nested


def _store_segment(nested: dict, mark, records: list[dict]) -> None:
    section = mark.section or "Unknown"
    question = f"Question {mark.number}"
    nested.setdefault(section, {})
    node = nested[section].setdefault(question, {"_coords": [], "text_lines": []})
    node["_coords"].extend(_coords_from_records(records))
    node["text_lines"].extend(
        text for text in (record.get("text") or "" for record in records) if text
    )
    pages = sorted({coord["page"] for coord in node["_coords"]})
    if pages:
        node["page"] = pages[0]
        node["pages"] = pages
