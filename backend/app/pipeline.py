"""CRAFT + TrOCR extraction used by the API.

Detects regions, reads them with TrOCR, then hands text + original-page boxes
to the question-paper parser and the answer mapper.  When an output directory
is given, every page, crop, box, and line of text is written there.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from app.config import PipelineConfig, use_cuda
from app.steps.binarize import binarize_image
from app.steps.detect import (
    crop_region,
    detect_boxes,
    draw_bounding_boxes,
    load_detector,
    sort_boxes_reading_order,
)
from app.steps.map_answers import map_answer_segments, pages_from_records
from app.steps.orient import unrotate_rect, upright_page
from app.steps.parse_paper import parse_questions
from app.steps.pdf_pages import iter_pdf_pages
from app.steps.questions import analyze_pages
from app.steps.recognize import free_cuda, load_trocr, recognize_batch_local, recognize_line_api

log = logging.getLogger(__name__)
Progress = Callable[[str], None]


def _page_name(page_index: int, page_count: int) -> str:
    digits = max(3, len(str(page_count)))
    return f"page_{page_index:0{digits}d}"


def _build_text_output(records: list[dict]) -> str:
    if not records:
        return "(no text recognized)\n"
    pages: dict[int, list[str]] = {}
    for record in records:
        pages.setdefault(record["page"], []).append(record.get("text") or "")
    blocks: list[str] = []
    for page in sorted(pages):
        lines = [f"===== Page {page} ====="]
        page_texts = [text for text in pages[page] if text]
        lines.extend(page_texts if page_texts else ["(no text recognized)"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _write_detections(out_dir: Path, records: list[dict], page_dims: dict[int, dict]) -> None:
    payload = {
        "page_dims": {str(page): dims for page, dims in page_dims.items()},
        "lines": [
            {
                "id": record["id"],
                "page": record["page"],
                "region_index": record["region_index"],
                "bbox": record["bbox"],
                "text": record.get("text") or "",
                "crop": record.get("crop_rel"),
            }
            for record in records
        ],
    }
    (out_dir / "detections.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "extracted_text.txt").write_text(
        _build_text_output(records), encoding="utf-8"
    )


@dataclass
class ExtractedDocument:
    lines: list[dict] = field(default_factory=list)
    page_dims: dict[int, dict] = field(default_factory=dict)


def _note(progress: Progress | None, message: str) -> None:
    log.info(message)
    if progress:
        progress(message)


def _detect_pages(
    pdf_path: Path,
    config: PipelineConfig,
    out_dir: Path,
    progress: Progress | None,
) -> tuple[list[dict], dict[int, dict]]:
    _note(progress, "Detecting text regions…")
    craft_net, refine_net = load_detector(config.models_dir)
    records: list[dict] = []
    page_dims: dict[int, dict] = {}
    global_index = 0
    pages_dir = out_dir / "pages"
    regions_dir = out_dir / "regions"
    pages_dir.mkdir(parents=True, exist_ok=True)
    regions_dir.mkdir(parents=True, exist_ok=True)

    try:
        for page_index, page_count, page_bgr in iter_pdf_pages(pdf_path, dpi=config.dpi):
            orig_h, orig_w = page_bgr.shape[:2]
            page_dims[page_index] = {"width_px": orig_w, "height_px": orig_h}

            page_bgr, orientation = upright_page(page_bgr)
            if orientation.rotated:
                log.info(
                    "page %s/%s: rotated %s° to upright",
                    page_index,
                    page_count,
                    orientation.degrees,
                )
            binary = binarize_image(page_bgr)
            boxes = detect_boxes(
                binary,
                config.models_dir,
                text_threshold=config.text_threshold,
                link_threshold=config.link_threshold,
                low_text=config.low_text,
                long_size=config.long_size,
                min_ink_ratio=config.min_ink_ratio,
                craft_net=craft_net,
                refine_net=refine_net,
            )
            regions = sort_boxes_reading_order(boxes)
            name = _page_name(page_index, page_count)
            annotated = draw_bounding_boxes(binary, boxes)
            page_image = pages_dir / f"{name}.png"
            if not cv2.imwrite(str(page_image), annotated):
                raise RuntimeError(f"Could not write {page_image}")

            page_region_dir = regions_dir / name
            page_region_dir.mkdir(parents=True, exist_ok=True)

            for region_index, rect in enumerate(regions, start=1):
                crop = crop_region(binary, rect)
                if crop.size == 0:
                    continue
                crop_rel = f"regions/{name}/region_{region_index:04d}.png"
                crop_path = out_dir / crop_rel
                if not cv2.imwrite(str(crop_path), crop):
                    continue
                x1, y1, x2, y2 = unrotate_rect(
                    rect, orientation.degrees, orig_w, orig_h
                )
                global_index += 1
                records.append(
                    {
                        "id": global_index,
                        "page": page_index,
                        "region_index": region_index,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "crop_path": str(crop_path),
                        "crop_rel": crop_rel,
                        "page_image": f"pages/{name}.png",
                        "text": None,
                    }
                )

            del binary, annotated, page_bgr, boxes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("page %s: %s region(s)", page_index, len(regions))
    finally:
        free_cuda(craft_net, refine_net)

    return records, page_dims


def _recognize(
    records: list[dict],
    config: PipelineConfig,
    progress: Progress | None,
) -> list[dict]:
    if not records:
        return records
    _note(progress, "Reading text with TrOCR…")

    if config.use_api:
        if not config.huggingface_api_key:
            raise RuntimeError("USE_API=true but HUGGINGFACE_API_KEY is missing")
        for index, record in enumerate(records, start=1):
            crop = cv2.imread(record["crop_path"], cv2.IMREAD_UNCHANGED)
            if crop is None:
                record["text"] = ""
                continue
            record["text"] = recognize_line_api(config.huggingface_api_key, crop)
            if index % 20 == 0:
                log.info("recognized %s/%s", index, len(records))
        return records

    processor, model, device = load_trocr(config.models_dir, config.trocr_model)
    batch_size = max(1, config.batch_size)
    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            crops = []
            for record in batch:
                crop = cv2.imread(record["crop_path"], cv2.IMREAD_UNCHANGED)
                if crop is None:
                    crop = np.full((32, 32), 255, dtype=np.uint8)
                crops.append(crop)
            try:
                texts = recognize_batch_local(processor, model, device, crops)
            except torch.cuda.OutOfMemoryError:
                log.warning("batch OOM; falling back to one-by-one")
                free_cuda()
                texts = []
                for crop in crops:
                    texts.extend(recognize_batch_local(processor, model, device, [crop]))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            for record, text in zip(batch, texts):
                record["text"] = text
            log.info("recognized %s/%s", min(start + batch_size, len(records)), len(records))
    finally:
        free_cuda(processor, model)

    return records


def extract_document(
    pdf_path: Path,
    config: PipelineConfig,
    progress: Progress | None = None,
    out_dir: Path | None = None,
) -> ExtractedDocument:
    """Run detection + TrOCR and return lines with boxes in original page space."""
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    log.info("CUDA enabled: %s", use_cuda())
    persist = out_dir is not None
    if persist:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
        work_dir = out_dir
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="textt-ocr-")
        work_dir = Path(tmp_ctx.name)

    try:
        records, page_dims = _detect_pages(pdf_path, config, work_dir, progress)
        records = _recognize(records, config, progress)
        if persist:
            _write_detections(work_dir, records, page_dims)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if not persist:
        for record in records:
            record.pop("crop_path", None)
            record.pop("crop_rel", None)
    return ExtractedDocument(lines=records, page_dims=page_dims)


def extract_question_paper(
    pdf_path: Path,
    config: PipelineConfig,
    progress: Progress | None = None,
    out_dir: Path | None = None,
) -> dict:
    document = extract_document(pdf_path, config, progress, out_dir=out_dir)
    lines = [
        (line["page"], line.get("text") or "", list(line["bbox"]))
        for line in document.lines
        if line.get("text")
    ]
    parsed = parse_questions(lines)
    if not parsed.get("questions"):
        parsed = {"questions": _questions_from_markers(document.lines)}
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "questions.json").write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    log.info("extracted %s section(s) from question paper", len(parsed.get("questions", {})))
    return parsed


def _questions_from_markers(records: list[dict]) -> dict:
    """Fallback when the printed-paper parser cannot see section headers."""
    nested: dict = {}
    marks = analyze_pages(pages_from_records(records)).marks
    by_index = {mark.line_index: mark for mark in marks if mark.line_index is not None}
    for index, record in enumerate(records):
        mark = by_index.get(index)
        if mark is None:
            continue
        section = mark.section or "Section A"
        question = f"Question {mark.number}"
        nested.setdefault(section, {}).setdefault(question, {"_text": ""})
        nested[section][question]["_text"] += (record.get("text") or "") + " "
    return nested


def extract_answer_sheet(
    pdf_path: Path,
    config: PipelineConfig,
    progress: Progress | None = None,
    out_dir: Path | None = None,
) -> tuple[dict, dict]:
    document = extract_document(pdf_path, config, progress, out_dir=out_dir)
    segments = map_answer_segments(document.lines)
    page_dims = {str(page): dims for page, dims in document.page_dims.items()}
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "answersheet_results.json").write_text(
            json.dumps(
                {"page_dims": page_dims, "segments": segments},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    log.info("detected %s answer section(s)", len(segments))
    return segments, page_dims


def align_segment_sections(questions: dict, segments: dict) -> dict:
    """Copy answers filed under Unknown onto the matching question-paper section."""
    unknown = segments.get("Unknown")
    if not unknown or not questions:
        return segments
    q_to_section = {
        qkey: section
        for section, qs in questions.items()
        if isinstance(qs, dict)
        for qkey in qs
        if str(qkey).startswith("Question")
    }
    aligned = {key: value for key, value in segments.items() if key != "Unknown"}
    leftover: dict = {}
    for qkey, node in unknown.items():
        section = q_to_section.get(qkey)
        if section:
            aligned.setdefault(section, {})[qkey] = node
        else:
            leftover[qkey] = node
    if leftover:
        aligned["Unknown"] = leftover
    return aligned
