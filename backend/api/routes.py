"""
routes.py — same endpoints as the previous backend:

  POST /api/v1/upload
  GET  /api/v1/status/{job_id}
  GET  /api/v1/results/{job_id}
  GET  /api/v1/pdf/{job_id}

Extraction is CRAFT + TrOCR from this folder, not PaddleOCR.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.models import JobStatus, StatusResponse, UploadResponse
from api.services.job_store import store
from app.config import PipelineConfig, load_env, resolve_models_dir
from app.pipeline import (
    align_segment_sections,
    extract_answer_sheet,
    extract_question_paper,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")
_executor = ThreadPoolExecutor(max_workers=1)

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "user_data")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _config() -> PipelineConfig:
    load_env(BASE_DIR)
    return PipelineConfig.from_env(resolve_models_dir(BASE_DIR))


def _run_pipeline(job_id: str, qp_path: Path, ans_path: Path) -> None:
    try:
        config = _config()

        store.update(
            job_id,
            status=JobStatus.PROCESSING,
            progress="Running OCR on question paper…",
        )
        job_dir = qp_path.parent
        q_data = extract_question_paper(
            qp_path,
            config,
            progress=lambda message: store.update(job_id, progress=message),
            out_dir=job_dir / "question_paper",
        )

        store.update(job_id, progress="Running OCR on answer sheet…")
        segments, page_dims = extract_answer_sheet(
            ans_path,
            config,
            progress=lambda message: store.update(job_id, progress=message),
            out_dir=job_dir / "answer_sheet",
        )
        questions = q_data.get("questions", {})
        segments = align_segment_sections(questions, segments)
        results_path = job_dir / "answer_sheet" / "answersheet_results.json"
        results_path.write_text(
            json.dumps(
                {"page_dims": page_dims, "segments": segments},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        store.update(
            job_id,
            status=JobStatus.COMPLETED,
            progress="Done",
            payload={
                "job_id": job_id,
                "questions": questions,
                "segments": segments,
                "page_dims": page_dims,
            },
        )
        log.info("Job %s completed.", job_id)
    except Exception as exc:
        log.exception("Job %s failed: %s", job_id, exc)
        store.update(job_id, status=JobStatus.FAILED, error=str(exc))


@router.post("/upload", response_model=UploadResponse)
async def upload(
    question_paper: UploadFile = File(...),
    answer_sheet: UploadFile = File(...),
) -> UploadResponse:
    for upload in (question_paper, answer_sheet):
        if not (upload.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"'{upload.filename}' is not a PDF.")

    job = store.create()
    job_dir = UPLOAD_DIR / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    qp_path = job_dir / "question_paper.pdf"
    ans_path = job_dir / "answer_sheet.pdf"
    for upload, dest in ((question_paper, qp_path), (answer_sheet, ans_path)):
        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)

    store.update(job.job_id, qp_path=qp_path, ans_path=ans_path)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline, job.job_id, qp_path, ans_path)
    return UploadResponse(job_id=job.job_id)


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str) -> StatusResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return StatusResponse(
        job_id=job_id,
        status=job.status,
        progress=job.progress,
        error=job.error,
    )


@router.get("/results/{job_id}")
async def get_results(job_id: str) -> Dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status in {JobStatus.PENDING, JobStatus.PROCESSING}:
        raise HTTPException(202, "Still processing.")
    if job.status == JobStatus.FAILED:
        raise HTTPException(500, job.error or "Pipeline failed.")
    if not job.payload:
        raise HTTPException(404, "Results not ready.")
    return job.payload


@router.get("/pdf/{job_id}")
async def serve_pdf(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if not job or not job.ans_path or not job.ans_path.exists():
        raise HTTPException(404, "Answer sheet not found.")
    return FileResponse(job.ans_path, media_type="application/pdf")
