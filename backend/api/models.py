"""Job status and API response models — same contract as the previous backend."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: Optional[str] = None
    error: Optional[str] = None


class ResultsResponse(BaseModel):
    job_id: str
    questions: Dict[str, Any]
    segments: Dict[str, Any]
    page_dims: Dict[str, Any]
