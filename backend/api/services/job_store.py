"""In-memory job store — same contract as the previous backend."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from api.models import JobStatus


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: str = "Queued"
    error: Optional[str] = None
    ans_path: Optional[Path] = None
    qp_path: Optional[Path] = None
    payload: Optional[Dict[str, Any]] = None


class JobStore:
    def __init__(self) -> None:
        self._store: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(job_id=str(uuid.uuid4()))
        with self._lock:
            self._store[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._store.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if job:
                for key, value in kwargs.items():
                    setattr(job, key, value)


store = JobStore()
