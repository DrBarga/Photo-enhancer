from __future__ import annotations

import json
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from uuid import uuid4


class JobService:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or self._default_base_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.postgres_dsn = os.getenv("AI_LIGHT_POSTGRES_DSN", "")
        self._lock = Lock()

    def _default_base_dir(self) -> str:
        configured_dir = os.getenv("AI_LIGHT_JOB_DIR")
        if configured_dir:
            return configured_dir
        if os.getenv("VERCEL"):
            return str(Path(tempfile.gettempdir()) / "ai-light-jobs")
        return "backend/data/jobs"

    def create(self, kind: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        job = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex[:8],
            "kind": kind,
            "status": "queued",
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "result": None,
            "error": None,
        }
        self._write(job)
        return self.public(job)

    def run(self, job_id: str, work: Callable[[Callable[[int, str], None]], dict[str, Any]]) -> None:
        try:
            self._update(job_id, status="running", progress=5, error=None)

            def report(progress: int, message: str) -> None:
                self._update(job_id, progress=max(0, min(progress, 99)), message=message)

            result = work(report)
            self._update(job_id, status="done", progress=100, result=result, message="done")
        except Exception as exc:  # noqa: BLE001
            self._update(
                job_id,
                status="failed",
                error={"message": str(exc), "traceback": traceback.format_exc(limit=12)},
                message="failed",
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        if self.postgres_dsn:
            job = self._read_postgres(job_id)
            return self.public(job) if job is not None else None

        path = self._path(job_id)
        if not path.exists():
            return None
        return self.public(json.loads(path.read_text(encoding="utf-8")))

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        return job

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._read(job_id)
            if job is None:
                raise FileNotFoundError(job_id)
            job.update(changes)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write(job)

    def _write(self, job: dict[str, Any]) -> None:
        if self.postgres_dsn:
            self._write_postgres(job)
            return
        path = self._path(job["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read(self, job_id: str) -> dict[str, Any] | None:
        if self.postgres_dsn:
            return self._read_postgres(job_id)
        path = self._path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_postgres(self, job: dict[str, Any]) -> None:
        import psycopg
        from psycopg.types.json import Jsonb

        with psycopg.connect(self.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into image_jobs (
                        id, kind, status, progress, created_at, updated_at,
                        metadata, result, error
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (id) do update set
                        status = excluded.status,
                        progress = excluded.progress,
                        updated_at = excluded.updated_at,
                        metadata = excluded.metadata,
                        result = excluded.result,
                        error = excluded.error
                    """,
                    (
                        job["id"],
                        job.get("kind", "job"),
                        job.get("status", "queued"),
                        job.get("progress", 0),
                        job.get("created_at"),
                        job.get("updated_at"),
                        Jsonb(job.get("metadata") or {}),
                        Jsonb(job.get("result")) if job.get("result") is not None else None,
                        Jsonb(job.get("error")) if job.get("error") is not None else None,
                    ),
                )

    def _read_postgres(self, job_id: str) -> dict[str, Any] | None:
        import psycopg

        with psycopg.connect(self.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, kind, status, progress, created_at, updated_at,
                           metadata, result, error
                    from image_jobs
                    where id = %s
                    """,
                    (job_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": row[0],
                    "kind": row[1],
                    "status": row[2],
                    "progress": row[3],
                    "created_at": row[4].isoformat() if hasattr(row[4], "isoformat") else row[4],
                    "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else row[5],
                    "metadata": row[6] or {},
                    "result": row[7],
                    "error": row[8],
                }

    def _path(self, job_id: str) -> Path:
        return self.base_dir / f"{job_id}.json"
