from __future__ import annotations

import json
import os
from typing import Any


class PostgresHistoryRepository:
    def __init__(self) -> None:
        self.dsn = os.getenv("AI_LIGHT_POSTGRES_DSN", "")
        if not self.dsn:
            raise RuntimeError("AI_LIGHT_POSTGRES_DSN is required")

    @property
    def enabled(self) -> bool:
        return bool(self.dsn)

    def save(self, record: dict[str, Any]) -> None:
        import psycopg
        from psycopg.types.json import Jsonb

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into image_history (
                        id, created_at, modes, prompt, total_score, problem,
                        material, strength, problem_level, result_thumb, payload
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (id) do update set
                        total_score = excluded.total_score,
                        result_thumb = excluded.result_thumb,
                        payload = excluded.payload
                    """,
                    (
                        record.get("id"),
                        record.get("created_at"),
                        Jsonb(record.get("modes", [])),
                        record.get("prompt", ""),
                        record.get("total_score"),
                        record.get("problem"),
                        record.get("material"),
                        record.get("strength"),
                        record.get("problem_level"),
                        record.get("result_thumb"),
                        Jsonb(record),
                    ),
                )

    def list(self, limit: int) -> list[dict[str, Any]]:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select payload from image_history order by created_at desc limit %s",
                    (max(1, min(limit, 100)),),
                )
                return [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in cur.fetchall()]

    def get(self, record_id: str) -> dict[str, Any] | None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("select payload from image_history where id = %s", (record_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])
