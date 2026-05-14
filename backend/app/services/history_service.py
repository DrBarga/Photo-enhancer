from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from app.services.postgres_history import PostgresHistoryRepository
from app.services.storage_service import StorageService, create_storage_service
from app.utils.image_io import encode_png_data_url


class HistoryService:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or self._default_base_dir())
        self.index_path = self.base_dir / "index.jsonl"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.storage: StorageService = create_storage_service(str(self.base_dir))
        self.postgres: PostgresHistoryRepository | None = None
        if os.getenv("AI_LIGHT_POSTGRES_DSN"):
            self.postgres = PostgresHistoryRepository()

    def _default_base_dir(self) -> str:
        configured_dir = os.getenv("AI_LIGHT_HISTORY_DIR")
        if configured_dir:
            return configured_dir
        if os.getenv("VERCEL"):
            return str(Path(tempfile.gettempdir()) / "ai-light-history")
        return "backend/data/history"

    def save(
        self,
        *,
        input_rgb: np.ndarray,
        result_rgb: np.ndarray,
        heatmap_before_rgb: np.ndarray,
        heatmap_after_rgb: np.ndarray,
        heatmap_delta_rgb: np.ndarray,
        modes: list[str],
        prompt: str,
        metrics: list[dict[str, Any]],
        total_score: int,
        analysis_summary: dict[str, Any],
        ml_understanding: dict[str, Any],
    ) -> dict[str, Any]:
        record_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex[:8]
        record_dir = self.base_dir / record_id
        record_dir.mkdir(parents=True, exist_ok=True)

        stored_images = {
            "input": self.storage.put_png(f"{record_id}/input.png", input_rgb),
            "result": self.storage.put_png(f"{record_id}/result.png", result_rgb),
            "heatmap_before": self.storage.put_png(f"{record_id}/heatmap_before.png", heatmap_before_rgb),
            "heatmap_after": self.storage.put_png(f"{record_id}/heatmap_after.png", heatmap_after_rgb),
            "heatmap_delta": self.storage.put_png(f"{record_id}/heatmap_delta.png", heatmap_delta_rgb),
        }

        summary = {
            "id": record_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "modes": modes,
            "prompt": prompt,
            "total_score": total_score,
            "problem": ml_understanding.get("problem"),
            "material": ml_understanding.get("material"),
            "strength": ml_understanding.get("strength"),
            "problem_level": analysis_summary.get("problem_level"),
            "result_thumb": encode_png_data_url(self._thumbnail(result_rgb)),
            "result_url": stored_images["result"].public_url,
        }
        full_record = {
            **summary,
            "metrics": metrics,
            "analysis_summary": analysis_summary,
            "ml_understanding": ml_understanding,
            "images": {
                key: value.key for key, value in stored_images.items()
            },
            "image_urls": {
                key: value.public_url for key, value in stored_images.items() if value.public_url
            },
        }
        (record_dir / "record.json").write_text(json.dumps(full_record, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.index_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(summary, ensure_ascii=False) + "\n")
        if self.postgres is not None:
            self.postgres.save(full_record)
        return summary

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.postgres is not None:
            return self.postgres.list(limit)
        if not self.index_path.exists():
            return []
        rows = []
        with self.index_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return rows[: max(1, min(limit, 100))]

    def get(self, record_id: str) -> dict[str, Any] | None:
        if self.postgres is not None:
            record = self.postgres.get(record_id)
            if record is None:
                return None
            return self._attach_image_data(record, record_id)

        record_dir = self.base_dir / record_id
        record_path = record_dir / "record.json"
        if not record_path.exists():
            return None
        record = json.loads(record_path.read_text(encoding="utf-8"))
        return self._attach_image_data(record, record_id)

    def _attach_image_data(self, record: dict[str, Any], record_id: str) -> dict[str, Any]:
        images = record.get("images", {})
        image_data = {}
        for key, file_name in images.items():
            if not file_name:
                continue
            try:
                image_data[key] = encode_png_data_url(self._read_stored_image(record_id, file_name))
            except FileNotFoundError:
                continue
        record["image_data"] = image_data
        return record

    def compare(self, left_id: str, right_id: str) -> dict[str, Any] | None:
        left = self.get(left_id)
        right = self.get(right_id)
        if left is None or right is None:
            return None
        return {
            "left": left,
            "right": right,
            "score_delta": int(right.get("total_score", 0)) - int(left.get("total_score", 0)),
        }

    def _read_stored_image(self, record_id: str, file_name: str) -> np.ndarray:
        try:
            return self.storage.read_image(file_name)
        except FileNotFoundError:
            return self.storage.read_image(f"{record_id}/{file_name}")

    def _thumbnail(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        scale = min(1.0, 240.0 / max(height, width))
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)
