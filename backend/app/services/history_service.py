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

from app.utils.image_io import encode_png_data_url


class HistoryService:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or self._default_base_dir())
        self.index_path = self.base_dir / "index.jsonl"
        self.base_dir.mkdir(parents=True, exist_ok=True)

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

        self._write_image(record_dir / "input.png", input_rgb)
        self._write_image(record_dir / "result.png", result_rgb)
        self._write_image(record_dir / "heatmap_before.png", heatmap_before_rgb)
        self._write_image(record_dir / "heatmap_after.png", heatmap_after_rgb)
        self._write_image(record_dir / "heatmap_delta.png", heatmap_delta_rgb)

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
        }
        full_record = {
            **summary,
            "metrics": metrics,
            "analysis_summary": analysis_summary,
            "ml_understanding": ml_understanding,
            "images": {
                "input": "input.png",
                "result": "result.png",
                "heatmap_before": "heatmap_before.png",
                "heatmap_after": "heatmap_after.png",
                "heatmap_delta": "heatmap_delta.png",
            },
        }
        (record_dir / "record.json").write_text(json.dumps(full_record, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.index_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return summary

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
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
        record_dir = self.base_dir / record_id
        record_path = record_dir / "record.json"
        if not record_path.exists():
            return None
        record = json.loads(record_path.read_text(encoding="utf-8"))
        images = record.get("images", {})
        record["image_data"] = {
            key: encode_png_data_url(self._read_image(record_dir / file_name))
            for key, file_name in images.items()
            if (record_dir / file_name).exists()
        }
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

    def _write_image(self, path: Path, image_rgb: np.ndarray) -> None:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    def _read_image(self, path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(str(path))
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _thumbnail(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        scale = min(1.0, 240.0 / max(height, width))
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)
