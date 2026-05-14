import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Light Correction API"
    max_upload_bytes: int = int(os.getenv("AI_LIGHT_MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))
    max_processing_side: int = int(os.getenv("AI_LIGHT_MAX_PROCESSING_SIDE", "2200"))
    allowed_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/heic",
        "image/heif",
        "image/avif",
    )
    allowed_modes: tuple[str, ...] = ("gradient", "reflection", "shadow")


settings = Settings()
