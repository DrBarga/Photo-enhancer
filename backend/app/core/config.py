from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Light Correction API"
    max_upload_bytes: int = 12 * 1024 * 1024
    max_processing_side: int = 1800
    allowed_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    )
    allowed_modes: tuple[str, ...] = ("gradient", "reflection", "shadow")


settings = Settings()
