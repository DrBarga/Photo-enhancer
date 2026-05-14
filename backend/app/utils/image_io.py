import base64
from io import BytesIO

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings


try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass


def validate_upload(file: UploadFile, content: bytes) -> None:
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Файл изображения пустой.",
        )

    if len(content) > settings.max_upload_bytes:
        max_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Изображение слишком большое. Максимальный размер: {max_mb} MB.",
        )

    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Поддерживаются JPG, PNG, WEBP, BMP, HEIC/HEIF и AVIF изображения.",
        )


def load_image_rgb(content: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(content))
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            background.alpha_composite(image.convert("RGBA"))
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось прочитать изображение.",
        ) from exc

    array = np.asarray(image, dtype=np.uint8)
    return limit_processing_size(array)


def limit_processing_size(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side <= settings.max_processing_side:
        return image

    scale = settings.max_processing_side / max_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def encode_png_data_url(image_rgb: np.ndarray) -> str:
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось закодировать результат обработки.",
        )
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"
