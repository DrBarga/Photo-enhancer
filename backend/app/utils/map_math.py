import cv2
import numpy as np


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - min_value) / (max_value - min_value), 0.0, 1.0)


def local_mean_std(values: np.ndarray, kernel_size: int = 21) -> tuple[np.ndarray, np.ndarray]:
    values = values.astype(np.float32)
    mean = cv2.blur(values, (kernel_size, kernel_size))
    squared_mean = cv2.blur(values * values, (kernel_size, kernel_size))
    variance = np.maximum(squared_mean - mean * mean, 0.0)
    return mean, np.sqrt(variance)


def feather_mask(mask: np.ndarray, sigma: float = 9.0) -> np.ndarray:
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0)


def blend_by_mask(original: np.ndarray, corrected: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_3d = np.clip(mask, 0.0, 1.0).astype(np.float32)[..., None]
    original_float = original.astype(np.float32)
    corrected_float = corrected.astype(np.float32)
    blended = original_float * (1.0 - mask_3d) + corrected_float * mask_3d
    return np.clip(blended, 0, 255).astype(np.uint8)


def clamp_score(value: float) -> int:
    return int(round(float(np.clip(value, 0, 100))))
