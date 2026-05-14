import numpy as np


def colorize_problem_map(problem_map: np.ndarray) -> np.ndarray:
    score = np.clip(problem_map.astype(np.float32), 0.0, 1.0)

    green = np.array([34, 197, 94], dtype=np.float32)
    orange = np.array([245, 158, 11], dtype=np.float32)
    red = np.array([239, 68, 68], dtype=np.float32)

    low_t = np.clip(score / 0.5, 0.0, 1.0)[..., None]
    high_t = np.clip((score - 0.5) / 0.5, 0.0, 1.0)[..., None]

    low_mix = green * (1.0 - low_t) + orange * low_t
    high_mix = orange * (1.0 - high_t) + red * high_t
    colorized = np.where((score[..., None] <= 0.5), low_mix, high_mix)
    return np.clip(colorized, 0, 255).astype(np.uint8)


def render_heatmap_overlay(image_rgb: np.ndarray, problem_map: np.ndarray) -> np.ndarray:
    colorized = colorize_problem_map(problem_map)
    score = np.clip(problem_map.astype(np.float32), 0.0, 1.0)
    visible = np.clip((score - 0.035) / 0.34, 0.0, 1.0)
    alpha = (visible ** 0.72 * 0.74)[..., None]
    overlay = image_rgb.astype(np.float32) * (1.0 - alpha) + colorized.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_delta_heatmap_overlay(
    image_rgb: np.ndarray,
    before_map: np.ndarray,
    after_map: np.ndarray,
) -> np.ndarray:
    improvement = np.clip(before_map.astype(np.float32) - after_map.astype(np.float32), 0.0, 1.0)
    regression = np.clip(after_map.astype(np.float32) - before_map.astype(np.float32), 0.0, 1.0)
    blue = np.array([59, 130, 246], dtype=np.float32)
    red = np.array([239, 68, 68], dtype=np.float32)
    colorized = blue * improvement[..., None] + red * regression[..., None]
    strength = np.maximum(improvement, regression)
    alpha = (np.clip(strength / 0.22, 0.0, 1.0) ** 0.72 * 0.68)[..., None]
    overlay = image_rgb.astype(np.float32) * (1.0 - alpha) + colorized * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_depth_map(depth_map: np.ndarray) -> np.ndarray:
    depth = np.clip(depth_map.astype(np.float32), 0.0, 1.0)
    far = np.array([30, 64, 175], dtype=np.float32)
    mid = np.array([34, 211, 238], dtype=np.float32)
    near = np.array([251, 191, 36], dtype=np.float32)

    low_t = np.clip(depth / 0.55, 0.0, 1.0)[..., None]
    high_t = np.clip((depth - 0.55) / 0.45, 0.0, 1.0)[..., None]
    far_mid = far * (1.0 - low_t) + mid * low_t
    mid_near = mid * (1.0 - high_t) + near * high_t
    colorized = np.where(depth[..., None] <= 0.55, far_mid, mid_near)
    return np.clip(colorized, 0, 255).astype(np.uint8)
