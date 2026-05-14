from __future__ import annotations

import cv2
import numpy as np

from app.models.schemas import AnalysisMaps
from app.utils.map_math import feather_mask, normalize_map


class UniversalImageAnalyzer:
    def analyze(self, image_rgb: np.ndarray, analysis: AnalysisMaps) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        height, width = gray.shape

        masks = self._semantic_masks(image_rgb, analysis, hsv, lab)
        quality = self._quality_metrics(image_rgb, analysis, gray, hsv, lab)
        scene_scores = self._scene_scores(image_rgb, analysis, masks, quality, hsv)
        recommendations = self._recommendations(quality, scene_scores, analysis)

        payload: dict[str, object] = {
            "dimensions": {"width": int(width), "height": int(height), "megapixels": round(float(width * height) / 1_000_000.0, 3)},
            "quality": quality,
            "scene_scores": {key: round(float(value), 4) for key, value in scene_scores.items()},
            "semantic_coverage": {
                key: {
                    "mean_percent": round(float(mask.mean()) * 100.0, 2),
                    "active_area_percent": round(float((mask > 0.20).mean()) * 100.0, 2),
                }
                for key, mask in masks.items()
            },
            "recommendations": recommendations,
            "supported_inputs": ["jpeg", "png", "webp", "bmp", "heic-if-pillow-heif-installed"],
        }
        return payload, masks

    def _quality_metrics(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        gray: np.ndarray,
        hsv: np.ndarray,
        lab: np.ndarray,
    ) -> dict[str, object]:
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        blur_score = float(np.clip(1.0 - np.log1p(blur_variance) / np.log1p(1200.0), 0.0, 1.0))

        flat_support = (analysis.texture < 0.35) & (analysis.edges < 0.12)
        residual = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.15))
        noise_value = float(np.mean(residual[flat_support])) if int(np.count_nonzero(flat_support)) > 40 else float(np.mean(residual))
        noise_score = float(np.clip(noise_value / 16.0, 0.0, 1.0))

        luma = analysis.luminance
        underexposure = float(np.mean(luma < 28.0))
        overexposure = float(np.mean(luma > 238.0))
        mid_contrast = float(np.std(luma) / 64.0)
        contrast_score = float(np.clip(mid_contrast, 0.0, 1.0))
        saturation_score = float(np.clip(np.mean(hsv[..., 1]) / 180.0, 0.0, 1.0))

        channel_means = image_rgb.astype(np.float32).mean(axis=(0, 1))
        wb_spread = float(np.std(channel_means) / (np.mean(channel_means) + 1e-6))
        white_balance_shift = float(np.clip(wb_spread / 0.22, 0.0, 1.0))

        jpeg_score = self._jpeg_block_score(gray)
        color_cast = {
            "red": round(float(channel_means[0]), 2),
            "green": round(float(channel_means[1]), 2),
            "blue": round(float(channel_means[2]), 2),
        }

        return {
            "blur": round(blur_score, 4),
            "noise": round(noise_score, 4),
            "overexposure": round(overexposure, 4),
            "underexposure": round(underexposure, 4),
            "white_balance_shift": round(white_balance_shift, 4),
            "contrast": round(contrast_score, 4),
            "saturation": round(saturation_score, 4),
            "jpeg_artifacts": round(jpeg_score, 4),
            "banding": round(float(np.mean(analysis.banding)), 4),
            "shadow_problem": round(float(np.mean(np.maximum(analysis.shadow_noise, analysis.cast_shadow_problem))), 4),
            "reflection_problem": round(float(np.mean(analysis.reflection_problem)), 4),
            "depth_available": analysis.ml_status.get("depth_status", "fallback") == "ok",
            "color_cast_rgb": color_cast,
        }

    def _semantic_masks(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        hsv: np.ndarray,
        lab: np.ndarray,
    ) -> dict[str, np.ndarray]:
        height, width = analysis.luminance.shape
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        yy = np.broadcast_to(yy, analysis.luminance.shape)
        hue = hsv[..., 0]
        sat = hsv[..., 1] / 255.0
        val = hsv[..., 2] / 255.0

        sky = ((yy < 0.62) & (hue > 86) & (hue < 132) & (sat > 0.16) & (val > 0.36) & (analysis.texture < 0.48)).astype(np.float32)
        water = ((yy > 0.34) & (hue > 82) & (hue < 136) & (sat > 0.10) & (analysis.texture < 0.64)).astype(np.float32)
        glass = np.clip((analysis.specular * 0.58 + analysis.reflection_mask * 0.34 + (1.0 - analysis.texture) * 0.08), 0.0, 1.0)
        asphalt = ((yy > 0.45) & (analysis.luminance < 130) & (sat < 0.34) & (analysis.texture > 0.22)).astype(np.float32)
        green_nature = (((hue > 35) & (hue < 86) & (sat > 0.20)) | ((hue > 22) & (hue < 42) & (sat > 0.28))).astype(np.float32)
        skin = self._skin_mask(image_rgb)
        face = self._face_mask(image_rgb, skin)
        text = self._text_candidate_mask(analysis)
        object_mask = self._object_mask(image_rgb, analysis, sat)
        background = np.clip(1.0 - feather_mask(np.maximum.reduce([object_mask, face, text]), sigma=2.4), 0.0, 1.0)

        interior = ((sat < 0.46) & (analysis.edges > 0.08) & (analysis.texture > 0.12) & (sky < 0.1) & (green_nature < 0.2)).astype(np.float32)
        food = (((hue < 28) | ((hue > 148) & (hue < 178))) & (sat > 0.28) & (val > 0.28) & (yy > 0.18)).astype(np.float32)
        product = np.clip(object_mask * (0.45 + background * 0.55), 0.0, 1.0)

        return {
            "face": feather_mask(face, sigma=2.0),
            "person": feather_mask(np.maximum(face, skin * 0.65), sigma=2.0),
            "object": feather_mask(object_mask, sigma=1.8),
            "background": background,
            "sky": feather_mask(sky, sigma=2.5),
            "water": feather_mask(water, sigma=2.6),
            "glass": feather_mask((glass > 0.22).astype(np.float32), sigma=2.0),
            "asphalt": feather_mask(asphalt, sigma=2.4),
            "interior": feather_mask(interior, sigma=2.2),
            "food": feather_mask(food, sigma=2.0),
            "product": feather_mask(product, sigma=2.0),
            "text": feather_mask(text, sigma=1.2),
            "nature": feather_mask(green_nature, sigma=2.2),
        }

    def _scene_scores(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        masks: dict[str, np.ndarray],
        quality: dict[str, object],
        hsv: np.ndarray,
    ) -> dict[str, float]:
        edge_density = float(analysis.edge_density)
        sat_mean = float(np.mean(hsv[..., 1]) / 255.0)
        luma_mean = float(np.mean(analysis.luminance) / 255.0)
        screenshot_score = float(np.clip(edge_density * 2.3 + float(quality["jpeg_artifacts"]) * 0.6 - float(quality["noise"]) * 0.25, 0.0, 1.0))
        night_score = float(np.clip((0.42 - luma_mean) / 0.34 + float(quality["underexposure"]) * 2.0, 0.0, 1.0))
        portrait_score = float(np.clip(masks["face"].mean() * 12.0 + masks["person"].mean() * 2.2, 0.0, 1.0))
        product_score = float(np.clip(masks["product"].mean() * 3.6 + masks["background"].mean() * 0.30 - masks["nature"].mean() * 0.25, 0.0, 1.0))
        interior_score = float(np.clip(masks["interior"].mean() * 2.8 + edge_density * 0.7 - masks["sky"].mean() * 1.2, 0.0, 1.0))
        nature_score = float(np.clip(masks["nature"].mean() * 2.2 + masks["sky"].mean() * 1.1 + masks["water"].mean() * 0.8, 0.0, 1.0))
        food_score = float(np.clip(masks["food"].mean() * 3.0 + sat_mean * 0.25, 0.0, 1.0))
        text_score = float(np.clip(masks["text"].mean() * 8.0 + screenshot_score * 0.22, 0.0, 1.0))

        return {
            "face": portrait_score,
            "person": float(np.clip(masks["person"].mean() * 4.0, 0.0, 1.0)),
            "object": float(np.clip(masks["object"].mean() * 2.0, 0.0, 1.0)),
            "background": float(np.clip(masks["background"].mean(), 0.0, 1.0)),
            "sky": float(np.clip(masks["sky"].mean() * 3.2, 0.0, 1.0)),
            "water": float(np.clip(masks["water"].mean() * 3.0, 0.0, 1.0)),
            "glass": float(np.clip(masks["glass"].mean() * 4.0, 0.0, 1.0)),
            "asphalt": float(np.clip(masks["asphalt"].mean() * 3.0, 0.0, 1.0)),
            "interior": interior_score,
            "food": food_score,
            "product": product_score,
            "text": text_score,
            "screenshot": screenshot_score,
            "portrait": portrait_score,
            "night": night_score,
            "nature": nature_score,
        }

    def _recommendations(
        self,
        quality: dict[str, object],
        scene_scores: dict[str, float],
        analysis: AnalysisMaps,
    ) -> list[dict[str, object]]:
        checks = [
            ("exposure_lift", float(quality["underexposure"]) > 0.035 or scene_scores["night"] > 0.42, "Поднять экспозицию и раскрыть тени"),
            ("highlight_recovery", float(quality["overexposure"]) > 0.025, "Сдержать пересветы"),
            ("white_balance", float(quality["white_balance_shift"]) > 0.20, "Стабилизировать баланс белого"),
            ("denoise", float(quality["noise"]) > 0.22 or scene_scores["night"] > 0.36, "Убрать шум без разрушения деталей"),
            ("deblur_or_sharpen", float(quality["blur"]) > 0.30, "Добавить резкость или deblur"),
            ("jpeg_cleanup", float(quality["jpeg_artifacts"]) > 0.16, "Снизить JPEG/block artifacts"),
            ("gradient_fix", float(quality["banding"]) > 0.025 or float(np.mean(analysis.gradient_problem)) > 0.035, "Исправить banding и резкие градиенты"),
            ("shadow_cleanup", float(quality["shadow_problem"]) > 0.025, "Почистить тени"),
            ("reflection_fix", float(quality["reflection_problem"]) > 0.014 or scene_scores["water"] > 0.25 or scene_scores["glass"] > 0.25, "Согласовать отражения"),
            ("protect_faces_text", scene_scores["portrait"] > 0.18 or scene_scores["text"] > 0.18, "Защитить лица и текст от агрессивных правок"),
        ]
        return [
            {"key": key, "label": label, "enabled": bool(enabled)}
            for key, enabled, label in checks
        ]

    def _skin_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        mask = ((cr > 135) & (cr < 180) & (cb > 78) & (cb < 135) & (y > 35)).astype(np.float32)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)).astype(np.float32)

    def _face_mask(self, image_rgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        face_mask = np.zeros(gray.shape, dtype=np.float32)
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
            for x, y, w, h in faces:
                face_mask[y : y + h, x : x + w] = 1.0
        except Exception:
            pass
        if float(face_mask.mean()) < 0.001 and float(skin.mean()) > 0.004:
            face_mask = self._largest_component(skin, max_coverage=0.18)
        return face_mask

    def _text_candidate_mask(self, analysis: AnalysisMaps) -> np.ndarray:
        edge = (analysis.edges > 0.12).astype(np.uint8)
        connected = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((3, 9), dtype=np.uint8))
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(connected, 8)
        text = np.zeros_like(edge, dtype=np.uint8)
        image_area = edge.size
        for index in range(1, component_count):
            x, y, w, h, area = stats[index]
            if area < 8 or area > image_area * 0.040:
                continue
            aspect = w / max(h, 1)
            if 1.5 <= aspect <= 18.0 and 5 <= h <= max(12, analysis.luminance.shape[0] * 0.12):
                text[labels == index] = 1
        return text.astype(np.float32)

    def _object_mask(self, image_rgb: np.ndarray, analysis: AnalysisMaps, saturation: np.ndarray) -> np.ndarray:
        saliency = np.clip(
            0.34 * analysis.gradient
            + 0.27 * analysis.texture
            + 0.20 * saturation
            + 0.13 * (1.0 - analysis.smooth_background)
            + 0.06 * analysis.depth,
            0.0,
            1.0,
        )
        threshold = max(0.20, float(np.percentile(saliency, 74)))
        return self._largest_component((saliency > threshold).astype(np.float32), max_coverage=0.62)

    def _largest_component(self, mask: np.ndarray, max_coverage: float) -> np.ndarray:
        mask_u8 = (mask > 0.1).astype(np.uint8)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
        if component_count <= 1:
            return mask_u8.astype(np.float32)
        image_area = mask_u8.size
        best_index = 0
        best_area = 0
        for index in range(1, component_count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area > best_area and area <= image_area * max_coverage:
                best_area = area
                best_index = index
        if best_index == 0:
            return np.zeros_like(mask, dtype=np.float32)
        return (labels == best_index).astype(np.float32)

    def _jpeg_block_score(self, gray: np.ndarray) -> float:
        if gray.shape[0] < 16 or gray.shape[1] < 16:
            return 0.0
        vertical_boundaries = np.arange(8, gray.shape[1], 8)
        horizontal_boundaries = np.arange(8, gray.shape[0], 8)
        if vertical_boundaries.size == 0 or horizontal_boundaries.size == 0:
            return 0.0
        v = np.mean(np.abs(gray[:, vertical_boundaries] - gray[:, vertical_boundaries - 1]))
        h = np.mean(np.abs(gray[horizontal_boundaries, :] - gray[horizontal_boundaries - 1, :]))
        baseline = np.mean(np.abs(np.diff(gray, axis=1))) * 0.5 + np.mean(np.abs(np.diff(gray, axis=0))) * 0.5 + 1e-6
        return float(np.clip(((v + h) * 0.5 - baseline) / 18.0, 0.0, 1.0))
