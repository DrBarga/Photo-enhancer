import cv2
import numpy as np

from app.models.schemas import AnalysisMaps
from app.services.ml_providers import get_ml_services
from app.utils.map_math import feather_mask, normalize_map


class SmartMaskBuilder:
    """Builds local correction masks with unsupervised pixel clustering."""

    def gradient_mask(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
    ) -> np.ndarray:
        prior = np.clip(
            0.68 * analysis.gradient_problem
            + 0.24 * analysis.banding
            + 0.08 * analysis.overexposure * analysis.smooth_background,
            0.0,
            1.0,
        )
        segments = get_ml_services().segmentation.masks(image_rgb, analysis).value
        surface = segments.get("surface", np.ones_like(analysis.luminance, dtype=np.float32))
        object_mask = segments.get("object", np.zeros_like(analysis.luminance, dtype=np.float32))
        object_block = cv2.dilate(
            (object_mask > 0.12).astype(np.uint8),
            np.ones((13, 13), dtype=np.uint8),
            iterations=1,
        ).astype(np.float32)
        object_block = feather_mask(object_block, sigma=4.0)
        allowed = (analysis.edges < 0.30) & (analysis.texture < 0.66) & (analysis.smooth_background > 0.18)
        allowed = allowed.astype(np.float32) * (0.32 + 0.68 * surface) * (1.0 - object_block * 0.90)
        mask = self._cluster_problem_regions(
            image_rgb=image_rgb,
            analysis=analysis,
            prior=prior,
            allowed=allowed.astype(np.float32),
            min_prior=0.035,
            percentile=72,
            feather_sigma=5.8,
            max_coverage=0.38,
        )
        if float(np.mean(mask)) >= 0.001:
            return mask
        return self._threshold_problem_regions(
            prior=prior,
            allowed=allowed.astype(np.float32),
            min_prior=0.026,
            percentile=66,
            feather_sigma=6.5,
            max_coverage=0.34,
            min_component_ratio=0.0008,
        )

    def reflection_mask(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
    ) -> np.ndarray:
        height = image_rgb.shape[0]
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        lower_region = (yy > 0.48).astype(np.float32)
        lower_region = np.broadcast_to(lower_region, analysis.luminance.shape)

        prior = np.clip(
            0.64 * analysis.reflection_problem
            + 0.24 * analysis.reflection_mask * analysis.contrast
            + 0.12 * analysis.reflection_mask * analysis.specular,
            0.0,
            1.0,
        )
        segments = get_ml_services().segmentation.masks(image_rgb, analysis).value
        surface = segments.get("surface", np.ones_like(analysis.luminance, dtype=np.float32))
        object_mask = segments.get("object", np.zeros_like(analysis.luminance, dtype=np.float32))
        allowed = lower_region * (analysis.reflection_mask > 0.20).astype(np.float32)
        allowed = allowed * (0.50 + 0.50 * surface) * (1.0 - object_mask * 0.45)
        return self._cluster_problem_regions(
            image_rgb=image_rgb,
            analysis=analysis,
            prior=prior,
            allowed=allowed,
            min_prior=0.055,
            percentile=90,
            feather_sigma=4.5,
            max_coverage=0.070,
        )

    def shadow_mask(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
    ) -> np.ndarray:
        local_darkness = normalize_map(255.0 - analysis.luminance)
        prior = np.clip(
            0.52 * analysis.cast_shadow_problem
            + 0.24 * analysis.shadow_noise
            + 0.16 * analysis.shadow_mask * analysis.contrast
            + 0.08 * local_darkness * analysis.shadow_mask,
            0.0,
            1.0,
        )
        segments = get_ml_services().segmentation.masks(image_rgb, analysis).value
        surface = segments.get("surface", np.ones_like(analysis.luminance, dtype=np.float32))
        allowed = ((analysis.shadow_mask > 0.08) | (analysis.cast_shadow_problem > 0.08)).astype(np.float32)
        allowed = allowed * (0.45 + 0.55 * surface)
        return self._threshold_problem_regions(
            prior=prior,
            allowed=allowed,
            min_prior=0.045,
            percentile=70,
            feather_sigma=5.0,
            max_coverage=0.16,
            min_component_ratio=0.00045,
        )

    def _threshold_problem_regions(
        self,
        prior: np.ndarray,
        allowed: np.ndarray,
        min_prior: float,
        percentile: float,
        feather_sigma: float,
        max_coverage: float,
        min_component_ratio: float,
    ) -> np.ndarray:
        allowed = (allowed > 0.05).astype(np.float32)
        if int(np.count_nonzero(allowed)) < 25:
            return np.zeros_like(prior, dtype=np.float32)

        values = prior[allowed > 0]
        threshold = max(min_prior, float(np.percentile(values, percentile)))
        mask = ((prior >= threshold) & (allowed > 0)).astype(np.float32)
        mask = self._limit_total_coverage(mask, prior, max_coverage)
        mask = self._cleanup(mask, min_component_ratio=min_component_ratio)
        mask = self._limit_total_coverage(mask, prior, max_coverage)
        if float(np.mean(mask)) < 0.001:
            return np.zeros_like(prior, dtype=np.float32)
        support = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), dtype=np.uint8)).astype(np.float32)
        return feather_mask(mask, sigma=feather_sigma) * support

    def _cluster_problem_regions(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        prior: np.ndarray,
        allowed: np.ndarray,
        min_prior: float,
        percentile: float,
        feather_sigma: float,
        max_coverage: float,
    ) -> np.ndarray:
        height, width = prior.shape
        allowed = (allowed > 0.05).astype(np.float32)
        allowed_count = int(np.count_nonzero(allowed))
        if allowed_count < 25:
            return np.zeros_like(prior, dtype=np.float32)

        base_threshold = max(min_prior, float(np.percentile(prior[allowed > 0], percentile)))
        high_prior = ((prior >= base_threshold) & (allowed > 0)).astype(np.float32)

        if int(np.count_nonzero(high_prior)) < 20:
            high_prior = ((prior >= min_prior) & (allowed > 0)).astype(np.float32)

        clustered = self._kmeans_select(image_rgb, analysis, prior, allowed)
        mask = np.maximum(high_prior, clustered)
        mask = self._limit_coverage(mask, prior, allowed, max_coverage)
        mask = self._cleanup(mask)
        mask = self._limit_coverage(mask, prior, allowed, max_coverage)
        if float(np.mean(mask)) < 0.001:
            return np.zeros_like(prior, dtype=np.float32)
        support = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)).astype(np.float32)
        return feather_mask(mask, sigma=feather_sigma) * support

    def _kmeans_select(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        prior: np.ndarray,
        allowed: np.ndarray,
    ) -> np.ndarray:
        height, width = prior.shape
        scale = min(1.0, 360.0 / max(height, width))
        small_size = (max(1, int(width * scale)), max(1, int(height * scale)))

        def resize_map(values: np.ndarray, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
            return cv2.resize(values.astype(np.float32), small_size, interpolation=interpolation)

        image_small = cv2.resize(image_rgb, small_size, interpolation=cv2.INTER_AREA)
        prior_small = resize_map(prior)
        allowed_small = resize_map(allowed, cv2.INTER_NEAREST) > 0.05
        if int(np.count_nonzero(allowed_small)) < 20:
            return np.zeros_like(prior, dtype=np.float32)

        hsv_small = cv2.cvtColor(image_small, cv2.COLOR_RGB2HSV).astype(np.float32)
        yy = np.linspace(0.0, 1.0, small_size[1], dtype=np.float32)[:, None]
        yy = np.broadcast_to(yy, prior_small.shape)

        features = np.stack(
            [
                prior_small,
                resize_map(analysis.luminance / 255.0),
                resize_map(analysis.gradient),
                resize_map(analysis.contrast),
                resize_map(analysis.edges),
                hsv_small[..., 1] / 255.0,
                yy,
            ],
            axis=-1,
        )

        flat_features = features[allowed_small].astype(np.float32)
        if flat_features.shape[0] < 80:
            return np.zeros_like(prior, dtype=np.float32)

        cluster_count = min(4, max(2, flat_features.shape[0] // 80))
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            24,
            0.025,
        )
        _compactness, labels, _centers = cv2.kmeans(
            flat_features,
            cluster_count,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )

        labels = labels.reshape(-1)
        flat_prior = prior_small[allowed_small]
        cluster_scores = []
        for cluster_index in range(cluster_count):
            selected = labels == cluster_index
            if int(np.count_nonzero(selected)) == 0:
                cluster_scores.append(-1.0)
                continue
            cluster_scores.append(float(np.mean(flat_prior[selected])))

        best_cluster = int(np.argmax(cluster_scores))
        if cluster_scores[best_cluster] < 0.035:
            return np.zeros_like(prior, dtype=np.float32)

        small_mask = np.zeros_like(prior_small, dtype=np.float32)
        small_mask[allowed_small] = (labels == best_cluster).astype(np.float32)
        small_mask *= (prior_small >= max(0.02, np.percentile(flat_prior, 60))).astype(np.float32)
        return cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    def _cleanup(self, mask: np.ndarray, min_component_ratio: float = 0.0015) -> np.ndarray:
        mask_u8 = (mask > 0.18).astype(np.uint8)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
        if component_count <= 1:
            return mask_u8.astype(np.float32)

        image_area = mask.shape[0] * mask.shape[1]
        cleaned = np.zeros_like(mask_u8)
        min_area = max(18, int(image_area * min_component_ratio))
        for component_index in range(1, component_count):
            if stats[component_index, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == component_index] = 1
        return cleaned.astype(np.float32)

    def _limit_total_coverage(
        self,
        mask: np.ndarray,
        prior: np.ndarray,
        max_coverage: float,
    ) -> np.ndarray:
        selected = mask > 0.1
        selected_count = int(np.count_nonzero(selected))
        max_pixels = max(1, int(mask.size * max_coverage))
        if selected_count <= max_pixels:
            return selected.astype(np.float32)

        selected_indices = np.flatnonzero(selected.reshape(-1))
        selected_values = prior.reshape(-1)[selected_indices]
        top_positions = np.argpartition(selected_values, -max_pixels)[-max_pixels:]
        top_indices = selected_indices[top_positions]
        limited = np.zeros(mask.size, dtype=np.float32)
        limited[top_indices] = 1.0
        return limited.reshape(mask.shape)

    def _limit_coverage(
        self,
        mask: np.ndarray,
        prior: np.ndarray,
        allowed: np.ndarray,
        max_coverage: float,
    ) -> np.ndarray:
        allowed_count = max(1, int(np.count_nonzero(allowed)))
        max_pixels = max(1, int(allowed_count * max_coverage))
        selected = (mask > 0.1) & (allowed > 0)
        selected_count = int(np.count_nonzero(selected))
        if selected_count <= max_pixels:
            return selected.astype(np.float32)

        selected_indices = np.flatnonzero(selected.reshape(-1))
        selected_values = prior.reshape(-1)[selected_indices]
        top_positions = np.argpartition(selected_values, -max_pixels)[-max_pixels:]
        top_indices = selected_indices[top_positions]
        limited = np.zeros(mask.size, dtype=np.float32)
        limited[top_indices] = 1.0
        return limited.reshape(mask.shape)
