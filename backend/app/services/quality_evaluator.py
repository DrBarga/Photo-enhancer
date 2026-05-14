import cv2
import numpy as np

from app.models.schemas import AnalysisMaps, MetricResult
from app.utils.map_math import clamp_score


class QualityEvaluator:
    def evaluate(
        self,
        before: AnalysisMaps,
        after: AnalysisMaps,
        original_rgb: np.ndarray | None = None,
        result_rgb: np.ndarray | None = None,
        target_masks: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[MetricResult], int]:
        metrics = [
            self._problem_metric(
                key="gradient_smoothness",
                label="Плавность градиента",
                before_value=float(before.gradient_problem.mean()),
                after_value=float(after.gradient_problem.mean()),
                description="Резкие переходы, пятна, banding и нарушенная плавность фона.",
            ),
            self._problem_metric(
                key="banding_reduction",
                label="Снижение banding",
                before_value=float(before.banding.mean()),
                after_value=float(after.banding.mean()),
                description="Снижение дискретных полос в плавных световых переходах.",
            ),
            self._problem_metric(
                key="reflection_coherence",
                label="Натуральность отражения",
                before_value=float(before.reflection_problem.mean()),
                after_value=float(after.reflection_problem.mean()),
                description="Согласованность отражения с поверхностью, деталями и бликами.",
            ),
            self._problem_metric(
                key="shadow_cleanliness",
                label="Чистота теней",
                before_value=float(before.shadow_noise.mean()),
                after_value=float(after.shadow_noise.mean()),
                description="Шум, грязь и рваные края в темных зонах.",
            ),
            self._problem_metric(
                key="cast_shadow_realism",
                label="Реалистичность тени",
                before_value=float(before.cast_shadow_problem.mean()),
                after_value=float(after.cast_shadow_problem.mean()),
                description="Наличие и согласованность падающей тени с объектом, глубиной и светом.",
            ),
            self._problem_metric(
                key="heatmap_risk",
                label="Индекс проблемных зон",
                before_value=float(before.problem_map.mean()),
                after_value=float(after.problem_map.mean()),
                description="Сводная карта проблем: градиенты, banding, пересвет, тени и отражения.",
            ),
        ]

        if original_rgb is not None and result_rgb is not None and target_masks:
            metrics.extend(self._image_metrics(before, after, original_rgb, result_rgb, target_masks))

        weights = {
            "locality_precision": 1.35,
            "edge_preservation": 1.20,
            "depth_consistency": 1.10,
            "artifact_regression": 1.25,
        }
        weighted_total = sum(metric.value * weights.get(metric.key, 1.0) for metric in metrics)
        weight_sum = sum(weights.get(metric.key, 1.0) for metric in metrics)
        total = clamp_score(weighted_total / max(weight_sum, 1e-6))
        return metrics, total

    def _problem_metric(
        self,
        key: str,
        label: str,
        before_value: float,
        after_value: float,
        description: str,
    ) -> MetricResult:
        before_quality = clamp_score((1.0 - before_value) * 100.0)
        after_quality = clamp_score((1.0 - after_value) * 100.0)
        improvement = 0.0
        if before_value > 1e-5:
            improvement = max(0.0, (before_value - after_value) / before_value)
        score = clamp_score(after_quality + improvement * 12.0)
        return MetricResult(
            key=key,
            label=label,
            value=score,
            before=before_quality,
            after=after_quality,
            description=description,
        )

    def _image_metrics(
        self,
        before: AnalysisMaps,
        after: AnalysisMaps,
        original_rgb: np.ndarray,
        result_rgb: np.ndarray,
        target_masks: dict[str, np.ndarray],
    ) -> list[MetricResult]:
        target_mask = self._combined_mask(target_masks)
        changed = self._change_map(original_rgb, result_rgb)
        outside_mask = np.clip(1.0 - target_mask, 0.0, 1.0)
        inside_change = float(np.sum(changed * target_mask) / (np.sum(target_mask) + 1e-6))
        outside_change = float(np.sum(changed * outside_mask) / (np.sum(outside_mask) + 1e-6))
        locality = clamp_score((1.0 - outside_change * 4.0 + min(inside_change * 2.5, 0.18)) * 100.0)

        before_edges = cv2.Canny(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY), 55, 145).astype(np.float32) / 255.0
        after_edges = cv2.Canny(cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY), 55, 145).astype(np.float32) / 255.0
        edge_delta = np.abs(before_edges - after_edges)
        edge_loss = float(np.sum(edge_delta * outside_mask) / (np.sum(outside_mask) + 1e-6))
        edge_score = clamp_score((1.0 - edge_loss * 3.8) * 100.0)

        depth_delta = np.abs(after.depth - before.depth)
        depth_outside = float(np.sum(depth_delta * outside_mask) / (np.sum(outside_mask) + 1e-6))
        depth_inside = float(np.sum(depth_delta * target_mask) / (np.sum(target_mask) + 1e-6))
        depth_score = clamp_score((1.0 - depth_outside * 2.4 - max(0.0, depth_inside - 0.12) * 1.4) * 100.0)

        color_score = self._color_naturalness(original_rgb, result_rgb, outside_mask)
        artifact_score = self._artifact_regression_score(before, after)

        return [
            MetricResult(
                key="locality_precision",
                label="Точность локальной правки",
                value=locality,
                before=clamp_score((1.0 - outside_change * 4.0) * 100.0),
                after=locality,
                description="Насколько изменения сосредоточены в найденных проблемных зонах.",
            ),
            MetricResult(
                key="edge_preservation",
                label="Сохранение краев",
                value=edge_score,
                before=100,
                after=edge_score,
                description="Контроль, что обработка не разрушает края объектов вне маски.",
            ),
            MetricResult(
                key="depth_consistency",
                label="Depth-стабильность",
                value=depth_score,
                before=100,
                after=depth_score,
                description="Проверка, что правка не ломает глубину сцены вне целевой зоны.",
            ),
            MetricResult(
                key="color_naturalness",
                label="Естественность цвета",
                value=color_score,
                before=100,
                after=color_score,
                description="Оценка цветового сдвига и защита от неестественных оттенков.",
            ),
            MetricResult(
                key="artifact_regression",
                label="Контроль новых артефактов",
                value=artifact_score,
                before=clamp_score((1.0 - float(before.problem_map.mean())) * 100.0),
                after=clamp_score((1.0 - float(after.problem_map.mean())) * 100.0),
                description="Штраф, если после обработки на heatmap появляются новые проблемные зоны.",
            ),
        ]

    def _combined_mask(self, masks: dict[str, np.ndarray]) -> np.ndarray:
        if not masks:
            return np.zeros((1, 1), dtype=np.float32)
        combined = np.maximum.reduce([np.clip(mask.astype(np.float32), 0.0, 1.0) for mask in masks.values()])
        return cv2.GaussianBlur(combined, (0, 0), sigmaX=2.0)

    def _change_map(self, original_rgb: np.ndarray, result_rgb: np.ndarray) -> np.ndarray:
        original_lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        result_lab = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        diff = np.linalg.norm(original_lab - result_lab, axis=2) / 255.0
        return np.clip(diff, 0.0, 1.0)

    def _color_naturalness(
        self,
        original_rgb: np.ndarray,
        result_rgb: np.ndarray,
        outside_mask: np.ndarray,
    ) -> int:
        original_hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        result_hsv = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation_shift = np.abs(result_hsv[..., 1] - original_hsv[..., 1]) / 255.0
        value_shift = np.abs(result_hsv[..., 2] - original_hsv[..., 2]) / 255.0
        outside_shift = float(np.sum((0.56 * saturation_shift + 0.44 * value_shift) * outside_mask) / (np.sum(outside_mask) + 1e-6))
        return clamp_score((1.0 - outside_shift * 3.2) * 100.0)

    def _artifact_regression_score(self, before: AnalysisMaps, after: AnalysisMaps) -> int:
        regression = float(np.mean(np.clip(after.problem_map - before.problem_map, 0.0, 1.0)))
        improvement = float(np.mean(np.clip(before.problem_map - after.problem_map, 0.0, 1.0)))
        score = 100.0 - regression * 420.0 + min(improvement * 180.0, 12.0)
        return clamp_score(score)
