import cv2
import numpy as np

from app.models.schemas import AnalysisMaps, PromptParameters
from app.services.ml_providers import get_ml_services
from app.services.smart_masks import SmartMaskBuilder
from app.utils.map_math import blend_by_mask, feather_mask, normalize_map


smart_masks = SmartMaskBuilder()


class GradientProcessor:
    def apply(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        prompt: PromptParameters,
    ) -> np.ndarray:
        mask, object_protection = self.target_mask(image_rgb, analysis, prompt)
        if float(np.mean(mask)) < 0.001:
            return image_rgb

        has_user_gradient = len(prompt.colors) >= 2
        smooth = self._edge_aware_background_smooth(image_rgb, analysis, mask, prompt)
        corrected = smooth.astype(np.float32)

        active_band = analysis.banding[mask > 0.08] if int(np.count_nonzero(mask > 0.08)) else analysis.banding.reshape(-1)
        if (not has_user_gradient) and prompt.banding_fix and float(np.mean(active_band)) > 0.065:
            corrected = self._add_controlled_dither(corrected, mask, amount=0.55 + prompt.intensity * 0.75)

        if has_user_gradient:
            target_gradient = self._build_target_gradient(image_rgb.shape, prompt)
            corrected = self._add_controlled_dither(target_gradient, mask, amount=0.34)
        else:
            tonal_gradient = self._build_tonal_gradient(image_rgb)
            corrected = corrected * 0.50 + tonal_gradient * 0.50

        if not has_user_gradient:
            corrected = self._keep_background_detail(image_rgb, corrected, mask)
            corrected = self._remove_white_patches(image_rgb, corrected, max_luma_lift=24.0)
        blend_multiplier = 1.0 if has_user_gradient else 0.88 + prompt.softness * 0.08 + prompt.intensity * 0.06
        protection = 1.0 if has_user_gradient else object_protection
        blend_strength = np.clip(mask * protection * blend_multiplier, 0.0, 1.0)
        return blend_by_mask(image_rgb, corrected, blend_strength)

    def target_mask(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        prompt: PromptParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        seed_mask = smart_masks.gradient_mask(image_rgb, analysis)
        mask, object_protection = self._gradient_work_mask(image_rgb, analysis, seed_mask, prompt)
        return mask, object_protection

    def _gradient_work_mask(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        seed_mask: np.ndarray,
        prompt: PromptParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        segments = get_ml_services().segmentation.masks(image_rgb, analysis).value
        surface = segments.get("surface", np.ones_like(analysis.luminance, dtype=np.float32))
        object_mask = segments.get("object", np.zeros_like(analysis.luminance, dtype=np.float32))
        object_block = cv2.dilate(
            (object_mask > 0.10).astype(np.uint8),
            np.ones((15, 15), dtype=np.uint8),
            iterations=1,
        ).astype(np.float32)
        object_block = np.maximum(object_block, self._foreground_detail_guard(analysis))
        object_block = feather_mask(object_block, sigma=3.2)
        object_protection = np.clip(1.0 - object_block * 0.96, 0.0, 1.0)

        if len(prompt.colors) >= 2:
            return self._full_gradient_background_mask(
                image_rgb,
                analysis,
                seed_mask,
                surface,
                object_protection,
            ), object_protection

        smooth_surface = (
            (analysis.smooth_background > 0.16)
            & (analysis.edges < 0.34)
            & (analysis.texture < 0.72)
        ).astype(np.float32)
        smooth_surface *= (0.24 + 0.76 * surface) * object_protection

        values = analysis.gradient_problem[smooth_surface > 0.05]
        if values.size:
            threshold = max(0.030, float(np.percentile(values, 62 if len(prompt.colors) >= 2 else 72)))
        else:
            threshold = 0.040
        problem_surface = ((analysis.gradient_problem >= threshold) | (seed_mask > 0.035)).astype(np.float32)
        problem_surface *= smooth_surface

        seed = ((seed_mask > 0.035) | (problem_surface > 0.2)).astype(np.uint8)
        if int(np.count_nonzero(seed)) < 20:
            return seed_mask, object_protection

        kernel_size = 41 if len(prompt.colors) >= 2 else 25
        expanded = cv2.dilate(seed, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1).astype(np.float32)
        expanded *= smooth_surface
        expanded = self._keep_largest_surface_regions(expanded, analysis.gradient_problem, max_regions=4)
        expanded = feather_mask(expanded, sigma=7.0 if len(prompt.colors) >= 2 else 4.5)

        mask = np.maximum(seed_mask, expanded * (0.92 if len(prompt.colors) >= 2 else 0.68))
        mask = np.clip(mask * smooth_surface + seed_mask * 0.35, 0.0, 1.0)
        return mask, object_protection

    def _full_gradient_background_mask(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        seed_mask: np.ndarray,
        surface: np.ndarray,
        object_protection: np.ndarray,
    ) -> np.ndarray:
        letterbox_keep = self._letterbox_keep_mask(image_rgb, analysis)
        background_candidate = (
            (
                (analysis.smooth_background > 0.08)
                | (analysis.gradient_problem > 0.020)
                | (seed_mask > 0.025)
            )
            & (analysis.edges < 0.50)
            & (analysis.texture < 0.88)
        ).astype(np.float32)
        background_candidate *= (0.18 + 0.82 * surface) * object_protection * letterbox_keep

        seed = ((seed_mask > 0.025) | ((analysis.gradient_problem > 0.040) & (background_candidate > 0.05))).astype(np.uint8)
        if int(np.count_nonzero(seed)) < 20:
            seed = (background_candidate > 0.12).astype(np.uint8)

        closed = cv2.morphologyEx(
            (background_candidate > 0.05).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((33, 33), dtype=np.uint8),
            iterations=1,
        )
        closed = cv2.dilate(closed, np.ones((13, 13), dtype=np.uint8), iterations=1)
        closed = self._fill_internal_holes(closed)
        closed = (closed.astype(np.float32) * letterbox_keep * object_protection > 0.05).astype(np.uint8)

        selected = self._select_background_components(closed, seed, analysis.gradient_problem)
        if int(np.count_nonzero(selected)) < 20:
            selected = closed
        selected = cv2.morphologyEx(
            selected.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((29, 29), dtype=np.uint8),
            iterations=1,
        ).astype(np.float32)
        selected *= letterbox_keep * object_protection
        selected = self._limit_background_spill(selected, image_rgb, analysis, seed_mask)
        mask = feather_mask(selected, sigma=4.8)
        return np.clip(mask * letterbox_keep * object_protection, 0.0, 1.0)

    def _letterbox_keep_mask(self, image_rgb: np.ndarray, analysis: AnalysisMaps) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        dark_neutral = ((analysis.luminance < 14.0) & (hsv[..., 1] < 34.0)).astype(np.uint8)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_neutral, 8)
        if component_count <= 1:
            return np.ones_like(analysis.luminance, dtype=np.float32)

        height, width = analysis.luminance.shape
        border_block = np.zeros_like(dark_neutral, dtype=np.uint8)
        min_area = max(24, int(height * width * 0.006))
        for component_index in range(1, component_count):
            x = stats[component_index, cv2.CC_STAT_LEFT]
            y = stats[component_index, cv2.CC_STAT_TOP]
            w = stats[component_index, cv2.CC_STAT_WIDTH]
            h = stats[component_index, cv2.CC_STAT_HEIGHT]
            area = stats[component_index, cv2.CC_STAT_AREA]
            touches_border = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
            if touches_border and area >= min_area:
                border_block[labels == component_index] = 1
        border_block = cv2.dilate(border_block, np.ones((5, 5), dtype=np.uint8), iterations=1).astype(np.float32)
        return np.clip(1.0 - feather_mask(border_block, sigma=2.2), 0.0, 1.0)

    def _fill_internal_holes(self, mask_u8: np.ndarray) -> np.ndarray:
        inverse = (mask_u8 == 0).astype(np.uint8)
        component_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(inverse, 8)
        filled = mask_u8.copy()
        height, width = mask_u8.shape
        for component_index in range(1, component_count):
            component = labels == component_index
            ys, xs = np.where(component)
            if ys.size == 0:
                continue
            touches_border = xs.min() == 0 or ys.min() == 0 or xs.max() == width - 1 or ys.max() == height - 1
            if not touches_border:
                filled[component] = 1
        return filled

    def _select_background_components(
        self,
        mask_u8: np.ndarray,
        seed_u8: np.ndarray,
        priority: np.ndarray,
    ) -> np.ndarray:
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
        if component_count <= 1:
            return mask_u8.astype(np.float32)

        regions: list[tuple[float, int]] = []
        image_area = float(mask_u8.size)
        for component_index in range(1, component_count):
            area = float(stats[component_index, cv2.CC_STAT_AREA])
            if area < max(32.0, image_area * 0.001):
                continue
            component = labels == component_index
            seed_overlap = float(np.mean(seed_u8[component] > 0))
            problem_mean = float(np.mean(priority[component]))
            score = area * (0.40 + seed_overlap * 1.80 + problem_mean * 1.25)
            regions.append((score, component_index))
        if not regions:
            return np.zeros_like(mask_u8, dtype=np.float32)

        selected = np.zeros_like(mask_u8, dtype=np.uint8)
        for _score, component_index in sorted(regions, reverse=True)[:3]:
            selected[labels == component_index] = 1
        return selected.astype(np.float32)

    def _limit_background_spill(
        self,
        mask: np.ndarray,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        seed_mask: np.ndarray,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[..., 1] / 255.0
        chroma_support = cv2.GaussianBlur(saturation, (0, 0), sigmaX=9.0, sigmaY=9.0)
        likely_background = (
            (analysis.smooth_background > 0.06)
            | (analysis.gradient_problem > 0.018)
            | (seed_mask > 0.02)
            | (chroma_support > 0.18)
        ).astype(np.float32)
        likely_background = cv2.morphologyEx(
            likely_background.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((21, 21), dtype=np.uint8),
            iterations=1,
        ).astype(np.float32)
        return np.clip(mask * feather_mask(likely_background, sigma=3.0), 0.0, 1.0)

    def _foreground_detail_guard(self, analysis: AnalysisMaps) -> np.ndarray:
        detail = ((analysis.edges > 0.10) | (analysis.texture > 0.82)).astype(np.uint8)
        detail = cv2.dilate(detail, np.ones((7, 7), dtype=np.uint8), iterations=1).astype(np.float32)
        return feather_mask(detail, sigma=2.0) * 0.72

    def _keep_largest_surface_regions(
        self,
        mask: np.ndarray,
        priority: np.ndarray,
        max_regions: int,
    ) -> np.ndarray:
        mask_u8 = (mask > 0.12).astype(np.uint8)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
        if component_count <= 1:
            return mask_u8.astype(np.float32)

        regions: list[tuple[float, int]] = []
        for component_index in range(1, component_count):
            area = float(stats[component_index, cv2.CC_STAT_AREA])
            if area < max(20.0, mask.size * 0.00055):
                continue
            component = labels == component_index
            score = area * (0.55 + float(np.mean(priority[component])) * 0.45)
            regions.append((score, component_index))
        if not regions:
            return np.zeros_like(mask, dtype=np.float32)

        selected = {index for _score, index in sorted(regions, reverse=True)[:max_regions]}
        kept = np.zeros_like(mask_u8)
        for index in selected:
            kept[labels == index] = 1
        return kept.astype(np.float32)

    def _edge_aware_background_smooth(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        mask: np.ndarray,
        prompt: PromptParameters,
    ) -> np.ndarray:
        bilateral = cv2.bilateralFilter(image_rgb, d=17, sigmaColor=62, sigmaSpace=92)
        wide = cv2.GaussianBlur(bilateral, (0, 0), sigmaX=5.0 + prompt.softness * 3.0, sigmaY=5.0 + prompt.softness * 3.0)
        medium = cv2.GaussianBlur(bilateral, (0, 0), sigmaX=1.1 + prompt.softness * 1.5, sigmaY=1.1 + prompt.softness * 1.5)
        problem_strength = np.clip(0.38 + analysis.gradient_problem[..., None] * 0.62 + mask[..., None] * 0.20, 0.0, 1.0)
        smooth = medium * (1.0 - problem_strength) + wide * problem_strength
        return np.clip(smooth, 0, 255).astype(np.float32)

    def _keep_background_detail(
        self,
        original_rgb: np.ndarray,
        corrected: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        original_float = original_rgb.astype(np.float32)
        base = cv2.GaussianBlur(original_float, (0, 0), sigmaX=1.4, sigmaY=1.4)
        detail = original_float - base
        detail_amount = (1.0 - np.clip(mask, 0.0, 1.0))[..., None] * 0.20 + 0.08
        return np.clip(corrected + detail * detail_amount, 0, 255)

    def _add_controlled_dither(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        amount: float,
    ) -> np.ndarray:
        height, width = mask.shape
        yy, xx = np.indices((height, width))
        pattern = (((xx * 13 + yy * 7) % 11) / 10.0 - 0.5) * amount
        dither = pattern[..., None] * mask[..., None]
        return np.clip(image + dither, 0, 255)

    def _build_tonal_gradient(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        top_color = np.percentile(image_rgb[: max(1, height // 3)], 68, axis=(0, 1))
        bottom_color = np.percentile(image_rgb[-max(1, height // 3) :], 42, axis=(0, 1))
        alpha = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        gradient = top_color * (1.0 - alpha) + bottom_color * alpha
        return np.broadcast_to(gradient, (height, width, 3)).astype(np.float32)

    def _build_target_gradient(
        self,
        shape: tuple[int, int, int],
        prompt: PromptParameters,
    ) -> np.ndarray:
        height, width = shape[:2]
        colors = [np.array(color, dtype=np.float32) for color in prompt.colors[:3]]
        if len(colors) == 2 and prompt.gradient_stops == 3:
            middle = (colors[0] * 0.55 + colors[1] * 0.45 + np.array([18, 18, 18], dtype=np.float32)) / 1.10
            colors.insert(1, np.clip(middle, 0, 255))

        if prompt.gradient_style == "radial" or prompt.direction == "radial":
            y, x = np.indices((height, width), dtype=np.float32)
            cx = (width - 1) * 0.50
            cy = (height - 1) * 0.46
            alpha = normalize_map(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))
        elif prompt.direction == "horizontal":
            alpha = np.broadcast_to(np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :], (height, width))
        elif prompt.direction == "diagonal":
            y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
            x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
            alpha = (x + y) / 2.0
        else:
            alpha = np.broadcast_to(np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None], (height, width))

        if len(colors) >= 3:
            first_half = alpha <= 0.5
            second_alpha = np.clip((alpha - 0.5) * 2.0, 0.0, 1.0)
            first_alpha = np.clip(alpha * 2.0, 0.0, 1.0)
            first = colors[0] * (1.0 - first_alpha[..., None]) + colors[1] * first_alpha[..., None]
            second = colors[1] * (1.0 - second_alpha[..., None]) + colors[2] * second_alpha[..., None]
            return np.where(first_half[..., None], first, second).astype(np.float32)

        return (colors[0] * (1.0 - alpha[..., None]) + colors[1] * alpha[..., None]).astype(np.float32)

    def _preserve_scene_luminance(
        self,
        original_rgb: np.ndarray,
        target_rgb: np.ndarray,
        preserve_ratio: float,
    ) -> np.ndarray:
        original_lab = cv2.cvtColor(np.clip(original_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(np.clip(target_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab[..., 0] = original_lab[..., 0] * preserve_ratio + target_lab[..., 0] * (1.0 - preserve_ratio)
        return cv2.cvtColor(np.clip(target_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    def _remove_white_patches(
        self,
        original_rgb: np.ndarray,
        corrected: np.ndarray,
        max_luma_lift: float = 18.0,
    ) -> np.ndarray:
        original_luma = (
            0.299 * original_rgb[..., 0] + 0.587 * original_rgb[..., 1] + 0.114 * original_rgb[..., 2]
        ).astype(np.float32)
        corrected_luma = (
            0.299 * corrected[..., 0] + 0.587 * corrected[..., 1] + 0.114 * corrected[..., 2]
        ).astype(np.float32)
        excessive = np.clip((corrected_luma - original_luma - max_luma_lift) / 42.0, 0.0, 1.0)
        return corrected * (1.0 - excessive[..., None] * 0.72) + original_rgb.astype(np.float32) * (excessive[..., None] * 0.72)


class ReflectionProcessor:
    material_profiles = {
        "water": {
            "sharpness": 0.56,
            "ripple": 5.8,
            "opacity": 0.50,
            "texture": 0.18,
            "sheen": 0.30,
            "vertical_scale": 0.80,
            "fade_end": 0.10,
        },
        "asphalt": {
            "sharpness": 0.40,
            "ripple": 1.9,
            "opacity": 0.38,
            "texture": 0.46,
            "sheen": 0.20,
            "vertical_scale": 0.68,
            "fade_end": 0.06,
        },
        "mirror": {
            "sharpness": 0.96,
            "ripple": 0.1,
            "opacity": 0.82,
            "texture": 0.03,
            "sheen": 0.14,
            "vertical_scale": 0.96,
            "fade_end": 0.36,
        },
        "glass": {
            "sharpness": 0.84,
            "ripple": 0.45,
            "opacity": 0.68,
            "texture": 0.08,
            "sheen": 0.22,
            "vertical_scale": 0.88,
            "fade_end": 0.18,
        },
    }

    def apply(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        prompt: PromptParameters,
    ) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        geometry = self._reflection_geometry(image_rgb, analysis, prompt)
        mask = geometry["surface_mask"]
        plane_y = int(geometry["plane_y"])
        if plane_y >= height - 4 or float(np.mean(mask)) < 0.001:
            return image_rgb

        material = str(geometry["material"])
        profile = self.material_profiles.get(material, self.material_profiles["glass"])
        profile = {
            **profile,
            "opacity": min(0.98, profile["opacity"] + prompt.reflection_strength * 0.34),
        }

        reflection_rgb, reflection_alpha = self._object_reflection_layer(
            image_rgb,
            geometry["source_mask"],
            plane_y,
            material,
            profile,
        )
        if float(np.mean(reflection_alpha)) < 0.001:
            reflection_rgb, reflection_alpha = self._fallback_scene_reflection(
                image_rgb,
                plane_y,
                material,
                profile,
            )
        if float(np.mean(reflection_alpha)) < 0.001:
            return image_rgb

        corrected = image_rgb.astype(np.float32).copy()
        lower_original = corrected[plane_y:, :, :]
        lower_reflection = reflection_rgb[plane_y:, :, :]
        lower_mask = mask[plane_y:, :]
        surface_gate = np.clip((lower_mask - 0.02) / 0.24, 0.0, 1.0)
        lower_alpha = np.clip(
            reflection_alpha[plane_y:, :]
            * (0.48 + 0.52 * surface_gate)
            * (lower_mask > 0.02).astype(np.float32)
            * profile["opacity"],
            0.0,
            0.98,
        )
        target = self._compose_reflection(lower_original, lower_reflection, lower_alpha, material)
        target = self._preserve_surface_detail(lower_original, target, profile["texture"])
        target += self._surface_sheen(analysis, mask, plane_y, material, profile["sheen"])[..., None]
        corrected[plane_y:, :, :] = np.clip(target, 0, 255)

        local_result = np.clip(corrected, 0, 255).astype(np.uint8)
        return self._maybe_inpaint_reflection(local_result, mask, material, prompt, analysis)

    def target_mask(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        prompt: PromptParameters,
    ) -> np.ndarray:
        return self._reflection_geometry(image_rgb, analysis, prompt)["surface_mask"]

    def _reflection_geometry(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        prompt: PromptParameters,
    ) -> dict[str, object]:
        height, width = image_rgb.shape[:2]
        material = analysis.reflection_material if prompt.reflection_material == "auto" else prompt.reflection_material
        material = material if material in self.material_profiles else "glass"

        segments = get_ml_services().segmentation.masks(image_rgb, analysis).value
        sam_object = segments.get("object", np.zeros_like(analysis.luminance, dtype=np.float32))
        sam_coverage = float(np.mean(sam_object > 0.10))
        cv_object = self._cv_foreground_mask(image_rgb, analysis)
        if sam_coverage > 0.55:
            object_mask = cv_object
            surface_hint = np.ones_like(analysis.luminance, dtype=np.float32)
        else:
            object_mask = np.maximum(sam_object, cv_object)
            surface_hint = segments.get("surface", np.ones_like(analysis.luminance, dtype=np.float32))
        content_keep = self._content_keep_mask(image_rgb, analysis)
        object_mask = feather_mask((object_mask > 0.10).astype(np.float32), sigma=1.6) * content_keep

        source_mask = self._source_object_mask(object_mask, analysis, content_keep)
        plane_y = self._estimate_reflection_plane(source_mask, analysis, height)
        surface_mask = self._reflective_surface_mask(
            image_rgb,
            analysis,
            plane_y,
            source_mask,
            object_mask,
            surface_hint,
            content_keep,
            material,
        )
        return {
            "surface_mask": surface_mask,
            "source_mask": source_mask,
            "plane_y": plane_y,
            "material": material,
        }

    def _cv_foreground_mask(self, image_rgb: np.ndarray, analysis: AnalysisMaps) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        saliency = np.clip(
            0.34 * analysis.edges
            + 0.30 * analysis.texture
            + 0.20 * normalize_map(hsv[..., 1])
            + 0.16 * (1.0 - analysis.smooth_background),
            0.0,
            1.0,
        )
        threshold = max(0.20, float(np.percentile(saliency, 76)))
        mask = (saliency >= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
        return feather_mask(mask.astype(np.float32), sigma=1.8)

    def _source_object_mask(
        self,
        object_mask: np.ndarray,
        analysis: AnalysisMaps,
        content_keep: np.ndarray,
    ) -> np.ndarray:
        detail = (
            (analysis.edges > 0.045)
            | (analysis.texture > 0.26)
            | (analysis.contrast > 0.12)
        ).astype(np.float32)
        detail = feather_mask(cv2.dilate(detail.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)).astype(np.float32), sigma=1.4)
        source = np.clip(object_mask * (0.45 + 0.55 * detail) * content_keep, 0.0, 1.0)
        if int(np.count_nonzero(source > 0.12)) < 40:
            source = np.clip(object_mask * content_keep, 0.0, 1.0)
        solid_source = cv2.morphologyEx(
            (source > 0.10).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((17, 17), dtype=np.uint8),
            iterations=1,
        ).astype(np.float32)
        solid_source = cv2.dilate(solid_source.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(np.float32)
        source = np.maximum(source, feather_mask(solid_source, sigma=2.2) * 0.86 * content_keep)
        source = self._dense_reflection_source(source, object_mask, analysis, content_keep)
        cleaned = self._remove_surface_like_source_components(source)
        if int(np.count_nonzero(cleaned > 0.12)) >= 40:
            return cleaned
        return source

    def _dense_reflection_source(
        self,
        source: np.ndarray,
        object_mask: np.ndarray,
        analysis: AnalysisMaps,
        content_keep: np.ndarray,
    ) -> np.ndarray:
        height, width = source.shape
        seed = ((source > 0.08) | ((object_mask > 0.20) & (analysis.texture > 0.12))).astype(np.uint8)
        if int(np.count_nonzero(seed)) < 40:
            return source

        close_w = max(17, int(width * 0.035))
        close_h = max(13, int(height * 0.024))
        envelope = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((close_h, close_w), dtype=np.uint8), iterations=1)
        envelope = cv2.dilate(envelope, np.ones((7, 7), dtype=np.uint8), iterations=1)

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(envelope, 8)
        hull_mask = np.zeros_like(envelope, dtype=np.uint8)
        image_area = height * width
        for component_index in range(1, component_count):
            x = stats[component_index, cv2.CC_STAT_LEFT]
            y = stats[component_index, cv2.CC_STAT_TOP]
            w = stats[component_index, cv2.CC_STAT_WIDTH]
            h = stats[component_index, cv2.CC_STAT_HEIGHT]
            area = stats[component_index, cv2.CC_STAT_AREA]
            if area < max(24, int(image_area * 0.00045)):
                continue
            wide_surface = w > width * 0.48 and h < height * 0.22 and y > height * 0.38
            too_large = area > image_area * 0.42
            if wide_surface or too_large:
                continue
            component = (labels == component_index).astype(np.uint8)
            contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.shape[0] < 3:
                    continue
                hull = cv2.convexHull(contour)
                cv2.fillConvexPoly(hull_mask, hull, 1)

        if int(np.count_nonzero(hull_mask)) < 40:
            return source
        dense = feather_mask(hull_mask.astype(np.float32), sigma=3.0) * content_keep
        texture_gate = np.clip(0.48 + 0.52 * (analysis.texture + analysis.edges), 0.0, 1.0)
        dense = np.clip(dense * texture_gate, 0.0, 1.0)
        return np.maximum(source, dense * 0.92)

    def _remove_surface_like_source_components(self, source: np.ndarray) -> np.ndarray:
        mask_u8 = (source > 0.12).astype(np.uint8)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
        if component_count <= 1:
            return source
        height, width = source.shape
        kept = np.zeros_like(source, dtype=np.float32)
        for component_index in range(1, component_count):
            x = stats[component_index, cv2.CC_STAT_LEFT]
            y = stats[component_index, cv2.CC_STAT_TOP]
            w = stats[component_index, cv2.CC_STAT_WIDTH]
            h = stats[component_index, cv2.CC_STAT_HEIGHT]
            area = stats[component_index, cv2.CC_STAT_AREA]
            wide_surface = w > width * 0.42 and h < height * 0.26 and y > height * 0.38
            near_bottom_strip = y > height * 0.66 and w > width * 0.24
            too_large_surface = area > width * height * 0.38
            if wide_surface or near_bottom_strip or too_large_surface:
                continue
            kept[labels == component_index] = source[labels == component_index]
        return kept

    def _estimate_reflection_plane(
        self,
        source_mask: np.ndarray,
        analysis: AnalysisMaps,
        height: int,
    ) -> int:
        weighted = source_mask * (0.55 + 0.45 * analysis.edges)
        ys = np.where(weighted > 0.12)[0]
        if ys.size < 40:
            return int(height * 0.55)
        upper_limit = int(height * 0.74)
        upper_ys = ys[ys < upper_limit]
        if upper_ys.size >= max(40, int(ys.size * 0.30)):
            ys = upper_ys

        row_density = np.mean(weighted > 0.12, axis=1)
        dense_surface_rows = np.where((row_density > 0.18) & (np.arange(height) > height * 0.45))[0]
        if dense_surface_rows.size:
            surface_start = int(dense_surface_rows[0])
            filtered = ys[ys < surface_start]
            if filtered.size >= 40:
                ys = filtered

        plane = int(np.percentile(ys, 94))
        return int(np.clip(plane, height * 0.32, height * 0.80))

    def _reflective_surface_mask(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        plane_y: int,
        source_mask: np.ndarray,
        object_mask: np.ndarray,
        surface_hint: np.ndarray,
        content_keep: np.ndarray,
        material: str,
    ) -> np.ndarray:
        height, width = analysis.luminance.shape
        yy = np.arange(height, dtype=np.float32)[:, None]
        below_plane = (yy >= max(0, plane_y - int(height * 0.035))).astype(np.float32)
        below_plane = np.broadcast_to(below_plane, (height, width))

        object_block = cv2.dilate(
            (object_mask > 0.16).astype(np.uint8),
            np.ones((13, 13), dtype=np.uint8),
            iterations=1,
        ).astype(np.float32)
        object_protection = np.clip(1.0 - feather_mask(object_block, sigma=2.5) * 0.92, 0.0, 1.0)

        projection = np.zeros_like(source_mask, dtype=np.float32)
        source_columns = cv2.dilate(
            (np.max(source_mask, axis=0, keepdims=True) > 0.05).astype(np.uint8),
            np.ones((1, max(9, width // 14)), dtype=np.uint8),
            iterations=1,
        ).astype(np.float32)
        projection[plane_y:, :] = np.broadcast_to(source_columns, (height - plane_y, width))
        projection = feather_mask(projection, sigma=max(5.0, width * 0.018))

        surface_signal = np.clip(
            0.30 * analysis.reflection_mask
            + 0.24 * analysis.specular
            + 0.20 * analysis.smooth_background
            + 0.14 * (1.0 - analysis.texture)
            + 0.12 * (1.0 - analysis.depth),
            0.0,
            1.0,
        )
        candidate = (
            (surface_signal > 0.16)
            | (analysis.reflection_mask > 0.05)
            | ((analysis.smooth_background > 0.10) & (analysis.texture < 0.72))
            | (projection > 0.18)
        ).astype(np.float32)
        candidate *= below_plane * content_keep * object_protection
        candidate *= (0.22 + 0.50 * surface_hint + 0.28 * np.clip(projection + surface_signal, 0.0, 1.0))

        mask = cv2.morphologyEx(
            (candidate > 0.045).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((25, 25), dtype=np.uint8),
            iterations=1,
        ).astype(np.float32)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), dtype=np.uint8), iterations=1).astype(np.float32)
        mask = self._keep_reflection_components(mask, projection, surface_signal)
        if float(np.mean(mask)) < 0.002:
            fallback = np.zeros_like(mask, dtype=np.float32)
            start = int(np.clip(plane_y, 0, height - 1))
            fallback[start:, :] = 1.0
            mask = fallback * content_keep * object_protection
        mask = feather_mask(mask, sigma=4.5 if material in {"glass", "mirror"} else 5.8)
        vertical_ramp = np.clip((yy - plane_y) / max(height * 0.055, 1.0), 0.0, 1.0)
        vertical_ramp = np.broadcast_to(vertical_ramp, (height, width))
        return np.clip(mask * vertical_ramp * content_keep * object_protection, 0.0, 1.0)

    def _keep_reflection_components(
        self,
        mask: np.ndarray,
        projection: np.ndarray,
        surface_signal: np.ndarray,
    ) -> np.ndarray:
        mask_u8 = (mask > 0.08).astype(np.uint8)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
        if component_count <= 1:
            return mask_u8.astype(np.float32)
        image_area = mask.size
        regions: list[tuple[float, int]] = []
        for component_index in range(1, component_count):
            area = float(stats[component_index, cv2.CC_STAT_AREA])
            if area < max(24.0, image_area * 0.0009):
                continue
            component = labels == component_index
            score = area * (
                0.45
                + 0.90 * float(np.mean(projection[component]))
                + 0.55 * float(np.mean(surface_signal[component]))
            )
            regions.append((score, component_index))
        if not regions:
            return np.zeros_like(mask, dtype=np.float32)
        selected = np.zeros_like(mask_u8, dtype=np.uint8)
        for _score, component_index in sorted(regions, reverse=True)[:3]:
            selected[labels == component_index] = 1
        return selected.astype(np.float32)

    def _content_keep_mask(self, image_rgb: np.ndarray, analysis: AnalysisMaps) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        dark_neutral = ((analysis.luminance < 14.0) & (hsv[..., 1] < 34.0)).astype(np.uint8)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_neutral, 8)
        if component_count <= 1:
            return np.ones_like(analysis.luminance, dtype=np.float32)
        height, width = analysis.luminance.shape
        border_block = np.zeros_like(dark_neutral, dtype=np.uint8)
        min_area = max(24, int(height * width * 0.006))
        for component_index in range(1, component_count):
            x = stats[component_index, cv2.CC_STAT_LEFT]
            y = stats[component_index, cv2.CC_STAT_TOP]
            w = stats[component_index, cv2.CC_STAT_WIDTH]
            h = stats[component_index, cv2.CC_STAT_HEIGHT]
            area = stats[component_index, cv2.CC_STAT_AREA]
            touches_border = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
            if touches_border and area >= min_area:
                border_block[labels == component_index] = 1
        border_block = cv2.dilate(border_block, np.ones((5, 5), dtype=np.uint8), iterations=1).astype(np.float32)
        return np.clip(1.0 - feather_mask(border_block, sigma=2.2), 0.0, 1.0)

    def _object_reflection_layer(
        self,
        image_rgb: np.ndarray,
        source_mask: np.ndarray,
        plane_y: int,
        material: str,
        profile: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image_rgb.shape[:2]
        source_h = max(4, plane_y)
        source_rgb = image_rgb[:source_h, :, :].astype(np.float32)
        source_alpha = np.clip(source_mask[:source_h, :], 0.0, 1.0)
        if int(np.count_nonzero(source_alpha > 0.08)) < 30:
            return np.zeros_like(image_rgb, dtype=np.float32), np.zeros((height, width), dtype=np.float32)

        reflected_rgb = np.flipud(source_rgb)
        reflected_alpha = np.flipud(source_alpha)
        scaled_h = max(4, int(reflected_rgb.shape[0] * float(profile.get("vertical_scale", 0.86))))
        reflected_rgb = cv2.resize(reflected_rgb, (width, scaled_h), interpolation=cv2.INTER_LINEAR)
        reflected_alpha = cv2.resize(reflected_alpha, (width, scaled_h), interpolation=cv2.INTER_LINEAR)

        target_h = height - plane_y
        reflected_rgb = reflected_rgb[:target_h, :, :]
        reflected_alpha = reflected_alpha[:target_h, :]
        if reflected_rgb.shape[0] < target_h:
            pad_h = target_h - reflected_rgb.shape[0]
            reflected_rgb = np.pad(reflected_rgb, ((0, pad_h), (0, 0), (0, 0)), mode="edge")
            reflected_alpha = np.pad(reflected_alpha, ((0, pad_h), (0, 0)), mode="constant")

        reflected_rgb = self._material_warp(reflected_rgb, material, float(profile["ripple"]))
        blur_sigma = max(0.0, (1.0 - float(profile["sharpness"])) * 3.2)
        if blur_sigma > 0.10:
            reflected_rgb = cv2.GaussianBlur(reflected_rgb, (0, 0), sigmaX=blur_sigma, sigmaY=max(0.05, blur_sigma * 0.55))
        reflected_rgb = self._tone_reflected_layer(reflected_rgb, material, profile)

        fade = np.linspace(1.0, float(profile.get("fade_end", 0.16)), target_h, dtype=np.float32)[:, None]
        alpha_matte = feather_mask(reflected_alpha, sigma=1.25 if material in {"glass", "mirror"} else 2.2)
        contact = np.exp(-np.arange(target_h, dtype=np.float32)[:, None] / max(target_h * 0.23, 1.0))
        contact_boost = 0.86 + 0.22 * contact
        reflected_alpha = np.clip((alpha_matte ** 0.58) * fade * contact_boost, 0.0, 1.0)
        if material == "asphalt":
            reflected_alpha *= self._rough_surface_noise(reflected_alpha.shape, amount=0.34)

        layer = np.zeros_like(image_rgb, dtype=np.float32)
        alpha = np.zeros((height, width), dtype=np.float32)
        layer[plane_y:, :, :] = reflected_rgb
        alpha[plane_y:, :] = reflected_alpha
        return layer, alpha

    def _tone_reflected_layer(
        self,
        reflected_rgb: np.ndarray,
        material: str,
        profile: dict[str, float],
    ) -> np.ndarray:
        reflected = reflected_rgb.astype(np.float32)
        if material in {"mirror", "glass"}:
            soft = cv2.GaussianBlur(reflected, (0, 0), sigmaX=0.78, sigmaY=0.48)
            detail_gain = 0.20 + 0.18 * float(profile.get("sharpness", 0.8))
            reflected = np.clip(reflected * (1.0 + detail_gain) - soft * detail_gain, 0, 255)
            hsv = cv2.cvtColor(np.clip(reflected, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= 0.92 if material == "glass" else 0.98
            hsv[..., 2] = np.clip(hsv[..., 2] * (1.03 if material == "glass" else 1.00) + 2.0, 0, 255)
            reflected = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        elif material == "water":
            reflected = cv2.GaussianBlur(reflected, (0, 0), sigmaX=0.55, sigmaY=0.35)
            reflected = np.clip(reflected * 1.03, 0, 255)
        elif material == "asphalt":
            reflected = np.clip(reflected * 0.86, 0, 255)
        return reflected

    def _compose_reflection(
        self,
        original: np.ndarray,
        reflected: np.ndarray,
        alpha: np.ndarray,
        material: str,
    ) -> np.ndarray:
        alpha_3d = np.clip(alpha, 0.0, 1.0)[..., None]
        base = original * (1.0 - alpha_3d) + reflected * alpha_3d
        if material not in {"glass", "mirror", "water"}:
            return np.clip(base, 0, 255)

        reflected_luma = (
            0.299 * reflected[..., 0] + 0.587 * reflected[..., 1] + 0.114 * reflected[..., 2]
        ).astype(np.float32)
        laplacian = normalize_map(np.abs(cv2.Laplacian(reflected_luma, cv2.CV_32F, ksize=3)))
        bright = np.clip((reflected_luma - 58.0) / 172.0, 0.0, 1.0)
        fresnel_like = np.clip(0.58 * bright + 0.42 * laplacian, 0.0, 1.0)
        screen = 255.0 - ((255.0 - original) * (255.0 - reflected) / 255.0)
        screen_strength = 0.14 if material == "glass" else 0.20 if material == "mirror" else 0.10
        screen_alpha = np.clip(alpha * fresnel_like * screen_strength, 0.0, 0.24)[..., None]
        return np.clip(base * (1.0 - screen_alpha) + screen * screen_alpha, 0, 255)

    def _fallback_scene_reflection(
        self,
        image_rgb: np.ndarray,
        plane_y: int,
        material: str,
        profile: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image_rgb.shape[:2]
        target_h = height - plane_y
        source_start = max(0, plane_y - target_h)
        source = image_rgb[source_start:plane_y, :, :]
        if source.shape[0] <= 3:
            return np.zeros_like(image_rgb, dtype=np.float32), np.zeros((height, width), dtype=np.float32)
        reflected = cv2.resize(source, (width, target_h), interpolation=cv2.INTER_LINEAR)
        reflected = np.flipud(reflected).astype(np.float32)
        reflected = self._material_warp(reflected, material, float(profile["ripple"]))
        alpha = np.linspace(0.46, float(profile.get("fade_end", 0.12)), target_h, dtype=np.float32)[:, None]
        alpha = np.broadcast_to(alpha, (target_h, width)).astype(np.float32)
        layer = np.zeros_like(image_rgb, dtype=np.float32)
        full_alpha = np.zeros((height, width), dtype=np.float32)
        layer[plane_y:, :, :] = reflected
        full_alpha[plane_y:, :] = alpha
        return layer, full_alpha

    def _rough_surface_noise(self, shape: tuple[int, int], amount: float) -> np.ndarray:
        height, width = shape
        yy, xx = np.indices((height, width), dtype=np.float32)
        noise = 0.5 + 0.5 * np.sin(xx / 11.0 + yy / 19.0) * np.sin(yy / 7.0)
        return np.clip(1.0 - amount * noise, 0.0, 1.0)

    def _material_warp(
        self,
        reflected: np.ndarray,
        material: str,
        ripple: float,
    ) -> np.ndarray:
        if ripple <= 0.3:
            return reflected
        height, width = reflected.shape[:2]
        y, x = np.indices((height, width), dtype=np.float32)
        if material == "water":
            x_offset = np.sin(y / 8.0) * ripple + np.sin(y / 21.0 + x / 37.0) * ripple * 0.55
            y_offset = np.sin(x / 29.0) * ripple * 0.28
        elif material == "asphalt":
            x_offset = np.sin(y / 17.0) * ripple * 0.35
            y_offset = np.sin(x / 23.0) * ripple * 0.20
        else:
            x_offset = np.sin(y / 30.0) * ripple * 0.18
            y_offset = np.zeros_like(x)
        map_x = np.clip(x + x_offset, 0, width - 1).astype(np.float32)
        map_y = np.clip(y + y_offset, 0, height - 1).astype(np.float32)
        return cv2.remap(reflected, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _surface_sheen(
        self,
        analysis: AnalysisMaps,
        mask: np.ndarray,
        plane_y: int,
        material: str,
        sheen_strength: float,
    ) -> np.ndarray:
        highlight = np.clip((analysis.specular * 185.0 + analysis.reflection_mask * 70.0) * mask, 0.0, 255.0)
        highlight = highlight[plane_y:, :]
        sigma_x = 22.0 if material in {"water", "asphalt"} else 8.0
        sigma_y = 2.2 if material in {"water", "asphalt"} else 5.0
        sheen = cv2.GaussianBlur(highlight, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
        return np.clip(sheen * sheen_strength, 0, 36)

    def _preserve_surface_detail(
        self,
        original: np.ndarray,
        target: np.ndarray,
        detail_strength: float,
    ) -> np.ndarray:
        if detail_strength <= 0.02:
            return target
        base = cv2.GaussianBlur(original, (0, 0), sigmaX=2.0)
        detail = original - base
        return np.clip(target + detail * detail_strength, 0, 255)

    def _maybe_inpaint_reflection(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        material: str,
        prompt: PromptParameters,
        analysis: AnalysisMaps,
    ) -> np.ndarray:
        if prompt.reflection_strength < 0.68 or float(np.mean(analysis.reflection_problem)) < 0.018:
            return image_rgb
        inpaint_prompt = (
            f"natural {material} reflection, physically plausible highlights, preserve object edges, "
            "no blur artifacts, photorealistic"
        )
        result = get_ml_services().inpainting.inpaint(image_rgb, mask, inpaint_prompt)
        if result.status != "ok" or result.value is None:
            return image_rgb
        strength = np.clip(mask * 0.88, 0.0, 0.88)
        return blend_by_mask(image_rgb, result.value, strength)


class ShadowProcessor:
    def apply(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        prompt: PromptParameters,
    ) -> np.ndarray:
        mask = smart_masks.shadow_mask(image_rgb, analysis)
        denoised = cv2.bilateralFilter(image_rgb, d=9, sigmaColor=58, sigmaSpace=58).astype(np.float32)
        original = image_rgb.astype(np.float32)

        if float(np.mean(mask)) >= 0.001:
            if prompt.shadow_goal == "dramatic":
                corrected = self._dramatic_shadow(original, denoised, prompt.contrast_boost, prompt.denoise)
                strength = np.clip(mask * (0.56 + prompt.intensity * 0.14), 0.0, 0.72)
            elif prompt.shadow_goal == "soft":
                corrected = self._soft_shadow(original, denoised, prompt.softness)
                strength = np.clip(mask * (0.66 + prompt.softness * 0.18), 0.0, 0.82)
            else:
                corrected = self._clean_shadow(original, denoised, prompt.softness, analysis)
                strength = np.clip(mask * (0.80 + prompt.softness * 0.14), 0.0, 0.92)
            result = blend_by_mask(image_rgb, corrected, strength).astype(np.float32)
        else:
            result = original.copy()

        cast_shadow_confidence = float(np.mean(np.maximum(analysis.cast_shadow_problem, analysis.shadow_noise)))
        missing_shadow_confidence = float(np.mean(analysis.shadow_mask)) < 0.010 and float(np.mean(analysis.gradient)) > 0.030
        needs_generated_shadow = cast_shadow_confidence > 0.010 or (prompt.shadow_generate and missing_shadow_confidence)
        if needs_generated_shadow:
            cast_shadow = self._generate_cast_shadow(image_rgb, analysis, prompt)
            if float(np.mean(cast_shadow)) > 0.001:
                opacity = 0.22 + 0.20 * prompt.intensity
                if prompt.shadow_goal == "dramatic":
                    opacity += 0.14
                result *= (1.0 - cast_shadow[..., None] * opacity)

        if prompt.shadow_generate and float(np.mean(mask)) >= 0.001:
            result = self._maybe_inpaint_shadow(np.clip(result, 0, 255).astype(np.uint8), mask, prompt)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _clean_shadow(
        self,
        original: np.ndarray,
        denoised: np.ndarray,
        softness: float,
        analysis: AnalysisMaps,
    ) -> np.ndarray:
        lit_reference = self._estimate_lit_reference(original, analysis)
        lift_ratio = 0.42 + 0.18 * softness
        lifted = original * (1.0 - lift_ratio) + lit_reference * lift_ratio
        return denoised * 0.42 + lifted * 0.58

    def _soft_shadow(
        self,
        original: np.ndarray,
        denoised: np.ndarray,
        softness: float,
    ) -> np.ndarray:
        blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=2.3 + softness * 2.4)
        lifted = original * 0.94 + 255.0 * 0.045
        return blurred * 0.58 + lifted * 0.42

    def _estimate_lit_reference(
        self,
        original: np.ndarray,
        analysis: AnalysisMaps,
    ) -> np.ndarray:
        height, width = analysis.luminance.shape
        kernel_size = max(25, int(min(height, width) * 0.055))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        channels = []
        for channel_index in range(3):
            channel = np.clip(original[..., channel_index], 0, 255).astype(np.uint8)
            closed = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel).astype(np.float32)
            closed = cv2.GaussianBlur(closed, (0, 0), sigmaX=kernel_size * 0.20)
            channels.append(closed)
        reference = np.stack(channels, axis=-1)
        return np.clip(reference, 0, 255)

    def _maybe_inpaint_shadow(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        prompt: PromptParameters,
    ) -> np.ndarray:
        inpaint_prompt = (
            f"{prompt.shadow_goal} realistic cast shadow, clean shadow boundaries, natural depth, "
            "preserve wall texture and object details, photorealistic lighting"
        )
        result = get_ml_services().inpainting.inpaint(image_rgb, mask, inpaint_prompt)
        if result.status != "ok" or result.value is None:
            return image_rgb
        strength = np.clip(mask * 0.90, 0.0, 0.90)
        return blend_by_mask(image_rgb, result.value, strength)

    def _dramatic_shadow(
        self,
        original: np.ndarray,
        denoised: np.ndarray,
        contrast_boost: float,
        denoise: bool,
    ) -> np.ndarray:
        base = denoised if denoise else (denoised * 0.7 + original * 0.3)
        mean = np.mean(base, axis=2, keepdims=True)
        contrasted = mean + (base - mean) * max(contrast_boost, 1.28)
        return np.clip(contrasted * 0.68, 0, 255)

    def _generate_cast_shadow(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
        prompt: PromptParameters,
    ) -> np.ndarray:
        subject = self._estimate_subject_mask(image_rgb, analysis)
        if float(np.mean(subject)) < 0.006:
            return np.zeros_like(analysis.luminance, dtype=np.float32)

        height, width = subject.shape
        direction = self._estimate_shadow_direction(analysis, prompt)
        subject_depth = float(np.sum(analysis.depth * subject) / (np.sum(subject) + 1e-5))
        depth_scale = float(np.clip(0.55 + subject_depth * 0.85, 0.65, 1.35))
        x_shift = int(width * (0.06 + 0.055 * subject_depth) * (1 if direction == "right" else -1 if direction == "left" else 0.45))
        y_shift = int(height * (0.055 + 0.080 * subject_depth))
        shear = (0.055 + 0.065 * subject_depth) * (1 if direction == "right" else -1 if direction == "left" else 0.4)

        matrix = np.array([[1.0, shear, x_shift], [0.0, 0.48 + 0.15 * depth_scale, y_shift + height * (0.30 + 0.08 * subject_depth)]], dtype=np.float32)
        shadow = cv2.warpAffine(subject, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        depth_weight = cv2.GaussianBlur(analysis.depth, (0, 0), sigmaX=12.0)
        lower_weight = np.linspace(0.10, 1.0, height, dtype=np.float32)[:, None]
        shadow = np.clip(shadow * (0.45 + depth_weight * 0.75) * lower_weight, 0.0, 1.0)
        sigma = 5.5 + subject_depth * 4.0 if prompt.shadow_goal == "dramatic" else 11.0 + subject_depth * 5.0 if prompt.shadow_goal == "soft" else 8.5 + subject_depth * 4.0
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=sigma, sigmaY=sigma * 0.65)
        shadow *= (1.0 - analysis.edges * 0.35)
        return self._limit_cast_shadow(shadow, max_coverage=0.13)

    def _limit_cast_shadow(
        self,
        shadow: np.ndarray,
        max_coverage: float,
    ) -> np.ndarray:
        active = shadow > 0.025
        active_count = int(np.count_nonzero(active))
        if active_count == 0:
            return np.zeros_like(shadow, dtype=np.float32)
        max_pixels = max(1, int(shadow.size * max_coverage))
        if active_count <= max_pixels:
            return np.clip(shadow, 0.0, 1.0)
        values = shadow[active]
        threshold = float(np.partition(values, -max_pixels)[-max_pixels])
        limited = np.where(shadow >= threshold, shadow, 0.0)
        return np.clip(limited, 0.0, 1.0)

    def _estimate_subject_mask(
        self,
        image_rgb: np.ndarray,
        analysis: AnalysisMaps,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        saliency = np.clip(
            0.38 * analysis.gradient
            + 0.30 * analysis.texture
            + 0.18 * normalize_map(hsv[..., 1])
            + 0.10 * (1.0 - analysis.smooth_background)
            + 0.04 * analysis.depth,
            0.0,
            1.0,
        )
        threshold = max(0.20, float(np.percentile(saliency, 72)))
        mask = (saliency >= threshold).astype(np.uint8)
        height, width = mask.shape
        center_bias = np.zeros_like(mask, dtype=np.uint8)
        center_bias[int(height * 0.12) : int(height * 0.88), int(width * 0.12) : int(width * 0.88)] = 1
        mask *= center_bias
        kernel = np.ones((7, 7), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if component_count <= 1:
            return mask.astype(np.float32)

        image_area = height * width
        cx = width / 2.0
        cy = height / 2.0
        best_index = 0
        best_score = -1.0
        for index in range(1, component_count):
            area = stats[index, cv2.CC_STAT_AREA]
            if area < image_area * 0.006:
                continue
            dist = np.hypot((centroids[index][0] - cx) / width, (centroids[index][1] - cy) / height)
            score = float(area) / image_area - float(dist) * 0.18
            if score > best_score:
                best_score = score
                best_index = index

        if best_index == 0:
            return np.zeros_like(mask, dtype=np.float32)
        subject = (labels == best_index).astype(np.float32)
        return feather_mask(subject, sigma=2.0)

    def _estimate_shadow_direction(
        self,
        analysis: AnalysisMaps,
        prompt: PromptParameters,
    ) -> str:
        if prompt.light_direction in {"left", "right"}:
            return "right" if prompt.light_direction == "left" else "left"
        height, width = analysis.luminance.shape
        bright = analysis.specular + analysis.overexposure
        if float(np.mean(bright)) < 0.005:
            return "right"
        y, x = np.indices((height, width), dtype=np.float32)
        weight = bright + 1e-4
        light_x = float(np.sum(x * weight) / np.sum(weight))
        return "right" if light_x < width / 2 else "left"
