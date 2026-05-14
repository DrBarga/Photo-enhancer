import cv2
import numpy as np

from app.models.schemas import AnalysisMaps
from app.services.ml_providers import get_ml_services
from app.services.universal_analyzer import UniversalImageAnalyzer
from app.utils.map_math import local_mean_std, normalize_map


class ImageAnalyzer:
    def __init__(self) -> None:
        self.universal_analyzer = UniversalImageAnalyzer()

    def analyze(self, image_rgb: np.ndarray) -> AnalysisMaps:
        image_float = image_rgb.astype(np.float32)
        luminance = (
            0.299 * image_float[..., 0]
            + 0.587 * image_float[..., 1]
            + 0.114 * image_float[..., 2]
        ).astype(np.float32)

        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        saturation = hsv[..., 1] / 255.0

        gradient = self._sobel_gradient(luminance)
        gradient_norm = normalize_map(gradient)
        local_mean_15, local_std_15 = local_mean_std(luminance, 15)
        local_mean_45, local_std_45 = local_mean_std(luminance, 45)
        contrast_norm = normalize_map(local_std_15)
        texture = self._texture_map(luminance, lab, saturation)
        edges = self._edges(luminance)
        edge_norm = edges.astype(np.float32) / 255.0

        banding = self._banding_map(luminance, gradient_norm, contrast_norm, texture)
        overexposure = np.clip((luminance - 235.0) / 18.0, 0.0, 1.0)
        specular = self._specular_map(luminance, saturation, edge_norm)
        smooth_background = self._smooth_background_map(contrast_norm, texture, edge_norm)
        cast_shadow_signal = self._cast_shadow_signal(
            luminance,
            saturation,
            contrast_norm,
            texture,
            edge_norm,
            smooth_background,
        )
        shadow_mask = self._shadow_mask(luminance, local_mean_45, local_std_45, cast_shadow_signal)
        shadow_noise = self._shadow_noise_map(luminance, shadow_mask, local_std_15, texture)
        cv_depth = self._estimate_depth_map(
            luminance,
            gradient_norm,
            contrast_norm,
            texture,
            smooth_background,
            shadow_mask,
            reflection_mask=None,
        )
        depth_result = get_ml_services().depth.estimate(image_rgb, cv_depth)
        depth = depth_result.value
        cast_shadow_problem = self._cast_shadow_problem(
            cast_shadow_signal,
            shadow_mask,
            edge_norm,
            texture,
        )
        reflection_mask, reflection_problem = self._reflection_maps(
            luminance,
            saturation,
            edge_norm,
            contrast_norm,
            texture,
            specular,
            shadow_noise,
        )

        laplacian = normalize_map(np.abs(cv2.Laplacian(luminance, cv2.CV_32F, ksize=3)))
        gradient_discontinuity = self._gradient_discontinuity_map(
            lab,
            luminance,
            gradient_norm,
            contrast_norm,
            texture,
            edge_norm,
            smooth_background,
        )
        base_gradient_problem = np.clip(
            (0.42 * banding + 0.34 * laplacian * smooth_background + 0.24 * overexposure)
            * (0.55 + 0.45 * smooth_background),
            0.0,
            1.0,
        )
        gradient_problem = np.clip(
            np.maximum(base_gradient_problem, gradient_discontinuity),
            0.0,
            1.0,
        )

        problem_map = np.clip(
            0.26 * gradient_problem
            + 0.20 * banding
            + 0.18 * shadow_noise
            + 0.13 * cast_shadow_problem
            + 0.13 * reflection_problem
            + 0.10 * overexposure,
            0.0,
            1.0,
        )

        reflection_material = self._classify_reflection_material(
            luminance,
            saturation,
            texture,
            specular,
            reflection_mask,
        )

        analysis_maps = AnalysisMaps(
            luminance=luminance,
            gradient=gradient_norm,
            contrast=contrast_norm,
            edges=edge_norm,
            gradient_problem=gradient_problem,
            banding=banding,
            overexposure=overexposure,
            shadow_noise=shadow_noise,
            reflection_problem=reflection_problem,
            texture=texture,
            specular=specular,
            smooth_background=smooth_background,
            depth=depth,
            cast_shadow_problem=cast_shadow_problem,
            shadow_mask=shadow_mask,
            reflection_mask=reflection_mask,
            problem_map=problem_map,
            edge_density=float(np.mean(edge_norm)),
            scene_type=self._classify_scene(
                gradient_problem,
                reflection_problem,
                shadow_noise,
                cast_shadow_problem,
            ),
            reflection_material=reflection_material,
            ml_status={
                "depth": depth_result.provider,
                "depth_status": depth_result.status,
                "depth_detail": depth_result.detail[:160],
            },
        )
        universal, semantic_masks = self.universal_analyzer.analyze(image_rgb, analysis_maps)
        analysis_maps.universal = universal
        analysis_maps.semantic_masks = semantic_masks
        return analysis_maps

    def _sobel_gradient(self, luminance: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    def _edges(self, luminance: np.ndarray) -> np.ndarray:
        luminance_u8 = np.clip(luminance, 0, 255).astype(np.uint8)
        return cv2.Canny(luminance_u8, threshold1=55, threshold2=145)

    def _texture_map(
        self,
        luminance: np.ndarray,
        lab: np.ndarray,
        saturation: np.ndarray,
    ) -> np.ndarray:
        laplacian = np.abs(cv2.Laplacian(luminance, cv2.CV_32F, ksize=3))
        chroma = cv2.magnitude(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
        _, local_std = local_mean_std(luminance, 9)
        texture = 0.46 * normalize_map(laplacian) + 0.34 * normalize_map(local_std)
        texture += 0.20 * normalize_map(chroma * (0.25 + saturation))
        return np.clip(texture, 0.0, 1.0)

    def _smooth_background_map(
        self,
        contrast_norm: np.ndarray,
        texture: np.ndarray,
        edge_norm: np.ndarray,
    ) -> np.ndarray:
        smooth = 1.0 - 0.52 * contrast_norm - 0.35 * texture - 0.34 * edge_norm
        smooth = cv2.GaussianBlur(np.clip(smooth, 0.0, 1.0), (0, 0), sigmaX=5.0)
        return np.clip(smooth, 0.0, 1.0)

    def _banding_map(
        self,
        luminance: np.ndarray,
        gradient_norm: np.ndarray,
        contrast_norm: np.ndarray,
        texture: np.ndarray,
    ) -> np.ndarray:
        diff_x = np.abs(np.diff(luminance, axis=1, prepend=luminance[:, :1]))
        diff_y = np.abs(np.diff(luminance, axis=0, prepend=luminance[:1, :]))
        quantized_steps = ((diff_x > 1.5) & (diff_x < 10.0)).astype(np.float32)
        quantized_steps += ((diff_y > 1.5) & (diff_y < 10.0)).astype(np.float32)
        quantized_steps = np.clip(quantized_steps / 2.0, 0.0, 1.0)

        row_pattern = normalize_map(cv2.blur(quantized_steps, (1, 21)))
        column_pattern = normalize_map(cv2.blur(quantized_steps, (21, 1)))
        smooth_zone = np.clip(1.0 - 1.35 * contrast_norm - 0.70 * texture - 0.35 * gradient_norm, 0.0, 1.0)
        return np.clip(np.maximum(row_pattern, column_pattern) * smooth_zone, 0.0, 1.0)

    def _gradient_discontinuity_map(
        self,
        lab: np.ndarray,
        luminance: np.ndarray,
        gradient_norm: np.ndarray,
        contrast_norm: np.ndarray,
        texture: np.ndarray,
        edge_norm: np.ndarray,
        smooth_background: np.ndarray,
    ) -> np.ndarray:
        height, width = luminance.shape
        lab_float = lab.astype(np.float32)
        small_sigma = 1.25
        large_sigma = float(np.clip(min(height, width) * 0.052, 12.0, 42.0))

        lab_small = cv2.GaussianBlur(lab_float, (0, 0), sigmaX=small_sigma, sigmaY=small_sigma)
        lab_large = cv2.GaussianBlur(lab_float, (0, 0), sigmaX=large_sigma, sigmaY=large_sigma)
        residual = np.linalg.norm(lab_small - lab_large, axis=2)

        lap_l = np.abs(cv2.Laplacian(lab_small[..., 0], cv2.CV_32F, ksize=3))
        lap_a = np.abs(cv2.Laplacian(lab_small[..., 1], cv2.CV_32F, ksize=3))
        lap_b = np.abs(cv2.Laplacian(lab_small[..., 2], cv2.CV_32F, ksize=3))
        curvature = lap_l + 0.55 * lap_a + 0.55 * lap_b

        diff_x = np.linalg.norm(np.diff(lab_small, axis=1, prepend=lab_small[:, :1]), axis=2)
        diff_y = np.linalg.norm(np.diff(lab_small, axis=0, prepend=lab_small[:1, :]), axis=2)
        directional_jump = np.maximum(diff_x, diff_y)

        edge_block = cv2.dilate(
            (edge_norm > 0.035).astype(np.uint8),
            np.ones((11, 11), dtype=np.uint8),
            iterations=1,
        ).astype(np.float32)
        edge_block = cv2.GaussianBlur(edge_block, (0, 0), sigmaX=3.2, sigmaY=3.2)
        detail_block = np.clip(0.58 * texture + 0.26 * contrast_norm + 0.16 * gradient_norm, 0.0, 1.0)
        support = smooth_background * (1.0 - 0.88 * edge_block) * (1.0 - 0.58 * detail_block)
        support = cv2.GaussianBlur(np.clip(support, 0.0, 1.0), (0, 0), sigmaX=3.0, sigmaY=3.0)

        residual_score = self._robust_normalize(residual, support, percentile=96.5)
        curvature_score = self._robust_normalize(curvature, support, percentile=97.2)
        jump_score = self._robust_normalize(directional_jump, support, percentile=96.0)
        abrupt_change = 0.46 * residual_score + 0.34 * curvature_score + 0.20 * jump_score
        abrupt_change = np.clip((abrupt_change - 0.12) / 0.62, 0.0, 1.0)

        broad_harsh_gradient = self._directional_gradient_ridge(lab_small, support)
        problem = np.maximum(abrupt_change, broad_harsh_gradient)
        problem = cv2.GaussianBlur(problem * support, (0, 0), sigmaX=2.2, sigmaY=2.2)
        return np.clip(problem, 0.0, 1.0)

    def _directional_gradient_ridge(self, lab_small: np.ndarray, support: np.ndarray) -> np.ndarray:
        color_luma = lab_small[..., 0]
        chroma_a = lab_small[..., 1]
        chroma_b = lab_small[..., 2]
        gx_l = np.abs(cv2.Sobel(color_luma, cv2.CV_32F, 1, 0, ksize=5))
        gy_l = np.abs(cv2.Sobel(color_luma, cv2.CV_32F, 0, 1, ksize=5))
        gx_a = np.abs(cv2.Sobel(chroma_a, cv2.CV_32F, 1, 0, ksize=5))
        gy_a = np.abs(cv2.Sobel(chroma_a, cv2.CV_32F, 0, 1, ksize=5))
        gx_b = np.abs(cv2.Sobel(chroma_b, cv2.CV_32F, 1, 0, ksize=5))
        gy_b = np.abs(cv2.Sobel(chroma_b, cv2.CV_32F, 0, 1, ksize=5))
        horizontal_pressure = gx_l + 0.42 * gx_a + 0.42 * gx_b
        vertical_pressure = gy_l + 0.42 * gy_a + 0.42 * gy_b
        directional_pressure = np.maximum(horizontal_pressure, vertical_pressure)
        ridge = np.abs(cv2.Laplacian(directional_pressure, cv2.CV_32F, ksize=3))
        pressure_score = self._robust_normalize(directional_pressure, support, percentile=96.0)
        ridge_score = self._robust_normalize(ridge, support, percentile=97.0)
        score = pressure_score * ridge_score
        return np.clip((score - 0.16) / 0.54, 0.0, 1.0)

    def _robust_normalize(
        self,
        values: np.ndarray,
        support: np.ndarray | None = None,
        percentile: float = 97.0,
    ) -> np.ndarray:
        values = values.astype(np.float32)
        if support is not None and int(np.count_nonzero(support > 0.08)) > 32:
            sample = values[support > 0.08]
        else:
            sample = values.reshape(-1)
        low = float(np.percentile(sample, 8))
        high = float(np.percentile(sample, percentile))
        if high - low < 1e-5:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values - low) / (high - low), 0.0, 1.0)

    def _specular_map(
        self,
        luminance: np.ndarray,
        saturation: np.ndarray,
        edge_norm: np.ndarray,
    ) -> np.ndarray:
        bright = np.clip((luminance - 205.0) / 45.0, 0.0, 1.0)
        low_saturation = np.clip(1.0 - saturation * 1.4, 0.0, 1.0)
        return np.clip(bright * (0.58 + 0.32 * low_saturation + 0.10 * edge_norm), 0.0, 1.0)

    def _shadow_mask(
        self,
        luminance: np.ndarray,
        local_mean: np.ndarray,
        local_std: np.ndarray,
        cast_shadow_signal: np.ndarray,
    ) -> np.ndarray:
        relative_dark = luminance < (local_mean - 0.48 * local_std - 7.0)
        absolute_dark = luminance < max(116.0, float(np.percentile(luminance, 38)))
        cast_shadow = cast_shadow_signal > 0.16
        mask = ((relative_dark & absolute_dark) | cast_shadow).astype(np.float32)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask.astype(np.float32)

    def _cast_shadow_signal(
        self,
        luminance: np.ndarray,
        saturation: np.ndarray,
        contrast_norm: np.ndarray,
        texture: np.ndarray,
        edge_norm: np.ndarray,
        smooth_background: np.ndarray,
    ) -> np.ndarray:
        height, width = luminance.shape
        kernel_size = max(25, int(min(height, width) * 0.055))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(np.clip(luminance, 0, 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        background = cv2.GaussianBlur(background.astype(np.float32), (0, 0), sigmaX=kernel_size * 0.22)

        deficit = np.clip((background - luminance) / np.maximum(background, 48.0), 0.0, 1.0)
        soft_deficit = cv2.GaussianBlur(deficit, (0, 0), sigmaX=1.8)
        illumination_drop = np.clip((soft_deficit - 0.045) / 0.24, 0.0, 1.0)

        surface_support = np.clip(
            0.58 * smooth_background
            + 0.20 * (1.0 - texture)
            + 0.14 * (1.0 - saturation)
            + 0.08 * (1.0 - contrast_norm),
            0.0,
            1.0,
        )
        not_hard_edge = np.clip(1.0 - edge_norm * 0.55, 0.0, 1.0)
        cast_shadow = illumination_drop * (0.35 + 0.65 * surface_support) * not_hard_edge
        return np.clip(cast_shadow, 0.0, 1.0)

    def _shadow_noise_map(
        self,
        luminance: np.ndarray,
        shadow_mask: np.ndarray,
        local_std: np.ndarray,
        texture: np.ndarray,
    ) -> np.ndarray:
        darkness = normalize_map(255.0 - luminance)
        dirty_texture = np.clip(0.62 * normalize_map(local_std) + 0.38 * texture, 0.0, 1.0)
        return np.clip(shadow_mask * darkness * dirty_texture, 0.0, 1.0)

    def _estimate_depth_map(
        self,
        luminance: np.ndarray,
        gradient_norm: np.ndarray,
        contrast_norm: np.ndarray,
        texture: np.ndarray,
        smooth_background: np.ndarray,
        shadow_mask: np.ndarray,
        reflection_mask: np.ndarray | None,
    ) -> np.ndarray:
        height, width = luminance.shape
        vertical_depth = np.linspace(0.08, 1.0, height, dtype=np.float32)[:, None]
        vertical_depth = np.broadcast_to(vertical_depth, (height, width))

        detail_depth = np.clip(0.42 * gradient_norm + 0.34 * contrast_norm + 0.24 * texture, 0.0, 1.0)
        atmospheric_depth = np.clip(1.0 - smooth_background * 0.55 - normalize_map(luminance) * 0.18, 0.0, 1.0)
        shadow_anchor = cv2.GaussianBlur(shadow_mask.astype(np.float32), (0, 0), sigmaX=9.0)
        reflection_penalty = 0.0 if reflection_mask is None else reflection_mask * 0.20

        depth = (
            0.46 * vertical_depth
            + 0.26 * detail_depth
            + 0.18 * atmospheric_depth
            + 0.10 * shadow_anchor
            - reflection_penalty
        )
        depth = cv2.GaussianBlur(np.clip(depth, 0.0, 1.0), (0, 0), sigmaX=4.0)
        return np.clip(depth, 0.0, 1.0).astype(np.float32)

    def _cast_shadow_problem(
        self,
        cast_shadow_signal: np.ndarray,
        shadow_mask: np.ndarray,
        edge_norm: np.ndarray,
        texture: np.ndarray,
    ) -> np.ndarray:
        soft_shadow = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=6.0)
        dirty_shadow = np.clip(edge_norm * soft_shadow * 0.45 + texture * shadow_mask * 0.28, 0.0, 1.0)
        return np.clip(0.76 * cast_shadow_signal + 0.24 * dirty_shadow, 0.0, 1.0)

    def _reflection_maps(
        self,
        luminance: np.ndarray,
        saturation: np.ndarray,
        edge_norm: np.ndarray,
        contrast_norm: np.ndarray,
        texture: np.ndarray,
        specular: np.ndarray,
        shadow_noise: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = luminance.shape
        start_y = int(height * 0.48)
        lower_height = height - start_y
        reflection_mask = np.zeros_like(luminance, dtype=np.float32)
        reflection_problem = np.zeros_like(luminance, dtype=np.float32)
        if lower_height < 8:
            return reflection_mask, reflection_problem

        lower = luminance[start_y:, :]
        upper_start = max(0, start_y - lower_height)
        mirrored_source = luminance[upper_start:start_y, :]
        if mirrored_source.shape[0] != lower.shape[0]:
            mirrored_source = cv2.resize(mirrored_source, (width, lower_height), interpolation=cv2.INTER_LINEAR)
        mirrored = np.flipud(mirrored_source)

        symmetry = 1.0 - normalize_map(np.abs(lower - mirrored))
        vertical_weight = np.linspace(0.20, 1.0, lower_height, dtype=np.float32)[:, None]
        lower_edges = edge_norm[start_y:, :]
        lower_contrast = contrast_norm[start_y:, :]
        lower_texture = texture[start_y:, :]
        lower_specular = specular[start_y:, :]
        lower_saturation = saturation[start_y:, :]
        lower_noise = shadow_noise[start_y:, :]
        smooth_surface = np.clip(
            0.42 * lower_specular
            + 0.24 * (1.0 - lower_texture)
            + 0.18 * (1.0 - lower_saturation * 0.72)
            + 0.16 * (1.0 - lower_contrast),
            0.0,
            1.0,
        )

        surface_likelihood = np.clip(
            0.30 * symmetry
            + 0.24 * smooth_surface
            + 0.18 * lower_specular
            + 0.12 * lower_edges
            + 0.10 * lower_contrast
            + 0.06 * (1.0 - lower_saturation * 0.65),
            0.0,
            1.0,
        )
        probability = np.clip(surface_likelihood * vertical_weight, 0.0, 1.0)
        missing_or_weak_reflection = np.clip(
            smooth_surface * (1.0 - symmetry) * (0.46 + 0.34 * lower_specular + 0.20 * vertical_weight),
            0.0,
            1.0,
        )
        problem = np.clip(
            probability * (
                0.24 * lower_edges
                + 0.20 * lower_contrast
                + 0.17 * lower_texture
                + 0.13 * lower_noise
                + 0.26 * missing_or_weak_reflection
            ),
            0.0,
            1.0,
        )

        reflection_mask[start_y:, :] = probability
        reflection_problem[start_y:, :] = problem
        return reflection_mask, reflection_problem

    def _classify_reflection_material(
        self,
        luminance: np.ndarray,
        saturation: np.ndarray,
        texture: np.ndarray,
        specular: np.ndarray,
        reflection_mask: np.ndarray,
    ) -> str:
        lower = reflection_mask > max(0.16, float(np.percentile(reflection_mask, 72)))
        if int(np.count_nonzero(lower)) < 30:
            lower = np.zeros_like(reflection_mask, dtype=bool)
            lower[int(luminance.shape[0] * 0.55) :, :] = True

        mean_luma = float(np.mean(luminance[lower]))
        mean_sat = float(np.mean(saturation[lower]))
        mean_texture = float(np.mean(texture[lower]))
        mean_specular = float(np.mean(specular[lower]))

        if mean_specular > 0.22 and mean_texture < 0.22 and mean_luma > 135:
            return "mirror"
        if mean_specular > 0.12 and mean_texture < 0.30 and mean_sat < 0.45:
            return "glass"
        if mean_texture > 0.36 and mean_luma < 115:
            return "asphalt"
        if mean_texture > 0.24 and mean_specular > 0.08:
            return "water"
        return "glass"

    def _classify_scene(
        self,
        gradient_problem: np.ndarray,
        reflection_problem: np.ndarray,
        shadow_noise: np.ndarray,
        cast_shadow_problem: np.ndarray,
    ) -> str:
        scores = {
            "gradient_background": float(np.mean(gradient_problem)),
            "reflective_surface": float(np.mean(reflection_problem)),
            "shadow_structure": float(np.mean(shadow_noise) + np.mean(cast_shadow_problem) * 0.65),
        }
        scene_type, score = max(scores.items(), key=lambda item: item[1])
        if score < 0.030:
            return "balanced_lighting"
        return scene_type
