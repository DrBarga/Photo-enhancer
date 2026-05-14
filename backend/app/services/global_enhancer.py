import cv2
import numpy as np

from app.models.schemas import AnalysisMaps
from app.utils.map_math import feather_mask


class GlobalEnhancer:
    def apply(self, image_rgb: np.ndarray, analysis: AnalysisMaps, actions: list[str]) -> np.ndarray:
        if not actions:
            return image_rgb
        current = image_rgb.astype(np.float32)
        protect = self._protection_mask(analysis)

        if "white_balance" in actions:
            current = self._gray_world_white_balance(current)
        if "exposure_lift" in actions or "highlight_recovery" in actions:
            current = self._smart_exposure(current, lift="exposure_lift" in actions, recover="highlight_recovery" in actions)
        if "smart_contrast" in actions:
            current = self._clahe_contrast(current)
        if "dehaze" in actions:
            current = self._dehaze(current)
        if "denoise" in actions:
            current = self._denoise(current)
        if "jpeg_cleanup" in actions:
            current = self._jpeg_cleanup(current)
        if "sharpen" in actions:
            current = self._unsharp(current)

        protection = np.clip(protect[..., None], 0.0, 1.0)
        protected = image_rgb.astype(np.float32) * protection + current * (1.0 - protection)
        return np.clip(protected, 0, 255).astype(np.uint8)

    def _protection_mask(self, analysis: AnalysisMaps) -> np.ndarray:
        masks = analysis.semantic_masks or {}
        face = masks.get("face")
        text = masks.get("text")
        person = masks.get("person")
        protect = np.zeros_like(analysis.luminance, dtype=np.float32)
        for mask, weight in ((face, 1.0), (text, 0.95), (person, 0.35)):
            if mask is not None:
                protect = np.maximum(protect, np.clip(mask.astype(np.float32) * weight, 0.0, 1.0))
        return feather_mask(protect, sigma=2.0)

    def _gray_world_white_balance(self, image: np.ndarray) -> np.ndarray:
        means = np.maximum(image.mean(axis=(0, 1)), 1.0)
        target = float(np.mean(means))
        gain = np.clip(target / means, 0.72, 1.32)
        return np.clip(image * gain, 0, 255)

    def _smart_exposure(self, image: np.ndarray, lift: bool, recover: bool) -> np.ndarray:
        lab = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        l = lab[..., 0] / 255.0
        if lift:
            l = np.power(np.clip(l, 0.0, 1.0), 0.88) + np.clip(0.22 - l, 0.0, 0.22) * 0.22
        if recover:
            l = np.where(l > 0.78, 0.78 + (l - 0.78) * 0.62, l)
        lab[..., 0] = np.clip(l * 255.0, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    def _clahe_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB).astype(np.float32)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(np.clip(image, 0, 255).astype(np.uint8), None, 5, 5, 7, 21).astype(np.float32)

    def _jpeg_cleanup(self, image: np.ndarray) -> np.ndarray:
        soft = cv2.bilateralFilter(np.clip(image, 0, 255).astype(np.uint8), d=5, sigmaColor=28, sigmaSpace=28).astype(np.float32)
        return image * 0.72 + soft * 0.28

    def _unsharp(self, image: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.15)
        return np.clip(image * 1.22 - blur * 0.22, 0, 255)

    def _dehaze(self, image: np.ndarray) -> np.ndarray:
        dark = np.min(image, axis=2)
        airlight = float(np.percentile(dark, 96))
        haze = np.clip((airlight - dark) / max(airlight, 1.0), 0.0, 1.0)
        gain = 1.0 + haze[..., None] * 0.10
        return np.clip((image - airlight * 0.025) * gain + airlight * 0.025, 0, 255)
