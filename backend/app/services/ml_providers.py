from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.models.schemas import AnalysisMaps
from app.services.ml_config import settings
from app.utils.map_math import feather_mask, normalize_map


@dataclass
class ProviderResult:
    value: Any
    provider: str
    status: str
    detail: str = ""
    confidence: float = 0.0


@dataclass
class MLRuntimeStatus:
    depth: str = "cv"
    segmentation: str = "cv"
    clip: str = "rules"
    inpainting: str = "disabled"
    classifier: str = "rules"
    details: dict[str, str] = field(default_factory=dict)


class DepthProvider:
    def __init__(self) -> None:
        self._pipeline = None
        self._load_error = ""

    def estimate(self, image_rgb: np.ndarray, fallback_depth: np.ndarray) -> ProviderResult:
        if not settings.enabled or settings.depth_provider == "cv":
            return ProviderResult(fallback_depth, "cv", "fallback", "ML depth disabled")

        try:
            pipeline = self._get_pipeline()
            pil_image = Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8))
            output = pipeline(pil_image)
            depth_image = output["depth"]
            depth = np.asarray(depth_image.resize((image_rgb.shape[1], image_rgb.shape[0])), dtype=np.float32)
            depth = normalize_map(depth)
            return ProviderResult(depth, settings.depth_model, "ok", "pretrained depth", 0.95)
        except Exception as exc:  # noqa: BLE001 - optional provider must never break product flow.
            self._load_error = str(exc)
            return ProviderResult(fallback_depth, "cv", "fallback", str(exc))

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("transformers/torch are not installed") from exc

        self._pipeline = pipeline(
            task="depth-estimation",
            model=settings.depth_model,
            device=-1,
        )
        return self._pipeline


class SegmentationProvider:
    def __init__(self) -> None:
        self._sam_generator = None
        self._load_error = ""
        self._last_key = ""
        self._last_result: ProviderResult | None = None

    def masks(self, image_rgb: np.ndarray, analysis: AnalysisMaps) -> ProviderResult:
        cache_key = self._cache_key(image_rgb)
        if self._last_key == cache_key and self._last_result is not None:
            return self._last_result

        if settings.enabled and settings.segmentation_provider == "sam" and settings.sam_checkpoint:
            try:
                masks = self._sam_masks(image_rgb)
                result = ProviderResult(masks, "sam", "ok", "SAM automatic masks", 0.92)
                self._remember(cache_key, result)
                return result
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)

        result = ProviderResult(self._cv_masks(image_rgb, analysis), "cv", "fallback", self._load_error)
        self._remember(cache_key, result)
        return result

    def _remember(self, cache_key: str, result: ProviderResult) -> None:
        self._last_key = cache_key
        self._last_result = result

    def _cache_key(self, image_rgb: np.ndarray) -> str:
        preview = cv2.resize(np.ascontiguousarray(image_rgb), (48, 48), interpolation=cv2.INTER_AREA)
        digest = hashlib.blake2b(preview.tobytes(), digest_size=12).hexdigest()
        return f"{image_rgb.shape[0]}x{image_rgb.shape[1]}:{digest}"

    def _sam_masks(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        generator = self._get_sam_generator()
        annotations = generator.generate(np.clip(image_rgb, 0, 255).astype(np.uint8))
        height, width = image_rgb.shape[:2]
        object_mask = np.zeros((height, width), dtype=np.float32)
        for annotation in annotations:
            area_ratio = float(annotation.get("area", 0)) / float(height * width)
            stability = float(annotation.get("stability_score", 0.0))
            if 0.006 <= area_ratio <= 0.55 and stability >= 0.82:
                object_mask = np.maximum(object_mask, annotation["segmentation"].astype(np.float32))
        object_mask = feather_mask(object_mask, sigma=1.6)
        surface_mask = np.clip(1.0 - object_mask, 0.0, 1.0)
        return {"object": object_mask, "surface": surface_mask}

    def _get_sam_generator(self):
        if self._sam_generator is not None:
            return self._sam_generator
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("segment-anything/torch are not installed") from exc

        model = sam_model_registry[settings.sam_model_type](checkpoint=settings.sam_checkpoint)
        self._sam_generator = SamAutomaticMaskGenerator(model)
        return self._sam_generator

    def _cv_masks(self, image_rgb: np.ndarray, analysis: AnalysisMaps) -> dict[str, np.ndarray]:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        saliency = np.clip(
            0.34 * analysis.gradient
            + 0.28 * analysis.texture
            + 0.20 * normalize_map(hsv[..., 1])
            + 0.18 * (1.0 - analysis.smooth_background),
            0.0,
            1.0,
        )
        threshold = max(0.22, float(np.percentile(saliency, 78)))
        object_mask = (saliency >= threshold).astype(np.float32)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        object_mask = feather_mask(object_mask, sigma=2.0)
        surface_mask = np.clip(analysis.smooth_background * (1.0 - object_mask * 0.55), 0.0, 1.0)
        return {"object": object_mask, "surface": surface_mask}


class ClipProvider:
    material_labels = {
        "water": "wet water reflection surface",
        "asphalt": "wet asphalt road reflective surface",
        "mirror": "sharp mirror reflection",
        "glass": "glass glossy reflective surface",
        "interior": "interior product photography",
    }

    problem_labels = {
        "gradient": "bad gradient banding posterized background",
        "shadow": "dirty unrealistic cast shadows",
        "reflection": "unnatural reflection artifact",
        "normal": "clean realistic lighting",
    }

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._load_error = ""

    def classify(self, image_rgb: np.ndarray, prompt: str) -> ProviderResult:
        if not settings.enabled or settings.clip_provider == "rules":
            return ProviderResult(self._rules(prompt), "rules", "fallback", "CLIP disabled")
        try:
            return ProviderResult(self._clip_classify(image_rgb), settings.clip_model, "ok", "CLIP zero-shot", 0.90)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            return ProviderResult(self._rules(prompt), "rules", "fallback", str(exc))

    def _clip_classify(self, image_rgb: np.ndarray) -> dict[str, Any]:
        model, processor = self._get_model()
        labels = {**self.material_labels, **self.problem_labels}
        texts = list(labels.values())
        pil_image = Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8))
        inputs = processor(text=texts, images=pil_image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits = outputs.logits_per_image.detach().cpu().numpy()[0]
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        scored = sorted(zip(labels.keys(), probs.tolist()), key=lambda item: item[1], reverse=True)
        return {
            "top": scored[0][0],
            "scores": {key: round(float(value), 4) for key, value in scored[:6]},
        }

    def _get_model(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("transformers/torch are not installed") from exc
        self._model = CLIPModel.from_pretrained(settings.clip_model)
        self._processor = CLIPProcessor.from_pretrained(settings.clip_model)
        return self._model, self._processor

    def _rules(self, prompt: str) -> dict[str, Any]:
        text = prompt.lower()
        scores = {
            "water": float(any(word in text for word in ["water", "вода", "луж"])),
            "asphalt": float(any(word in text for word in ["asphalt", "асфальт"])),
            "mirror": float(any(word in text for word in ["mirror", "зеркал"])),
            "glass": float(any(word in text for word in ["glass", "стекл"])),
            "shadow": float(any(word in text for word in ["shadow", "тень", "тени"])),
            "gradient": float(any(word in text for word in ["gradient", "градиент", "banding"])),
            "reflection": float(any(word in text for word in ["reflection", "отраж"])),
        }
        top = max(scores.items(), key=lambda item: item[1])[0] if any(scores.values()) else "normal"
        return {"top": top, "scores": scores}


class InpaintingProvider:
    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str) -> ProviderResult:
        if settings.inpaint_provider == "none":
            return ProviderResult(None, "none", "disabled", "Inpainting disabled")
        if settings.inpaint_provider == "stability":
            return self._stability_inpaint(image_rgb, mask, prompt)
        return ProviderResult(None, settings.inpaint_provider, "disabled", "Unknown provider")

    def _stability_inpaint(self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str) -> ProviderResult:
        if not settings.stability_api_key:
            return ProviderResult(None, "stability", "fallback", "STABILITY_API_KEY is missing")
        try:
            import requests
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(None, "stability", "fallback", f"requests not installed: {exc}")

        image_bytes = _png_bytes(image_rgb)
        mask_image = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
        mask_bytes = _png_bytes(mask_image)
        response = requests.post(
            settings.stability_inpaint_endpoint,
            headers={"authorization": f"Bearer {settings.stability_api_key}", "accept": "image/*"},
            files={"image": image_bytes, "mask": mask_bytes},
            data={"prompt": prompt, "output_format": "png"},
            timeout=120,
        )
        if response.status_code >= 400:
            return ProviderResult(None, "stability", "fallback", response.text[:300])
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return ProviderResult(np.asarray(image, dtype=np.uint8), "stability", "ok", "inpainting complete", 0.92)


class ProblemClassifier:
    def __init__(self) -> None:
        self._model = None
        self._load_error = ""

    def predict(self, prompt: str, analysis: AnalysisMaps) -> ProviderResult:
        if settings.enabled:
            try:
                model = self._get_model()
                features = self._feature_text(prompt, analysis)
                prediction = model.predict([features])[0]
                return ProviderResult(self._parse_prediction(prediction), "sklearn", "ok", "trained lightweight classifier", 0.85)
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
        return ProviderResult(self._rules(prompt, analysis), "rules", "fallback", self._load_error)

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            import joblib
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("joblib/scikit-learn are not installed") from exc
        self._model = joblib.load(settings.classifier_path)
        return self._model

    def _feature_text(self, prompt: str, analysis: AnalysisMaps) -> str:
        severity = analysis.severity()
        dominant_problem = max(
            {
                "gradient": severity["gradient"] + severity["banding"] * 0.7,
                "shadow": severity["shadow"] + severity["cast_shadow"] * 0.9,
                "reflection": severity["reflection"],
            }.items(),
            key=lambda item: item[1],
        )[0]
        return (
            f"{prompt} scene:{analysis.scene_type} material:{analysis.reflection_material} "
            f"problem:{dominant_problem} overall:{severity['overall']:.3f} "
            f"gradient:{severity['gradient']:.3f} shadow:{severity['shadow']:.3f} "
            f"cast_shadow:{severity['cast_shadow']:.3f} reflection:{severity['reflection']:.3f}"
        )

    def _parse_prediction(self, prediction: Any) -> dict[str, str]:
        if isinstance(prediction, dict):
            return {
                "problem": str(prediction.get("problem", "shadow")),
                "material": str(prediction.get("material", "glass")),
                "strength": str(prediction.get("strength", "medium")),
            }
        if isinstance(prediction, (list, tuple)) and len(prediction) >= 3:
            return {"problem": str(prediction[0]), "material": str(prediction[1]), "strength": str(prediction[2])}
        if hasattr(prediction, "tolist"):
            values = prediction.tolist()
            if isinstance(values, list) and len(values) >= 3:
                return {"problem": str(values[0]), "material": str(values[1]), "strength": str(values[2])}
        parts = str(prediction).split("|")
        if len(parts) == 3:
            return {"problem": parts[0], "material": parts[1], "strength": parts[2]}
        return {"problem": str(prediction), "material": "glass", "strength": "medium"}

    def _rules(self, prompt: str, analysis: AnalysisMaps) -> dict[str, Any]:
        severity = analysis.severity()
        text = prompt.lower()
        prompt_boost = {
            "gradient": 0.35
            if any(word in text for word in ["gradient", "banding", "градиент", "полос"])
            else 0.0,
            "shadow": 0.35
            if any(word in text for word in ["shadow", "shadows", "cast", "тень", "тени"])
            else 0.0,
            "reflection": 0.35
            if any(word in text for word in ["reflection", "reflect", "mirror", "отраж"])
            else 0.0,
        }
        problem = max(
            {
                "gradient": severity["gradient"] + severity["banding"] * 0.7 + prompt_boost["gradient"],
                "shadow": severity["shadow"] + severity["cast_shadow"] * 0.9 + prompt_boost["shadow"],
                "reflection": severity["reflection"] + prompt_boost["reflection"],
            }.items(),
            key=lambda item: item[1],
        )[0]
        strength_value = severity["overall"]
        strength = "high" if strength_value > 0.16 else "medium" if strength_value > 0.07 else "low"
        return {
            "problem": problem,
            "material": analysis.reflection_material,
            "strength": strength,
        }


def _png_bytes(image: np.ndarray) -> tuple[str, bytes, str]:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR) if image.ndim == 3 else image)
    if not success:
        raise RuntimeError("Could not encode PNG")
    return ("image.png", encoded.tobytes(), "image/png")


class MLServices:
    def __init__(self) -> None:
        self.depth = DepthProvider()
        self.segmentation = SegmentationProvider()
        self.clip = ClipProvider()
        self.inpainting = InpaintingProvider()
        self.classifier = ProblemClassifier()

    def status(self) -> MLRuntimeStatus:
        return MLRuntimeStatus(
            depth=settings.depth_provider if settings.enabled else "cv",
            segmentation=settings.segmentation_provider if settings.enabled else "cv",
            clip=settings.clip_provider if settings.enabled else "rules",
            inpainting=settings.inpaint_provider,
            classifier="sklearn" if settings.enabled else "rules",
            details={
                "ml_enabled": str(settings.enabled),
                "depth_model": settings.depth_model,
                "sam_model_type": settings.sam_model_type,
                "sam_checkpoint": "configured" if settings.sam_checkpoint else "missing",
                "clip_model": settings.clip_model,
                "classifier_path": settings.classifier_path,
                "stability_key": "configured" if settings.stability_api_key else "missing",
                "stability_inpaint_endpoint": settings.stability_inpaint_endpoint,
            },
        )


_services: MLServices | None = None


def get_ml_services() -> MLServices:
    global _services
    if _services is None:
        _services = MLServices()
    return _services
