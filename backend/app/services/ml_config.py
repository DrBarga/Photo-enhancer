import os
from dataclasses import dataclass


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MLSettings:
    enabled: bool = _truthy(os.getenv("AI_LIGHT_ML_ENABLED"))
    depth_provider: str = os.getenv("AI_LIGHT_DEPTH_PROVIDER", "auto")
    depth_model: str = os.getenv("AI_LIGHT_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
    segmentation_provider: str = os.getenv("AI_LIGHT_SEGMENTATION_PROVIDER", "cv")
    sam_checkpoint: str = os.getenv("AI_LIGHT_SAM_CHECKPOINT", "")
    sam_model_type: str = os.getenv("AI_LIGHT_SAM_MODEL_TYPE", "vit_b")
    clip_provider: str = os.getenv("AI_LIGHT_CLIP_PROVIDER", "auto")
    clip_model: str = os.getenv("AI_LIGHT_CLIP_MODEL", "openai/clip-vit-base-patch32")
    inpaint_provider: str = os.getenv("AI_LIGHT_INPAINT_PROVIDER", "none")
    stability_api_key: str = os.getenv("STABILITY_API_KEY", "")
    stability_inpaint_endpoint: str = os.getenv(
        "AI_LIGHT_STABILITY_INPAINT_ENDPOINT",
        "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
    )
    classifier_path: str = os.getenv("AI_LIGHT_CLASSIFIER_PATH", "backend/models/problem_classifier.joblib")


settings = MLSettings()
