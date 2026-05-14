from dataclasses import dataclass, field
from typing import Any

import numpy as np


ModeName = str


@dataclass
class PromptParameters:
    mode_hints: list[ModeName] = field(default_factory=list)
    softness: float = 0.5
    intensity: float = 0.55
    contrast_boost: float = 1.0
    denoise: bool = False
    banding_fix: bool = False
    colors: list[tuple[int, int, int]] = field(default_factory=list)
    color_names: list[str] = field(default_factory=list)
    direction: str = "vertical"
    gradient_style: str = "linear"
    gradient_stops: int = 2
    reflection_strength: float = 0.45
    blur_strength: float = 0.45
    reflection_material: str = "auto"
    shadow_goal: str = "clean"
    shadow_generate: bool = False
    light_direction: str = "auto"
    raw_tokens: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "mode_hints": self.mode_hints,
            "softness": round(self.softness, 3),
            "intensity": round(self.intensity, 3),
            "contrast_boost": round(self.contrast_boost, 3),
            "denoise": self.denoise,
            "banding_fix": self.banding_fix,
            "colors": self.color_names,
            "direction": self.direction,
            "gradient_style": self.gradient_style,
            "gradient_stops": self.gradient_stops,
            "reflection_strength": round(self.reflection_strength, 3),
            "blur_strength": round(self.blur_strength, 3),
            "reflection_material": self.reflection_material,
            "shadow_goal": self.shadow_goal,
            "shadow_generate": self.shadow_generate,
            "light_direction": self.light_direction,
            "tokens": self.raw_tokens,
        }


@dataclass
class AnalysisMaps:
    luminance: np.ndarray
    gradient: np.ndarray
    contrast: np.ndarray
    edges: np.ndarray
    gradient_problem: np.ndarray
    banding: np.ndarray
    overexposure: np.ndarray
    shadow_noise: np.ndarray
    reflection_problem: np.ndarray
    texture: np.ndarray
    specular: np.ndarray
    smooth_background: np.ndarray
    depth: np.ndarray
    cast_shadow_problem: np.ndarray
    shadow_mask: np.ndarray
    reflection_mask: np.ndarray
    problem_map: np.ndarray
    edge_density: float
    scene_type: str
    reflection_material: str
    ml_status: dict[str, Any] = field(default_factory=dict)

    def severity(self) -> dict[str, float]:
        return {
            "gradient": float(np.mean(self.gradient_problem)),
            "banding": float(np.mean(self.banding)),
            "overexposure": float(np.mean(self.overexposure)),
            "shadow": float(np.mean(self.shadow_noise)),
            "reflection": float(np.mean(self.reflection_problem)),
            "cast_shadow": float(np.mean(self.cast_shadow_problem)),
            "overall": float(np.mean(self.problem_map)),
        }


@dataclass
class MetricResult:
    key: str
    label: str
    value: int
    before: int
    after: int
    description: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "before": self.before,
            "after": self.after,
            "description": self.description,
        }
