import numpy as np

from app.models.schemas import AnalysisMaps, PromptParameters
from app.services.analyzer import ImageAnalyzer
from app.services.processors import GradientProcessor, ReflectionProcessor, ShadowProcessor


class CorrectionEngine:
    def __init__(self, analyzer: ImageAnalyzer) -> None:
        self.analyzer = analyzer
        self.processors = {
            "gradient": GradientProcessor(),
            "reflection": ReflectionProcessor(),
            "shadow": ShadowProcessor(),
        }

    def process(
        self,
        image_rgb: np.ndarray,
        modes: list[str],
        prompt: PromptParameters,
    ) -> tuple[np.ndarray, AnalysisMaps, AnalysisMaps, list[str]]:
        before_analysis = self.analyzer.analyze(image_rgb)
        current = image_rgb.copy()
        applied_modes: list[str] = []

        for mode in modes:
            processor = self.processors.get(mode)
            if processor is None:
                continue
            stage_analysis = self.analyzer.analyze(current)
            current = processor.apply(current, stage_analysis, prompt)
            applied_modes.append(mode)

        after_analysis = self.analyzer.analyze(current)
        return current, before_analysis, after_analysis, applied_modes
