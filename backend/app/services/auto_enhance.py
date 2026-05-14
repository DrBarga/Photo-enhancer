from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import AnalysisMaps, PromptParameters


@dataclass
class AutoEnhancePlan:
    modes: list[str]
    global_actions: list[str]
    prompt: PromptParameters
    notes: list[str]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "modes": self.modes,
            "global_actions": self.global_actions,
            "prompt": self.prompt.to_public_dict(),
            "notes": self.notes,
        }


class AutoEnhancePlanner:
    def plan(self, analysis: AnalysisMaps) -> AutoEnhancePlan:
        quality = dict(analysis.universal.get("quality", {}))
        scene_scores = dict(analysis.universal.get("scene_scores", {}))
        severity = analysis.severity()

        actions: list[str] = []
        notes: list[str] = []
        self._add_action(actions, notes, "white_balance", float(quality.get("white_balance_shift", 0.0)) > 0.20, "баланс белого нестабилен")
        self._add_action(actions, notes, "exposure_lift", float(quality.get("underexposure", 0.0)) > 0.035 or float(scene_scores.get("night", 0.0)) > 0.42, "кадр темный или ночной")
        self._add_action(actions, notes, "highlight_recovery", float(quality.get("overexposure", 0.0)) > 0.025, "есть пересветы")
        self._add_action(actions, notes, "smart_contrast", float(quality.get("contrast", 0.0)) < 0.32, "низкий локальный контраст")
        self._add_action(actions, notes, "denoise", float(quality.get("noise", 0.0)) > 0.20 or float(scene_scores.get("night", 0.0)) > 0.36, "есть шум")
        self._add_action(actions, notes, "jpeg_cleanup", float(quality.get("jpeg_artifacts", 0.0)) > 0.16, "заметны JPEG/block artifacts")
        self._add_action(actions, notes, "sharpen", 0.18 < float(quality.get("blur", 0.0)) < 0.64, "можно усилить микроконтраст")
        self._add_action(actions, notes, "dehaze", float(scene_scores.get("nature", 0.0)) > 0.34 and float(quality.get("contrast", 0.0)) < 0.46, "природная сцена с мягкой дымкой")

        modes: list[str] = []
        if severity["gradient"] + severity["banding"] * 0.8 > 0.035 or float(quality.get("banding", 0.0)) > 0.024:
            modes.append("gradient")
        if severity["reflection"] > 0.012 or float(scene_scores.get("water", 0.0)) > 0.22 or float(scene_scores.get("glass", 0.0)) > 0.22:
            modes.append("reflection")
        if severity["shadow"] + severity["cast_shadow"] * 0.9 > 0.022 or float(scene_scores.get("night", 0.0)) > 0.38:
            modes.append("shadow")
        if not modes:
            modes = ["gradient", "shadow"] if float(scene_scores.get("portrait", 0.0)) < 0.20 else ["shadow"]
            notes.append("явных проблем мало, выбран бережный режим")

        prompt = PromptParameters(
            mode_hints=modes,
            softness=0.70 if float(scene_scores.get("portrait", 0.0)) > 0.16 else 0.58,
            intensity=0.64 if severity["overall"] < 0.10 else 0.78,
            contrast_boost=1.14 if "smart_contrast" in actions else 1.0,
            denoise="denoise" in actions or "jpeg_cleanup" in actions,
            banding_fix="gradient" in modes,
            reflection_strength=0.72 if "reflection" in modes else 0.45,
            blur_strength=0.34,
            reflection_material=analysis.reflection_material,
            shadow_goal="soft" if float(scene_scores.get("portrait", 0.0)) > 0.18 else "clean",
            shadow_generate="shadow" in modes and severity["cast_shadow"] > 0.012,
            light_direction="auto",
            raw_tokens=["auto-enhance"],
        )
        return AutoEnhancePlan(modes=modes, global_actions=actions, prompt=prompt, notes=notes)

    def _add_action(self, actions: list[str], notes: list[str], action: str, enabled: bool, note: str) -> None:
        if enabled and action not in actions:
            actions.append(action)
            notes.append(note)
