from app.models.schemas import AnalysisMaps, MetricResult, PromptParameters


MODE_LABELS = {
    "gradient": "градиент",
    "reflection": "отражение",
    "shadow": "тени",
}


def build_system_comment(
    modes: list[str],
    prompt: PromptParameters,
    before: AnalysisMaps,
    after: AnalysisMaps,
    metrics: list[MetricResult],
) -> str:
    mode_text = ", ".join(MODE_LABELS.get(mode, mode) for mode in modes) or "анализ"
    severity_before = before.severity()
    severity_after = after.severity()
    changed = severity_before["overall"] - severity_after["overall"]

    if changed > 0.025:
        result = "Система заметно снизила концентрацию проблемных зон на heatmap."
    elif changed > 0.005:
        result = "Система выполнила локальную коррекцию без глобального перекрашивания кадра."
    else:
        result = "Система не стала агрессивно менять кадр, потому что проблемные зоны выражены слабо."

    prompt_parts: list[str] = []
    if prompt.color_names:
        prompt_parts.append(f"цвета: {', '.join(prompt.color_names)}")
    prompt_parts.append(f"градиент: {prompt.gradient_style}/{prompt.direction}")
    prompt_parts.append(f"материал отражения: {after.reflection_material if prompt.reflection_material == 'auto' else prompt.reflection_material}")
    prompt_parts.append(f"тени: {prompt.shadow_goal}")
    if prompt.shadow_generate:
        prompt_parts.append("генерация падающей тени")
    if prompt.banding_fix:
        prompt_parts.append("устранение banding")
    if prompt.denoise:
        prompt_parts.append("очистка шума")

    best_metric = max(metrics, key=lambda item: item.value)
    return (
        f"Обработаны режимы: {mode_text}. "
        f"Тип сцены: {after.scene_type}. "
        f"Авто-материал отражения: {after.reflection_material}. "
        f"{result} Самая сильная метрика после обработки: {best_metric.label.lower()}. "
        f"Учтены параметры: {', '.join(prompt_parts)}."
    )


def build_analysis_summary(before: AnalysisMaps, after: AnalysisMaps) -> dict[str, object]:
    before_severity = before.severity()
    after_severity = after.severity()
    return {
        "scene_type_before": before.scene_type,
        "scene_type_after": after.scene_type,
        "reflection_material_before": before.reflection_material,
        "reflection_material_after": after.reflection_material,
        "edge_density": round(after.edge_density, 4),
        "problem_severity_before": {key: round(value, 4) for key, value in before_severity.items()},
        "problem_severity_after": {key: round(value, 4) for key, value in after_severity.items()},
        "problem_level": _problem_level(after_severity["overall"]),
        "ml_depth_before": before.ml_status.get("depth", "cv"),
        "ml_depth_after": after.ml_status.get("depth", "cv"),
        "ml_depth_status": after.ml_status.get("depth_status", "fallback"),
    }


def _problem_level(value: float) -> str:
    if value >= 0.16:
        return "high"
    if value >= 0.065:
        return "medium"
    return "low"
