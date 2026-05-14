import json
from dataclasses import asdict

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.models.schemas import AnalysisMaps, PromptParameters
from app.services.analyzer import ImageAnalyzer
from app.services.auto_enhance import AutoEnhancePlanner
from app.services.correction_engine import CorrectionEngine
from app.services.global_enhancer import GlobalEnhancer
from app.services.history_service import HistoryService
from app.services.job_service import JobService
from app.services.prompt_interpreter import PromptInterpreter
from app.services.quality_evaluator import QualityEvaluator
from app.services.result_builder import build_analysis_summary, build_system_comment
from app.services.ml_providers import get_ml_services
from app.services.smart_masks import SmartMaskBuilder
from app.utils.heatmap import render_delta_heatmap_overlay, render_depth_map, render_heatmap_overlay
from app.utils.image_io import encode_png_data_url, load_image_rgb, validate_upload

router = APIRouter()

interpreter = PromptInterpreter()
analyzer = ImageAnalyzer()
engine = CorrectionEngine(analyzer)
quality_evaluator = QualityEvaluator()
smart_mask_builder = SmartMaskBuilder()
ml_services = get_ml_services()
history_service = HistoryService()
auto_planner = AutoEnhancePlanner()
global_enhancer = GlobalEnhancer()
job_service = JobService()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ml/status")
def ml_status() -> dict[str, object]:
    return _ml_runtime_status()


@router.get("/history")
def history(limit: int = 20) -> dict[str, object]:
    return {"items": history_service.list(limit)}


@router.get("/history/{record_id}")
def history_record(record_id: str) -> dict[str, object]:
    record = history_service.get(record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History record not found.")
    return record


@router.get("/history_compare/{left_id}/{right_id}")
def history_compare(left_id: str, right_id: str) -> dict[str, object]:
    comparison = history_service.compare(left_id, right_id)
    if comparison is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History records not found.")
    return comparison


@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    modes: str = Form("all"),
    prompt: str = Form(""),
) -> dict[str, object]:
    content = await image.read()
    validate_upload(image, content)
    selected_modes = _parse_modes(modes)
    prompt_parameters = interpreter.parse(prompt)
    image_rgb = load_image_rgb(content)
    analysis = analyzer.analyze(image_rgb)
    target_masks = _smart_masks(image_rgb, analysis, selected_modes, prompt_parameters)
    focused_map = _focused_problem_map(analysis, selected_modes, target_masks)
    ml_status = _ml_status(image_rgb, analysis)
    ml_understanding = _ml_understanding(image_rgb, prompt, analysis, selected_modes)

    return {
        "modes": selected_modes,
        "heatmap_image": encode_png_data_url(render_heatmap_overlay(image_rgb, focused_map)),
        "heatmap_all_image": encode_png_data_url(render_heatmap_overlay(image_rgb, analysis.problem_map)),
        "depth_map_image": encode_png_data_url(render_depth_map(analysis.depth)),
        "mode_heatmaps": _mode_heatmaps(image_rgb, analysis),
        "prompt_parameters": prompt_parameters.to_public_dict(),
        "analysis_summary": build_analysis_summary(analysis, analysis),
        "universal_analysis": analysis.universal,
        "smart_mask_coverage": _mask_coverage(target_masks),
        "ml_status": ml_status,
        "ml_understanding": ml_understanding,
        "system_comment": _build_analysis_comment(selected_modes, analysis),
    }


@router.post("/process")
async def process_image(
    image: UploadFile = File(...),
    modes: str = Form(...),
    prompt: str = Form(""),
    gradient_style: str = Form("auto"),
    gradient_direction: str = Form("auto"),
    gradient_stops: int = Form(2),
    gradient_color_a: str = Form(""),
    gradient_color_b: str = Form(""),
    gradient_color_c: str = Form(""),
    reflection_material: str = Form("auto"),
    shadow_style: str = Form("auto"),
    shadow_generate: str = Form("false"),
) -> dict[str, object]:
    content = await image.read()
    validate_upload(image, content)
    selected_modes = _parse_modes(modes)
    prompt_parameters = interpreter.parse(prompt)
    _apply_controls(
        prompt_parameters,
        gradient_style,
        gradient_direction,
        gradient_stops,
        [gradient_color_a, gradient_color_b, gradient_color_c],
        reflection_material,
        shadow_style,
        shadow_generate,
    )

    image_rgb = load_image_rgb(content)
    result_rgb, before_analysis, after_analysis, applied_modes = engine.process(
        image_rgb=image_rgb,
        modes=selected_modes,
        prompt=prompt_parameters,
    )

    target_masks = _smart_masks(image_rgb, before_analysis, selected_modes, prompt_parameters)
    metrics, total_score = quality_evaluator.evaluate(
        before_analysis,
        after_analysis,
        original_rgb=image_rgb,
        result_rgb=result_rgb,
        target_masks=target_masks,
    )
    before_map = _focused_problem_map(before_analysis, selected_modes, target_masks)
    after_map = _focused_problem_map(after_analysis, selected_modes, target_masks)
    ml_status = _ml_status(image_rgb, before_analysis, after_analysis)
    ml_understanding = _ml_understanding(image_rgb, prompt, before_analysis, selected_modes)
    heatmap_before_rgb = render_heatmap_overlay(image_rgb, before_map)
    heatmap_after_rgb = render_heatmap_overlay(result_rgb, after_map)
    heatmap_delta_rgb = render_delta_heatmap_overlay(result_rgb, before_map, after_map)
    depth_before_rgb = render_depth_map(before_analysis.depth)
    depth_after_rgb = render_depth_map(after_analysis.depth)
    metric_payload = [metric.to_public_dict() for metric in metrics]
    analysis_summary = build_analysis_summary(before_analysis, after_analysis)
    comment = build_system_comment(
        applied_modes,
        prompt_parameters,
        before_analysis,
        after_analysis,
        metrics,
    )
    history_item = history_service.save(
        input_rgb=image_rgb,
        result_rgb=result_rgb,
        heatmap_before_rgb=heatmap_before_rgb,
        heatmap_after_rgb=heatmap_after_rgb,
        heatmap_delta_rgb=heatmap_delta_rgb,
        modes=applied_modes,
        prompt=prompt,
        metrics=metric_payload,
        total_score=total_score,
        analysis_summary=analysis_summary,
        ml_understanding=ml_understanding,
    )

    return {
        "modes": applied_modes,
        "result_image": encode_png_data_url(result_rgb),
        "heatmap_image": encode_png_data_url(heatmap_before_rgb),
        "heatmap_before_image": encode_png_data_url(heatmap_before_rgb),
        "heatmap_after_image": encode_png_data_url(heatmap_after_rgb),
        "heatmap_delta_image": encode_png_data_url(heatmap_delta_rgb),
        "depth_map_before_image": encode_png_data_url(depth_before_rgb),
        "depth_map_after_image": encode_png_data_url(depth_after_rgb),
        "mode_heatmaps_before": _mode_heatmaps(image_rgb, before_analysis),
        "mode_heatmaps_after": _mode_heatmaps(result_rgb, after_analysis),
        "metrics": metric_payload,
        "total_score": total_score,
        "prompt_parameters": prompt_parameters.to_public_dict(),
        "analysis_summary": analysis_summary,
        "universal_analysis": before_analysis.universal,
        "smart_mask_coverage": _mask_coverage(target_masks),
        "ml_status": ml_status,
        "ml_understanding": ml_understanding,
        "history_item": history_item,
        "system_comment": comment,
    }


@router.post("/auto_enhance")
async def auto_enhance_image(
    image: UploadFile = File(...),
    prompt: str = Form(""),
) -> dict[str, object]:
    content = await image.read()
    validate_upload(image, content)
    image_rgb = load_image_rgb(content)
    return _auto_enhance_payload(image_rgb, prompt)


@router.post("/jobs/auto_enhance", status_code=status.HTTP_202_ACCEPTED)
async def create_auto_enhance_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = Form(""),
) -> dict[str, object]:
    content = await image.read()
    validate_upload(image, content)
    job = job_service.create(
        "auto_enhance",
        {
            "filename": image.filename,
            "content_type": image.content_type,
            "prompt": prompt,
            "size_bytes": len(content),
        },
    )

    def work(report):
        report(12, "loading image")
        image_rgb = load_image_rgb(content)
        return _auto_enhance_payload(image_rgb, prompt, report)

    background_tasks.add_task(job_service.run, str(job["id"]), work)
    return job


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    job = job_service.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job


def _auto_enhance_payload(
    image_rgb: np.ndarray,
    prompt: str,
    report=None,
) -> dict[str, object]:
    if report:
        report(20, "analyzing image")
    before_analysis = analyzer.analyze(image_rgb)
    plan = auto_planner.plan(before_analysis)
    if prompt.strip():
        user_prompt = interpreter.parse(prompt)
        plan.prompt.raw_tokens.extend(user_prompt.raw_tokens)
        plan.prompt.mode_hints = sorted(set(plan.prompt.mode_hints + user_prompt.mode_hints))
        if user_prompt.colors:
            plan.prompt.colors = user_prompt.colors
            plan.prompt.color_names = user_prompt.color_names
            plan.prompt.gradient_stops = user_prompt.gradient_stops
            if "gradient" not in plan.modes:
                plan.modes.insert(0, "gradient")
        if user_prompt.reflection_material != "auto":
            plan.prompt.reflection_material = user_prompt.reflection_material
        if user_prompt.shadow_generate:
            plan.prompt.shadow_generate = True
            if "shadow" not in plan.modes:
                plan.modes.append("shadow")

    if report:
        report(36, "running global corrections")
    prepared_rgb = global_enhancer.apply(image_rgb, before_analysis, plan.global_actions)

    if report:
        report(52, "running local correction pipeline")
    result_rgb, _stage_before, after_analysis, applied_modes = engine.process(
        image_rgb=prepared_rgb,
        modes=plan.modes,
        prompt=plan.prompt,
    )
    if not applied_modes:
        applied_modes = plan.modes

    if report:
        report(76, "building diagnostics")
    target_masks = _smart_masks(image_rgb, before_analysis, applied_modes, plan.prompt)
    metrics, total_score = quality_evaluator.evaluate(
        before_analysis,
        after_analysis,
        original_rgb=image_rgb,
        result_rgb=result_rgb,
        target_masks=target_masks,
    )
    before_map = _focused_problem_map(before_analysis, applied_modes, target_masks)
    after_map = _focused_problem_map(after_analysis, applied_modes, target_masks)
    heatmap_before_rgb = render_heatmap_overlay(image_rgb, before_map)
    heatmap_after_rgb = render_heatmap_overlay(result_rgb, after_map)
    heatmap_delta_rgb = render_delta_heatmap_overlay(result_rgb, before_map, after_map)
    depth_before_rgb = render_depth_map(before_analysis.depth)
    depth_after_rgb = render_depth_map(after_analysis.depth)
    metric_payload = [metric.to_public_dict() for metric in metrics]
    analysis_summary = build_analysis_summary(before_analysis, after_analysis)
    ml_status = _ml_status(image_rgb, before_analysis, after_analysis)
    ml_understanding = _ml_understanding(image_rgb, prompt, before_analysis, applied_modes)
    comment = build_system_comment(
        applied_modes,
        plan.prompt,
        before_analysis,
        after_analysis,
        metrics,
    )
    history_item = history_service.save(
        input_rgb=image_rgb,
        result_rgb=result_rgb,
        heatmap_before_rgb=heatmap_before_rgb,
        heatmap_after_rgb=heatmap_after_rgb,
        heatmap_delta_rgb=heatmap_delta_rgb,
        modes=applied_modes,
        prompt=prompt or "auto-enhance",
        metrics=metric_payload,
        total_score=total_score,
        analysis_summary=analysis_summary,
        ml_understanding=ml_understanding,
    )

    return {
        "modes": applied_modes,
        "auto_plan": plan.to_public_dict(),
        "result_image": encode_png_data_url(result_rgb),
        "heatmap_image": encode_png_data_url(heatmap_before_rgb),
        "heatmap_before_image": encode_png_data_url(heatmap_before_rgb),
        "heatmap_after_image": encode_png_data_url(heatmap_after_rgb),
        "heatmap_delta_image": encode_png_data_url(heatmap_delta_rgb),
        "depth_map_before_image": encode_png_data_url(depth_before_rgb),
        "depth_map_after_image": encode_png_data_url(depth_after_rgb),
        "mode_heatmaps_before": _mode_heatmaps(image_rgb, before_analysis),
        "mode_heatmaps_after": _mode_heatmaps(result_rgb, after_analysis),
        "metrics": metric_payload,
        "total_score": total_score,
        "prompt_parameters": plan.prompt.to_public_dict(),
        "analysis_summary": analysis_summary,
        "universal_analysis": before_analysis.universal,
        "smart_mask_coverage": _mask_coverage(target_masks),
        "ml_status": ml_status,
        "ml_understanding": ml_understanding,
        "history_item": history_item,
        "system_comment": f"Auto Enhance: {comment}",
    }


def _parse_modes(raw_modes: str) -> list[str]:
    try:
        parsed = json.loads(raw_modes)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in raw_modes.split(",")]

    if isinstance(parsed, str):
        parsed = [parsed]

    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Поле modes должно быть списком режимов.",
        )

    requested = [str(mode).strip().lower() for mode in parsed if str(mode).strip()]
    if "all" in requested:
        requested = list(settings.allowed_modes)

    unique_modes = []
    for mode in requested:
        if mode not in settings.allowed_modes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неизвестный режим обработки: {mode}.",
            )
        if mode not in unique_modes:
            unique_modes.append(mode)

    if not unique_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нужно выбрать хотя бы один режим обработки.",
        )
    return unique_modes


def _apply_controls(
    params: PromptParameters,
    gradient_style: str,
    gradient_direction: str,
    gradient_stops: int,
    gradient_colors: list[str],
    reflection_material: str,
    shadow_style: str,
    shadow_generate: str,
) -> None:
    if gradient_style in {"linear", "radial"}:
        params.gradient_style = gradient_style
        if gradient_style == "radial":
            params.direction = "radial"
    if gradient_direction in {"vertical", "horizontal", "diagonal", "radial"}:
        params.direction = gradient_direction
        if gradient_direction == "radial":
            params.gradient_style = "radial"
    params.gradient_stops = 3 if int(gradient_stops or 2) >= 3 else 2

    parsed_colors = [_parse_hex_color(color) for color in gradient_colors if _parse_hex_color(color) is not None]
    if len(parsed_colors) >= 2:
        params.colors = parsed_colors[: params.gradient_stops]
        params.color_names = [f"custom-{index + 1}" for index in range(len(params.colors))]
        params.banding_fix = True

    if reflection_material in {"auto", "water", "asphalt", "mirror", "glass"}:
        params.reflection_material = reflection_material
    if shadow_style in {"clean", "soft", "dramatic"}:
        params.shadow_goal = shadow_style
    if str(shadow_generate).lower() in {"true", "1", "yes", "on"}:
        params.shadow_generate = True


def _parse_hex_color(value: str) -> tuple[int, int, int] | None:
    value = value.strip()
    if not value:
        return None
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        return None
    try:
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return None


def _focused_problem_map(
    analysis: AnalysisMaps,
    modes: list[str],
    target_masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    maps: list[np.ndarray] = []
    if "gradient" in modes:
        maps.append(_masked_mode_map(
            np.maximum.reduce([analysis.gradient_problem, analysis.banding, analysis.overexposure * 0.75]),
            target_masks,
            "gradient",
            min_signal=0.48,
        ))
    if "reflection" in modes:
        maps.append(_masked_mode_map(analysis.reflection_problem, target_masks, "reflection", min_signal=0.68))
    if "shadow" in modes:
        maps.append(_masked_mode_map(
            np.maximum.reduce([analysis.shadow_noise, analysis.cast_shadow_problem, analysis.shadow_mask * 0.35]),
            target_masks,
            "shadow",
            min_signal=0.46,
        ))
    if not maps:
        return analysis.problem_map
    return np.clip(np.maximum.reduce(maps), 0.0, 1.0)


def _masked_mode_map(
    problem_map: np.ndarray,
    target_masks: dict[str, np.ndarray] | None,
    mode: str,
    min_signal: float = 0.0,
) -> np.ndarray:
    if not target_masks or mode not in target_masks:
        return problem_map
    mask = np.clip(target_masks[mode].astype(np.float32), 0.0, 1.0)
    focused = np.maximum(problem_map * mask, mask * min_signal)
    return np.clip(focused, 0.0, 1.0)


def _mode_heatmaps(image_rgb: np.ndarray, analysis: AnalysisMaps) -> dict[str, str]:
    maps = {
        "gradient": np.maximum.reduce([analysis.gradient_problem, analysis.banding, analysis.overexposure * 0.75]),
        "reflection": analysis.reflection_problem,
        "shadow": np.maximum.reduce([analysis.shadow_noise, analysis.cast_shadow_problem, analysis.shadow_mask * 0.35]),
        "all": analysis.problem_map,
    }
    return {
        key: encode_png_data_url(render_heatmap_overlay(image_rgb, value))
        for key, value in maps.items()
    }


def _smart_mask_coverage(
    image_rgb: np.ndarray,
    analysis: AnalysisMaps,
    modes: list[str] | None = None,
    prompt: PromptParameters | None = None,
) -> dict[str, dict[str, float]]:
    return _mask_coverage(_smart_masks(image_rgb, analysis, modes, prompt))


def _smart_masks(
    image_rgb: np.ndarray,
    analysis: AnalysisMaps,
    modes: list[str] | None = None,
    prompt: PromptParameters | None = None,
) -> dict[str, np.ndarray]:
    selected = set(modes or settings.allowed_modes)
    masks: dict[str, np.ndarray] = {}
    if "gradient" in selected:
        gradient_processor = engine.processors.get("gradient")
        if prompt is not None and hasattr(gradient_processor, "target_mask"):
            masks["gradient"] = gradient_processor.target_mask(image_rgb, analysis, prompt)[0]
        else:
            masks["gradient"] = smart_mask_builder.gradient_mask(image_rgb, analysis)
    if "reflection" in selected:
        reflection_processor = engine.processors.get("reflection")
        if prompt is not None and hasattr(reflection_processor, "target_mask"):
            masks["reflection"] = reflection_processor.target_mask(image_rgb, analysis, prompt)
        else:
            masks["reflection"] = smart_mask_builder.reflection_mask(image_rgb, analysis)
    if "shadow" in selected:
        masks["shadow"] = smart_mask_builder.shadow_mask(image_rgb, analysis)
    return masks


def _mask_coverage(masks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean_percent": round(float(mask.mean()) * 100.0, 2),
            "active_area_percent": round(float((mask > 0.1).mean()) * 100.0, 2),
        }
        for key, mask in masks.items()
    }


def _ml_runtime_status() -> dict[str, object]:
    return asdict(ml_services.status())


def _ml_status(
    image_rgb: np.ndarray,
    before: AnalysisMaps,
    after: AnalysisMaps | None = None,
) -> dict[str, object]:
    segmentation_result = ml_services.segmentation.masks(image_rgb, before)
    status_payload: dict[str, object] = {
        "runtime": _ml_runtime_status(),
        "depth_before": before.ml_status,
        "segmentation": {
            "provider": segmentation_result.provider,
            "status": segmentation_result.status,
            "detail": segmentation_result.detail[:160],
            "confidence": round(float(segmentation_result.confidence), 3),
        },
    }
    if after is not None:
        status_payload["depth_after"] = after.ml_status
    return status_payload


def _ml_understanding(
    image_rgb: np.ndarray,
    prompt: str,
    analysis: AnalysisMaps,
    selected_modes: list[str],
) -> dict[str, object]:
    clip_result = ml_services.clip.classify(image_rgb, prompt)
    classifier_result = ml_services.classifier.predict(prompt, analysis)
    classifier_value = classifier_result.value if isinstance(classifier_result.value, dict) else {}
    problem = classifier_value.get("problem", _dominant_problem(analysis))
    if len(selected_modes) == 1 and problem not in selected_modes:
        problem = selected_modes[0]
    return {
        "problem": problem,
        "material": classifier_value.get("material", analysis.reflection_material),
        "strength": classifier_value.get("strength", _problem_strength(analysis)),
        "scene_type": analysis.scene_type,
        "reflection_material": analysis.reflection_material,
        "clip": _provider_public(clip_result),
        "classifier": _provider_public(classifier_result),
    }


def _provider_public(result) -> dict[str, object]:
    return {
        "provider": result.provider,
        "status": result.status,
        "detail": result.detail[:160],
        "confidence": round(float(result.confidence), 3),
        "value": result.value,
    }


def _dominant_problem(analysis: AnalysisMaps) -> str:
    severity = analysis.severity()
    return max(
        {
            "gradient": severity["gradient"] + severity["banding"] * 0.7,
            "shadow": severity["shadow"] + severity["cast_shadow"] * 0.9,
            "reflection": severity["reflection"],
        }.items(),
        key=lambda item: item[1],
    )[0]


def _problem_strength(analysis: AnalysisMaps) -> str:
    overall = analysis.severity()["overall"]
    if overall > 0.16:
        return "high"
    if overall > 0.07:
        return "medium"
    return "low"


def _build_analysis_comment(modes: list[str], analysis: AnalysisMaps) -> str:
    mode_text = ", ".join(modes)
    severity = analysis.severity()
    return (
        f"Анализ выполнен до обработки. Фокус heatmap: {mode_text}. "
        f"Тип сцены: {analysis.scene_type}. "
        f"Авто-материал отражения: {analysis.reflection_material}. "
        f"Общий индекс проблемных зон: {severity['overall']:.3f}."
    )
