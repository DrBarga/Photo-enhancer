import { useEffect, useMemo, useRef, useState } from "react";
import defaultBefore from "./assets/hero.png";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "";

const processingOptions = [
  { value: "gradient", label: "Градиент" },
  { value: "reflection", label: "Отражение" },
  { value: "shadow", label: "Тени" },
];

const emptyMetrics = [
  { key: "gradient_smoothness", label: "Плавность градиента", value: 0, before: 0, after: 0 },
  { key: "banding_reduction", label: "Снижение banding", value: 0, before: 0, after: 0 },
  { key: "reflection_coherence", label: "Натуральность отражения", value: 0, before: 0, after: 0 },
  { key: "shadow_cleanliness", label: "Чистота теней", value: 0, before: 0, after: 0 },
  { key: "cast_shadow_realism", label: "Реалистичность тени", value: 0, before: 0, after: 0 },
  { key: "heatmap_risk", label: "Индекс проблемных зон", value: 0, before: 0, after: 0 },
];

function SurfaceButton({ children, onClick, disabled = false, className = "", type = "button" }) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`flex h-11 items-center justify-center rounded-lg border border-white/12 bg-white/[0.07] px-4 text-sm font-medium text-zinc-100 transition hover:border-cyan-300/50 hover:bg-cyan-300/10 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
    >
      {children}
    </button>
  );
}

function FieldLabel({ children }) {
  return <label className="mb-2 block text-xs font-medium uppercase tracking-wide text-zinc-500">{children}</label>;
}

function SelectField({ value, onChange, options }) {
  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="h-10 w-full rounded-lg border border-white/10 bg-[#0b1018] px-3 text-sm text-zinc-100 outline-none focus:border-cyan-300/50"
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

function modeSummary(modes) {
  if (modes.length === processingOptions.length) return "Все режимы";
  return modes
    .map((mode) => processingOptions.find((option) => option.value === mode)?.label)
    .filter(Boolean)
    .join(", ");
}

export default function App() {
  const fileInputRef = useRef(null);
  const compareRef = useRef(null);
  const objectUrlRef = useRef(null);

  const [uploadedFile, setUploadedFile] = useState(null);
  const [beforeImage, setBeforeImage] = useState(defaultBefore);
  const [afterImage, setAfterImage] = useState(defaultBefore);
  const [heatmapBefore, setHeatmapBefore] = useState(null);
  const [heatmapAfter, setHeatmapAfter] = useState(null);
  const [heatmapDelta, setHeatmapDelta] = useState(null);
  const [depthMapBefore, setDepthMapBefore] = useState(null);
  const [depthMapAfter, setDepthMapAfter] = useState(null);
  const [heatmapView, setHeatmapView] = useState("result");

  const [comparePosition, setComparePosition] = useState(50);
  const [selectedModes, setSelectedModes] = useState(["gradient", "reflection", "shadow"]);
  const [prompt, setPrompt] = useState("radial three color pink peach blue gradient, strong wet reflection, dramatic clean shadows");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastAction, setLastAction] = useState("Готово к анализу");

  const [gradientStyle, setGradientStyle] = useState("linear");
  const [gradientDirection, setGradientDirection] = useState("vertical");
  const [gradientStops, setGradientStops] = useState(2);
  const [gradientColors, setGradientColors] = useState(["#f472b6", "#fbbf92", "#60a5fa"]);
  const [reflectionMaterial, setReflectionMaterial] = useState("auto");
  const [shadowStyle, setShadowStyle] = useState("clean");
  const [shadowGenerate, setShadowGenerate] = useState(false);

  const [metrics, setMetrics] = useState(emptyMetrics);
  const [totalScore, setTotalScore] = useState(null);
  const [systemComment, setSystemComment] = useState("Загрузите изображение: система сразу построит heatmap проблемных зон.");
  const [analysisSummary, setAnalysisSummary] = useState(null);
  const [promptParameters, setPromptParameters] = useState(null);
  const [smartMaskCoverage, setSmartMaskCoverage] = useState(null);
  const [mlStatus, setMlStatus] = useState(null);
  const [mlUnderstanding, setMlUnderstanding] = useState(null);
  const [historyItems, setHistoryItems] = useState([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);

  const [imageNaturalSize, setImageNaturalSize] = useState({ width: 1, height: 1 });
  const [imageBounds, setImageBounds] = useState({ left: 0, top: 0, width: 100, height: 100 });

  const activeModeText = useMemo(() => modeSummary(selectedModes), [selectedModes]);
  const currentAfterImage = useMemo(() => {
    if (heatmapView === "before" && heatmapBefore) return heatmapBefore;
    if (heatmapView === "after" && heatmapAfter) return heatmapAfter;
    if (heatmapView === "delta" && heatmapDelta) return heatmapDelta;
    if (heatmapView === "depthBefore" && depthMapBefore) return depthMapBefore;
    if (heatmapView === "depthAfter" && depthMapAfter) return depthMapAfter;
    return afterImage;
  }, [afterImage, depthMapAfter, depthMapBefore, heatmapAfter, heatmapBefore, heatmapDelta, heatmapView]);

  useEffect(() => {
    const container = compareRef.current;
    if (!container) return;

    const recalcBounds = () => {
      const rect = container.getBoundingClientRect();
      const containerWidth = rect.width;
      const containerHeight = rect.height;
      const imageRatio = imageNaturalSize.width / imageNaturalSize.height;
      const containerRatio = containerWidth / containerHeight;

      let width = containerWidth;
      let height = containerHeight;
      let left = 0;
      let top = 0;

      if (imageRatio > containerRatio) {
        height = containerWidth / imageRatio;
        top = (containerHeight - height) / 2;
      } else {
        width = containerHeight * imageRatio;
        left = (containerWidth - width) / 2;
      }

      setImageBounds({ left, top, width, height });
    };

    recalcBounds();
    const observer = new ResizeObserver(recalcBounds);
    observer.observe(container);
    return () => observer.disconnect();
  }, [imageNaturalSize]);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      setIsHistoryLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/history?limit=12`);
      if (!response.ok) return;
      const data = await response.json();
      setHistoryItems(data.items ?? []);
    } finally {
      setIsHistoryLoading(false);
    }
  };

  const buildFormData = async (file, includeControls = false) => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("modes", JSON.stringify(selectedModes));
    formData.append("prompt", prompt);

    if (includeControls) {
      formData.append("gradient_style", gradientStyle);
      formData.append("gradient_direction", gradientDirection);
      formData.append("gradient_stops", String(gradientStops));
      formData.append("gradient_color_a", gradientColors[0]);
      formData.append("gradient_color_b", gradientColors[1]);
      formData.append("gradient_color_c", gradientStops >= 3 ? gradientColors[2] : "");
      formData.append("reflection_material", reflectionMaterial);
      formData.append("shadow_style", shadowStyle);
      formData.append("shadow_generate", String(shadowGenerate));
    }

    return formData;
  };

  const getProcessingFile = async () => {
    if (uploadedFile) return uploadedFile;
    const response = await fetch(beforeImage || defaultBefore);
    const blob = await response.blob();
    return new File([blob], beforeImage === defaultBefore ? "sample-image.png" : "current-image.png", { type: blob.type || "image/png" });
  };

  const analyzeFile = async (file) => {
    try {
      setIsAnalyzing(true);
      setLastAction("Анализ изображения...");
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: await buildFormData(file, false),
      });
      if (!response.ok) throw new Error("Не удалось выполнить анализ изображения");
      const data = await response.json();
      setHeatmapBefore(data.heatmap_image);
      setDepthMapBefore(data.depth_map_image ?? null);
      setDepthMapAfter(null);
      setHeatmapAfter(null);
      setHeatmapDelta(null);
      setHeatmapView("before");
      setAnalysisSummary(data.analysis_summary ?? null);
      setPromptParameters(data.prompt_parameters ?? null);
      setSmartMaskCoverage(data.smart_mask_coverage ?? null);
      setMlStatus(data.ml_status ?? null);
      setMlUnderstanding(data.ml_understanding ?? null);
      setSystemComment(data.system_comment ?? "Анализ выполнен.");
      setLastAction("Heatmap построена до обработки");
    } catch (error) {
      setLastAction(error.message || "Анализ не выполнен");
      setSystemComment("Проверьте, что backend запущен на 127.0.0.1:8000.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const updateComparePosition = (clientX) => {
    const container = compareRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const localX = clientX - rect.left;
    const minX = imageBounds.left;
    const maxX = imageBounds.left + imageBounds.width;
    const clampedX = Math.max(minX, Math.min(maxX, localX));
    setComparePosition(((clampedX - imageBounds.left) / imageBounds.width) * 100);
  };

  const handlePointerDown = (event) => {
    event.preventDefault();
    updateComparePosition(event.clientX);
    const handleMove = (moveEvent) => updateComparePosition(moveEvent.clientX);
    const handleUp = () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
    };
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
  };

  const toggleMode = (mode) => {
    setSelectedModes((current) => {
      if (current.includes(mode)) return current.filter((item) => item !== mode);
      return [...current, mode];
    });
    setLastAction("Режимы обновлены");
  };

  const selectAllModes = () => {
    setSelectedModes(processingOptions.map((option) => option.value));
    setLastAction("Выбраны все режимы");
  };

  const handleUploadClick = () => fileInputRef.current?.click();

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const localUrl = URL.createObjectURL(file);
    objectUrlRef.current = localUrl;

    setUploadedFile(file);
    setBeforeImage(localUrl);
    setAfterImage(localUrl);
    setComparePosition(50);
    setMetrics(emptyMetrics);
    setTotalScore(null);
    setHeatmapBefore(null);
    setHeatmapAfter(null);
    setHeatmapDelta(null);
    setDepthMapBefore(null);
    setDepthMapAfter(null);
    setMlStatus(null);
    setMlUnderstanding(null);
    await analyzeFile(file);
  };

  const handleAnalyzeAgain = async () => {
    const file = await getProcessingFile();
    await analyzeFile(file);
  };

  const handleProcess = async () => {
    if (selectedModes.length === 0) {
      setLastAction("Выберите хотя бы один режим");
      return;
    }

    try {
      setIsProcessing(true);
      const sourceImage = beforeImage;
      const processingFile = await getProcessingFile();
      setLastAction("Идет профессиональная локальная обработка...");
      const response = await fetch(`${API_BASE_URL}/api/process`, {
        method: "POST",
        body: await buildFormData(processingFile, true),
      });

      if (!response.ok) {
        let detail = "Сервер вернул ошибку обработки.";
        try {
          const errorData = await response.json();
          detail = errorData.detail || detail;
        } catch {
          detail = response.statusText || detail;
        }
        throw new Error(detail);
      }

      const data = await response.json();
      setBeforeImage(sourceImage);
      setAfterImage(data.result_image ?? sourceImage);
      setHeatmapBefore(data.heatmap_before_image ?? data.heatmap_image);
      setHeatmapAfter(data.heatmap_after_image ?? null);
      setHeatmapDelta(data.heatmap_delta_image ?? null);
      setDepthMapBefore(data.depth_map_before_image ?? null);
      setDepthMapAfter(data.depth_map_after_image ?? null);
      setHeatmapView("result");
      setMetrics(data.metrics ?? emptyMetrics);
      setTotalScore(data.total_score ?? null);
      setSystemComment(data.system_comment ?? "Обработка завершена.");
      setAnalysisSummary(data.analysis_summary ?? null);
      setPromptParameters(data.prompt_parameters ?? null);
      setSmartMaskCoverage(data.smart_mask_coverage ?? null);
      setMlStatus(data.ml_status ?? null);
      setMlUnderstanding(data.ml_understanding ?? null);
      if (data.history_item) {
        setHistoryItems((current) => [data.history_item, ...current.filter((item) => item.id !== data.history_item.id)].slice(0, 12));
      }
      setComparePosition(50);
      setLastAction(`Обработка завершена: ${modeSummary(data.modes ?? selectedModes)}`);
    } catch (error) {
      setLastAction(error.message || "Не удалось обработать изображение");
      setSystemComment("Проверьте, что backend запущен на 127.0.0.1:8000.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSelectedModes(["gradient", "reflection", "shadow"]);
    setPrompt("radial three color pink peach blue gradient, strong wet reflection, dramatic clean shadows");
    setComparePosition(50);
    setAfterImage(beforeImage);
    setMetrics(emptyMetrics);
    setTotalScore(null);
    setHeatmapView(heatmapBefore ? "before" : "result");
    setMlStatus(null);
    setMlUnderstanding(null);
    setSystemComment("Параметры сброшены.");
    setLastAction("Параметры сброшены");
  };

  const openHistoryItem = async (id) => {
    try {
      setIsHistoryLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/history/${id}`);
      if (!response.ok) throw new Error("Не удалось открыть результат из истории");
      const data = await response.json();
      const images = data.image_data ?? {};
      if (!images.input || !images.result) return;
      setUploadedFile(null);
      setBeforeImage(images.input);
      setAfterImage(images.result);
      setHeatmapBefore(images.heatmap_before ?? null);
      setHeatmapAfter(images.heatmap_after ?? null);
      setHeatmapDelta(images.heatmap_delta ?? null);
      setDepthMapBefore(null);
      setDepthMapAfter(null);
      setHeatmapView("result");
      setMetrics(data.metrics ?? emptyMetrics);
      setTotalScore(data.total_score ?? null);
      setAnalysisSummary(data.analysis_summary ?? null);
      setMlUnderstanding(data.ml_understanding ?? null);
      setPrompt(data.prompt ?? prompt);
      setSelectedModes(data.modes?.length ? data.modes : selectedModes);
      setComparePosition(50);
      setLastAction(`Открыт результат из истории: ${data.total_score ?? "--"} score`);
    } catch (error) {
      setLastAction(error.message || "История недоступна");
    } finally {
      setIsHistoryLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      const response = await fetch(currentAfterImage);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = heatmapView === "result" ? "processed-result.png" : `heatmap-${heatmapView}.png`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setLastAction("Файл сохранен");
    } catch {
      setLastAction("Не удалось сохранить файл");
    }
  };

  const updateGradientColor = (index, value) => {
    setGradientColors((current) => current.map((color, colorIndex) => (colorIndex === index ? value : color)));
  };

  const dividerLeft = imageBounds.left + (imageBounds.width * comparePosition) / 100;
  const visibleAfterWidth = imageBounds.width - (imageBounds.width * comparePosition) / 100;

  return (
    <div className="min-h-screen bg-[#07090f] text-white">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-5 py-6 lg:px-8">
        <header className="mb-5 flex flex-col gap-3 border-b border-white/10 pb-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight lg:text-3xl">AI Light Pro</h1>
            <div className="mt-2 text-sm text-zinc-400">{activeModeText || "Режим не выбран"}</div>
          </div>
          <div className="rounded-lg border border-cyan-300/25 bg-cyan-300/8 px-3 py-2 text-sm text-cyan-100">
            {isAnalyzing ? "Анализ..." : lastAction}
          </div>
        </header>

        <main className="grid flex-1 gap-5 lg:grid-cols-[minmax(0,1.58fr)_minmax(390px,0.95fr)]">
          <section className="flex min-h-0 flex-col gap-4">
            <div
              ref={compareRef}
              onPointerDown={handlePointerDown}
              className="relative flex h-[58vh] min-h-[390px] w-full cursor-ew-resize items-center justify-center overflow-hidden rounded-lg border border-white/10 bg-black shadow-[0_24px_70px_rgba(0,0,0,0.32)]"
            >
              <img
                src={beforeImage}
                alt=""
                className="hidden"
                onLoad={(event) => {
                  setImageNaturalSize({
                    width: event.currentTarget.naturalWidth || 1,
                    height: event.currentTarget.naturalHeight || 1,
                  });
                }}
              />

              <div
                className="absolute overflow-hidden"
                style={{ left: `${imageBounds.left}px`, top: `${imageBounds.top}px`, width: `${imageBounds.width}px`, height: `${imageBounds.height}px` }}
              >
                <img
                  src={beforeImage}
                  alt="До"
                  draggable={false}
                  className="absolute left-0 top-0 select-none"
                  style={{ width: `${imageBounds.width}px`, height: `${imageBounds.height}px`, objectFit: "contain", objectPosition: "center", maxWidth: "none", maxHeight: "none" }}
                />
              </div>

              <div
                className="absolute overflow-hidden"
                style={{ left: `${dividerLeft}px`, top: `${imageBounds.top}px`, width: `${visibleAfterWidth}px`, height: `${imageBounds.height}px` }}
              >
                <img
                  src={currentAfterImage}
                  alt="После"
                  draggable={false}
                  className="absolute top-0 select-none"
                  style={{ left: `-${(imageBounds.width * comparePosition) / 100}px`, width: `${imageBounds.width}px`, height: `${imageBounds.height}px`, objectFit: "contain", objectPosition: "center", maxWidth: "none", maxHeight: "none" }}
                />
              </div>

              <div className="absolute inset-y-0 w-[2px] bg-white/90 shadow-[0_0_18px_rgba(255,255,255,0.7)]" style={{ left: `${dividerLeft}px`, transform: "translateX(-50%)" }} />
              <button
                type="button"
                onPointerDown={handlePointerDown}
                className="absolute top-1/2 z-10 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-white/25 bg-[#111827]/85 text-lg text-white shadow-lg backdrop-blur"
                style={{ left: `${dividerLeft}px` }}
              >
                ↔
              </button>
              <div className="absolute left-4 top-4 rounded-md border border-white/15 bg-black/55 px-3 py-1 text-xs text-zinc-100 backdrop-blur">До</div>
              <div className="absolute right-4 top-4 rounded-md border border-cyan-300/30 bg-cyan-300/10 px-3 py-1 text-xs text-cyan-50 backdrop-blur">
                {heatmapView === "result" ? "После" : heatmapView.startsWith("depth") ? "Depth map" : `Heatmap ${heatmapView}`}
              </div>
            </div>

            <div className="rounded-lg border border-white/10 bg-white/[0.045] p-4">
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
                <div className="space-y-4">
                  <div>
                    <FieldLabel>Режимы коррекции</FieldLabel>
                    <div className="grid gap-2 sm:grid-cols-3">
                      {processingOptions.map((option) => {
                        const active = selectedModes.includes(option.value);
                        return (
                          <button
                            key={option.value}
                            type="button"
                            onClick={() => toggleMode(option.value)}
                            aria-pressed={active}
                            className={`flex h-11 items-center justify-between rounded-lg border px-3 text-sm transition ${active ? "border-cyan-300/55 bg-cyan-300/12 text-cyan-50" : "border-white/10 bg-black/25 text-zinc-300 hover:border-white/25"}`}
                          >
                            <span>{option.label}</span>
                            <span className={`flex h-5 w-5 items-center justify-center rounded border text-xs ${active ? "border-cyan-200 bg-cyan-200 text-[#071018]" : "border-white/25 text-transparent"}`}>✓</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-3">
                    <div>
                      <FieldLabel>Градиент</FieldLabel>
                      <SelectField value={gradientStyle} onChange={setGradientStyle} options={[{ value: "linear", label: "Линейный" }, { value: "radial", label: "Радиальный" }]} />
                    </div>
                    <div>
                      <FieldLabel>Направление</FieldLabel>
                      <SelectField value={gradientDirection} onChange={setGradientDirection} options={[{ value: "vertical", label: "Вертикальный" }, { value: "horizontal", label: "Горизонтальный" }, { value: "diagonal", label: "Диагональный" }, { value: "radial", label: "От центра" }]} />
                    </div>
                    <div>
                      <FieldLabel>Цвета</FieldLabel>
                      <SelectField value={String(gradientStops)} onChange={(value) => setGradientStops(Number(value))} options={[{ value: "2", label: "Двухцветный" }, { value: "3", label: "Трехцветный" }]} />
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-3">
                    {gradientColors.map((color, index) => (
                      <div key={index} className={index === 2 && gradientStops < 3 ? "opacity-35" : ""}>
                        <FieldLabel>Цвет {index + 1}</FieldLabel>
                        <input
                          type="color"
                          value={color}
                          disabled={index === 2 && gradientStops < 3}
                          onChange={(event) => updateGradientColor(index, event.target.value)}
                          className="h-10 w-full rounded-lg border border-white/10 bg-[#0b1018] p-1"
                        />
                      </div>
                    ))}
                  </div>

                  <div className="grid gap-3 md:grid-cols-3">
                    <div>
                      <FieldLabel>Материал отражения</FieldLabel>
                      <SelectField value={reflectionMaterial} onChange={setReflectionMaterial} options={[{ value: "auto", label: "Авто" }, { value: "water", label: "Вода" }, { value: "asphalt", label: "Асфальт" }, { value: "mirror", label: "Зеркало" }, { value: "glass", label: "Стекло" }]} />
                    </div>
                    <div>
                      <FieldLabel>Тип тени</FieldLabel>
                      <SelectField value={shadowStyle} onChange={setShadowStyle} options={[{ value: "clean", label: "Чистая" }, { value: "soft", label: "Мягкая" }, { value: "dramatic", label: "Драматическая" }]} />
                    </div>
                    <label className="mt-6 flex h-10 items-center gap-3 rounded-lg border border-white/10 bg-black/25 px-3 text-sm text-zinc-200">
                      <input type="checkbox" checked={shadowGenerate} onChange={(event) => setShadowGenerate(event.target.checked)} />
                      Генерировать тень
                    </label>
                  </div>

                  <div>
                    <FieldLabel>Текстовый промпт</FieldLabel>
                    <textarea
                      value={prompt}
                      onChange={(event) => setPrompt(event.target.value)}
                      rows={3}
                      placeholder="Например: radial peach blue gradient, wet asphalt reflection, soft clean shadow"
                      className="min-h-24 w-full resize-y rounded-lg border border-white/10 bg-black/25 px-4 py-3 text-sm leading-6 text-white outline-none placeholder:text-zinc-500 focus:border-cyan-300/50"
                    />
                  </div>
                </div>

                <div className="grid content-start gap-3 sm:grid-cols-2 xl:grid-cols-1">
                  <SurfaceButton onClick={handleUploadClick}>Загрузить изображение</SurfaceButton>
                  <SurfaceButton onClick={handleAnalyzeAgain} disabled={isAnalyzing}>Анализ / Heatmap</SurfaceButton>
                  <SurfaceButton onClick={selectAllModes}>Все режимы</SurfaceButton>
                  <SurfaceButton onClick={handleProcess} disabled={isProcessing || isAnalyzing}>{isProcessing ? "Обработка..." : "Запустить"}</SurfaceButton>
                  <SurfaceButton onClick={handleReset}>Сбросить</SurfaceButton>
                  <SurfaceButton onClick={handleDownload}>Скачать текущий вид</SurfaceButton>

                  <div className="grid grid-cols-2 gap-2 border-t border-white/10 pt-3 xl:grid-cols-1">
                    <SurfaceButton onClick={() => setHeatmapView("result")}>Результат</SurfaceButton>
                    <SurfaceButton onClick={() => setHeatmapView("before")} disabled={!heatmapBefore}>Heatmap до</SurfaceButton>
                    <SurfaceButton onClick={() => setHeatmapView("after")} disabled={!heatmapAfter}>Heatmap после</SurfaceButton>
                    <SurfaceButton onClick={() => setHeatmapView("delta")} disabled={!heatmapDelta}>Что исправлено</SurfaceButton>
                    <SurfaceButton onClick={() => setHeatmapView("depthBefore")} disabled={!depthMapBefore}>Depth до</SurfaceButton>
                    <SurfaceButton onClick={() => setHeatmapView("depthAfter")} disabled={!depthMapAfter}>Depth после</SurfaceButton>
                  </div>
                </div>
              </div>

              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
            </div>
          </section>

          <aside className="rounded-lg border border-white/10 bg-white/[0.045] p-4">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-medium">Диагностика качества</h2>
                <div className="mt-1 text-sm text-zinc-400">
                  {analysisSummary?.problem_level ? `Уровень проблем: ${analysisSummary.problem_level}` : "Ожидает анализа"}
                </div>
              </div>
              <div className="rounded-lg border border-amber-300/30 bg-amber-300/10 px-4 py-3 text-center">
                <div className="text-xs text-amber-100/75">Score</div>
                <div className="text-3xl font-semibold text-amber-100">{totalScore ?? "--"}</div>
              </div>
            </div>

            <div className="grid gap-3">
              {metrics.map((metric) => (
                <div key={metric.key} className="rounded-lg border border-white/10 bg-black/25 p-3">
                  <div className="mb-2 flex items-center justify-between gap-3 text-sm">
                    <span className="text-zinc-200">{metric.label}</span>
                    <span className="font-medium text-cyan-100">{metric.value}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-white/10">
                    <div className="h-2 rounded-full bg-gradient-to-r from-cyan-300 via-emerald-300 to-amber-200" style={{ width: `${metric.value}%` }} />
                  </div>
                  {totalScore !== null && (
                    <div className="mt-2 flex justify-between text-xs text-zinc-500">
                      <span>До: {metric.before}%</span>
                      <span>После: {metric.after}%</span>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="mt-4 rounded-lg border border-white/10 bg-black/25 p-4 text-sm leading-6 text-zinc-300">
              <div className="mb-2 text-zinc-100">Комментарий системы</div>
              {systemComment}
            </div>

            {analysisSummary && (
              <div className="mt-4 rounded-lg border border-white/10 bg-black/25 p-4 text-sm text-zinc-300">
                <div className="mb-3 text-zinc-100">Световая структура</div>
                <div className="grid grid-cols-2 gap-2">
                  <span className="text-zinc-500">Сцена</span>
                  <span>{analysisSummary.scene_type_after}</span>
                  <span className="text-zinc-500">Материал</span>
                  <span>{analysisSummary.reflection_material_after}</span>
                  <span className="text-zinc-500">Edge density</span>
                  <span>{analysisSummary.edge_density}</span>
                </div>
              </div>
            )}

            {promptParameters && (
              <div className="mt-4 rounded-lg border border-white/10 bg-black/25 p-4 text-sm text-zinc-300">
                <div className="mb-3 text-zinc-100">Параметры обработки</div>
                <div className="grid grid-cols-2 gap-2">
                  <span className="text-zinc-500">Gradient</span>
                  <span>{promptParameters.gradient_style}/{promptParameters.direction}</span>
                  <span className="text-zinc-500">Colors</span>
                  <span>{promptParameters.colors?.join(", ") || "нет"}</span>
                  <span className="text-zinc-500">Reflection</span>
                  <span>{promptParameters.reflection_material}</span>
                  <span className="text-zinc-500">Shadow</span>
                  <span>{promptParameters.shadow_goal}</span>
                </div>
              </div>
            )}

            {smartMaskCoverage && (
              <div className="mt-4 rounded-lg border border-white/10 bg-black/25 p-4 text-sm text-zinc-300">
                <div className="mb-3 text-zinc-100">ML-зоны коррекции</div>
                <div className="grid gap-2">
                  {processingOptions.map((option) => {
                    const coverage = smartMaskCoverage[option.value];
                    if (!coverage) return null;
                    return (
                      <div key={option.value} className="flex justify-between gap-3">
                        <span className="text-zinc-500">{option.label}</span>
                        <span>{coverage.active_area_percent}% изображения</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {(mlStatus || mlUnderstanding) && (
              <div className="mt-4 rounded-lg border border-white/10 bg-black/25 p-4 text-sm text-zinc-300">
                <div className="mb-3 text-zinc-100">ML-понимание сцены</div>
                {mlUnderstanding && (
                  <div className="grid grid-cols-2 gap-2">
                    <span className="text-zinc-500">Проблема</span>
                    <span>{mlUnderstanding.problem}</span>
                    <span className="text-zinc-500">Материал</span>
                    <span>{mlUnderstanding.material}</span>
                    <span className="text-zinc-500">Сила</span>
                    <span>{mlUnderstanding.strength}</span>
                  </div>
                )}
                {mlStatus && (
                  <div className="mt-3 grid grid-cols-2 gap-2 border-t border-white/10 pt-3">
                    <span className="text-zinc-500">Depth</span>
                    <span>{mlStatus.depth_before?.depth ?? "cv"} / {mlStatus.depth_before?.depth_status ?? "fallback"}</span>
                    <span className="text-zinc-500">SAM</span>
                    <span>{mlStatus.segmentation?.provider ?? "cv"} / {mlStatus.segmentation?.status ?? "fallback"}</span>
                    <span className="text-zinc-500">CLIP</span>
                    <span>{mlUnderstanding?.clip?.provider ?? "rules"} / {mlUnderstanding?.clip?.status ?? "fallback"}</span>
                    <span className="text-zinc-500">Classifier</span>
                    <span>{mlUnderstanding?.classifier?.provider ?? "rules"} / {mlUnderstanding?.classifier?.status ?? "fallback"}</span>
                  </div>
                )}
              </div>
            )}

            <div className="mt-4 rounded-lg border border-white/10 bg-black/25 p-4 text-sm text-zinc-300">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div className="text-zinc-100">История результатов</div>
                <button type="button" onClick={loadHistory} className="text-xs text-cyan-200 hover:text-cyan-100">
                  {isHistoryLoading ? "Обновление..." : "Обновить"}
                </button>
              </div>
              <div className="grid gap-2">
                {historyItems.length === 0 && <div className="text-zinc-500">Пока нет сохраненных запусков</div>}
                {historyItems.slice(0, 6).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => openHistoryItem(item.id)}
                    className="grid grid-cols-[56px_minmax(0,1fr)_auto] items-center gap-3 rounded-lg border border-white/10 bg-white/[0.04] p-2 text-left transition hover:border-cyan-300/40 hover:bg-cyan-300/10"
                  >
                    {item.result_thumb ? (
                      <img src={item.result_thumb} alt="" className="h-12 w-14 rounded object-cover" />
                    ) : (
                      <div className="h-12 w-14 rounded bg-white/10" />
                    )}
                    <span className="min-w-0">
                      <span className="block truncate text-zinc-100">{item.problem || "analysis"} / {item.strength || "auto"}</span>
                      <span className="block truncate text-xs text-zinc-500">{item.modes?.join(", ") || "modes"}</span>
                    </span>
                    <span className="rounded border border-amber-300/25 bg-amber-300/10 px-2 py-1 text-xs text-amber-100">
                      {item.total_score ?? "--"}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}
