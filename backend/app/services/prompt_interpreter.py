import re

from app.models.schemas import PromptParameters


class PromptInterpreter:
    color_dictionary: dict[str, tuple[str, tuple[int, int, int]]] = {
        "pink": ("pink", (244, 114, 182)),
        "rose": ("rose", (251, 113, 133)),
        "peach": ("peach", (251, 191, 146)),
        "orange": ("orange", (251, 146, 60)),
        "yellow": ("yellow", (250, 204, 21)),
        "gold": ("gold", (234, 179, 8)),
        "blue": ("blue", (96, 165, 250)),
        "cyan": ("cyan", (34, 211, 238)),
        "teal": ("teal", (45, 212, 191)),
        "green": ("green", (74, 222, 128)),
        "purple": ("purple", (168, 85, 247)),
        "violet": ("violet", (139, 92, 246)),
        "red": ("red", (248, 113, 113)),
        "white": ("white", (245, 245, 245)),
        "black": ("black", (20, 20, 24)),
        "розов": ("pink", (244, 114, 182)),
        "персик": ("peach", (251, 191, 146)),
        "оранж": ("orange", (251, 146, 60)),
        "желт": ("yellow", (250, 204, 21)),
        "жёлт": ("yellow", (250, 204, 21)),
        "золот": ("gold", (234, 179, 8)),
        "син": ("blue", (96, 165, 250)),
        "голуб": ("cyan", (34, 211, 238)),
        "бирюз": ("teal", (45, 212, 191)),
        "зелен": ("green", (74, 222, 128)),
        "фиолет": ("violet", (139, 92, 246)),
        "красн": ("red", (248, 113, 113)),
        "бел": ("white", (245, 245, 245)),
        "черн": ("black", (20, 20, 24)),
        "чёрн": ("black", (20, 20, 24)),
    }

    def parse(self, prompt: str) -> PromptParameters:
        normalized = prompt.lower().replace("ё", "е")
        tokens = [token for token in re.split(r"[^a-zа-яіїєґ0-9]+", normalized) if token]
        compact = re.sub(r"[^a-zа-яіїєґ0-9]+", "", normalized)

        params = PromptParameters(raw_tokens=tokens)
        self._detect_modes(tokens, compact, params)
        self._detect_intensity(tokens, compact, params)
        self._detect_colors(tokens, compact, params)
        self._detect_gradient(tokens, compact, params)
        self._detect_reflection(tokens, compact, params)
        self._detect_shadow(tokens, compact, params)
        return params

    def _contains(self, tokens: list[str], compact: str, words: tuple[str, ...]) -> bool:
        return any(word in tokens or word in compact for word in words)

    def _detect_modes(self, tokens: list[str], compact: str, params: PromptParameters) -> None:
        if self._contains(tokens, compact, ("gradient", "градиент", "градієнт", "banding", "полос", "смуг", "фон")):
            params.mode_hints.append("gradient")
            params.banding_fix = True

        if self._contains(tokens, compact, ("reflection", "reflect", "mirror", "wet", "отраж", "відображ", "блик", "мокр", "вода", "асфальт", "стекл", "зеркал")):
            params.mode_hints.append("reflection")

        if self._contains(tokens, compact, ("shadow", "shadows", "тень", "тени", "тін", "dramatic", "dark", "теньсгенер", "сгенерируйтень")):
            params.mode_hints.append("shadow")

    def _detect_intensity(self, tokens: list[str], compact: str, params: PromptParameters) -> None:
        if self._contains(tokens, compact, ("soft", "smooth", "мягк", "плавн", "gentle", "мягкий")):
            params.softness = 0.82
            params.blur_strength = 0.55
            params.intensity = max(params.intensity, 0.62)

        if self._contains(tokens, compact, ("strong", "intense", "high", "усиль", "сильн", "мощн")):
            params.intensity = 0.82
            params.reflection_strength = 0.72

        if self._contains(tokens, compact, ("clean", "cleaner", "очист", "убери", "remove", "denoise", "noise", "шум", "гряз")):
            params.denoise = True
            params.banding_fix = True

    def _detect_colors(self, tokens: list[str], compact: str, params: PromptParameters) -> None:
        found: list[tuple[str, tuple[int, int, int]]] = []
        for key, color in self.color_dictionary.items():
            if any(key in token for token in tokens) or key in compact:
                if color[0] not in [name for name, _ in found]:
                    found.append(color)

        params.color_names = [name for name, _ in found[:3]]
        params.colors = [value for _, value in found[:3]]
        if len(params.colors) >= 3:
            params.gradient_stops = 3

    def _detect_gradient(self, tokens: list[str], compact: str, params: PromptParameters) -> None:
        if self._contains(tokens, compact, ("horizontal", "left", "right", "горизонт", "слева", "справа")):
            params.direction = "horizontal"
        elif self._contains(tokens, compact, ("diagonal", "диагон", "угол")):
            params.direction = "diagonal"
        elif self._contains(tokens, compact, ("vertical", "top", "bottom", "вертик", "сверху", "снизу")):
            params.direction = "vertical"

        if self._contains(tokens, compact, ("radial", "circle", "радиал", "круг", "центр")):
            params.gradient_style = "radial"
            params.direction = "radial"
        elif self._contains(tokens, compact, ("linear", "линейн", "линия")):
            params.gradient_style = "linear"

        if self._contains(tokens, compact, ("three", "3", "трех", "трехцвет", "три", "трёх", "трёхцвет")):
            params.gradient_stops = 3
        elif self._contains(tokens, compact, ("two", "2", "двух", "двухцвет", "два")):
            params.gradient_stops = 2

    def _detect_reflection(self, tokens: list[str], compact: str, params: PromptParameters) -> None:
        material_checks = (
            ("water", ("water", "вода", "водн", "море", "луж", "ripple")),
            ("asphalt", ("asphalt", "асфальт", "дорог", "мокрыйасфальт")),
            ("mirror", ("mirror", "зеркал", "mirrorlike", "четк", "чётк")),
            ("glass", ("glass", "стекл", "витрин", "глянц")),
        )
        for material, aliases in material_checks:
            if self._contains(tokens, compact, aliases):
                params.reflection_material = material
                break

        if self._contains(tokens, compact, ("wet", "мокр", "луж", "water", "вода")):
            params.reflection_strength = max(params.reflection_strength, 0.78)
            params.blur_strength = max(params.blur_strength, 0.34)
        if self._contains(tokens, compact, ("sharp", "четк", "чётк", "mirror", "зеркал")):
            params.blur_strength = min(params.blur_strength, 0.22)
            params.reflection_strength = max(params.reflection_strength, 0.74)

    def _detect_shadow(self, tokens: list[str], compact: str, params: PromptParameters) -> None:
        if self._contains(tokens, compact, ("dramatic", "cinematic", "драмат", "контраст", "глубок")):
            params.shadow_goal = "dramatic"
            params.contrast_boost = 1.28
            params.intensity = max(params.intensity, 0.76)
        elif self._contains(tokens, compact, ("softshadow", "softshadows", "мягкиетени", "мягкаятень")):
            params.shadow_goal = "soft"
            params.softness = max(params.softness, 0.84)
        elif self._contains(tokens, compact, ("clean", "cleaner", "очист", "чист")):
            params.shadow_goal = "clean"

        if self._contains(tokens, compact, ("generate", "create", "cast", "сгенер", "создай", "добавьтень", "реалистичн")):
            params.shadow_generate = True

        if self._contains(tokens, compact, ("left", "слева", "влево")):
            params.light_direction = "left"
        elif self._contains(tokens, compact, ("right", "справа", "вправо")):
            params.light_direction = "right"
        elif self._contains(tokens, compact, ("top", "сверху")):
            params.light_direction = "top"
