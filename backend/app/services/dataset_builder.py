from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AI-Light-Pro-Diploma/1.0 (local research dataset builder)"

SEARCH_QUERIES = (
    "interior sunlight shadow",
    "plant shadow wall",
    "room window sunlight",
    "product photography shadows",
    "wet asphalt reflection",
    "water reflection surface",
    "glass reflection interior",
    "mirror reflection room",
    "gradient sky sunset",
    "studio background gradient",
    "smooth wall light",
    "modern interior table",
)

PROMPTS = {
    "gradient": (
        "remove gradient banding and smooth color transition",
        "clean posterized background gradient",
        "soft linear gradient without white patches",
        "исправить полосы на градиенте и сохранить фон",
    ),
    "shadow": (
        "clean realistic cast shadows preserve texture",
        "soft natural shadow correction",
        "remove dirty shadow edges",
        "исправить грязные тени точечно",
    ),
    "reflection": (
        "natural reflection on glass surface",
        "repair water reflection with realistic highlights",
        "clean mirror reflection without blur artifacts",
        "исправить отражение точечно",
    ),
}

MATERIALS = {
    "gradient": ("glass", "mirror"),
    "shadow": ("glass", "asphalt"),
    "reflection": ("water", "asphalt", "mirror", "glass"),
}

STRENGTHS = ("low", "medium", "high")


@dataclass
class SourceImage:
    title: str
    file_name: str
    source_url: str
    image_url: str
    license_name: str
    license_url: str
    artist: str


@dataclass
class TrainingExample:
    image_path: str
    source_file: str
    problem: str
    material: str
    strength: str
    prompt: str
    label: str
    synthetic: bool = True


class DatasetBuilder:
    def __init__(
        self,
        output_dir: str = "backend/data/training",
        target_examples: int = 150,
        max_sources: int = 60,
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.source_dir = self.output_dir / "source"
        self.image_dir = self.output_dir / "images"
        self.target_examples = target_examples
        self.max_sources = max_sources
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def build(self) -> dict[str, Any]:
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        sources = self._collect_sources()
        downloaded = self._download_sources(sources)
        examples = self._create_examples(downloaded)
        self._write_jsonl(self.output_dir / "manifest.jsonl", [asdict(example) for example in examples])
        self._write_json(self.output_dir / "summary.json", {
            "source": "Wikimedia Commons API + synthetic labelled artifacts",
            "raw_images": len(downloaded),
            "training_examples": len(examples),
            "problems": sorted({example.problem for example in examples}),
            "materials": sorted({example.material for example in examples}),
            "target_examples": self.target_examples,
        })
        return {
            "raw_images": len(downloaded),
            "training_examples": len(examples),
            "manifest": str(self.output_dir / "manifest.jsonl"),
            "summary": str(self.output_dir / "summary.json"),
        }

    def _collect_sources(self) -> list[SourceImage]:
        collected: dict[str, SourceImage] = {}
        for query in SEARCH_QUERIES:
            if len(collected) >= self.max_sources:
                break
            for item in self._search_commons(query):
                collected.setdefault(item.title, item)
                if len(collected) >= self.max_sources:
                    break
            time.sleep(0.2)
        sources = list(collected.values())[: self.max_sources]
        self._write_jsonl(self.output_dir / "source_metadata.jsonl", [asdict(source) for source in sources])
        return sources

    def _search_commons(self, query: str) -> list[SourceImage]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrlimit": 24,
            "gsrsearch": query,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": 720,
        }
        try:
            response = requests.get(COMMONS_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
        except Exception:
            return []

        results: list[SourceImage] = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            mime = str(info.get("mime", ""))
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            width = int(info.get("width", 0) or 0)
            height = int(info.get("height", 0) or 0)
            if width < 420 or height < 280:
                continue
            ext = info.get("extmetadata") or {}
            license_name = _clean_html(ext.get("LicenseShortName", {}).get("value", "unknown"))
            license_url = _clean_html(ext.get("LicenseUrl", {}).get("value", ""))
            artist = _clean_html(ext.get("Artist", {}).get("value", "unknown"))
            title = str(page.get("title", "File:unknown"))
            image_url = str(info.get("thumburl") or info.get("url") or "")
            source_url = str(info.get("descriptionurl") or "")
            if not image_url:
                continue
            results.append(
                SourceImage(
                    title=title,
                    file_name=_safe_name(title),
                    source_url=source_url,
                    image_url=image_url,
                    license_name=license_name,
                    license_url=license_url,
                    artist=artist,
                )
            )
        return results

    def _download_sources(self, sources: list[SourceImage]) -> list[Path]:
        downloaded: list[Path] = []
        for index, source in enumerate(sources, start=1):
            suffix = ".jpg"
            if source.image_url.lower().split("?")[0].endswith(".png"):
                suffix = ".png"
            target = self.source_dir / f"{index:03d}_{source.file_name}{suffix}"
            if not target.exists():
                try:
                    response = self._download_with_retry(source.image_url)
                    target.write_bytes(response.content)
                except Exception:
                    continue
            if self._read_image(target) is not None:
                downloaded.append(target)
        return downloaded

    def _download_with_retry(self, url: str) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
                if response.status_code == 429:
                    time.sleep(6.0 + attempt * 4.0)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "image/" not in content_type:
                    raise RuntimeError(f"Unexpected content type: {content_type}")
                time.sleep(1.15)
                return response
            except Exception as exc:
                last_error = exc
                time.sleep(2.0 + attempt * 2.0)
        raise RuntimeError(f"Could not download image: {last_error}")

    def _create_examples(self, source_paths: list[Path]) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        if not source_paths:
            return examples

        problem_cycle = ["gradient", "shadow", "reflection"]
        index = 0
        while len(examples) < self.target_examples:
            source_path = source_paths[index % len(source_paths)]
            image = self._read_image(source_path)
            if image is None:
                index += 1
                continue

            problem = problem_cycle[index % len(problem_cycle)]
            material = self.rng.choice(MATERIALS[problem])
            strength = self.rng.choice(STRENGTHS)
            prompt = self.rng.choice(PROMPTS[problem])

            if problem == "gradient":
                synthetic = self._add_gradient_problem(image, strength)
            elif problem == "shadow":
                synthetic = self._add_shadow_problem(image, strength)
            else:
                synthetic = self._add_reflection_problem(image, material, strength)

            file_name = f"{len(examples) + 1:04d}_{problem}_{material}_{strength}.jpg"
            output_path = self.image_dir / file_name
            cv2.imwrite(str(output_path), cv2.cvtColor(synthetic, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            examples.append(
                TrainingExample(
                    image_path=str(output_path),
                    source_file=str(source_path),
                    problem=problem,
                    material=material,
                    strength=strength,
                    prompt=prompt,
                    label=f"{problem}|{material}|{strength}",
                )
            )
            index += 1
        return examples

    def _read_image(self, path: Path) -> np.ndarray | None:
        try:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.uint8)
            if min(array.shape[:2]) < 220:
                return None
            return array
        except Exception:
            return None

    def _add_gradient_problem(self, image: np.ndarray, strength: str) -> np.ndarray:
        result = image.astype(np.float32).copy()
        height, width = result.shape[:2]
        horizontal = self.rng.random() > 0.5
        axis = np.linspace(0.0, 1.0, width if horizontal else height, dtype=np.float32)
        if strength == "low":
            steps, opacity = 18, 0.18
        elif strength == "medium":
            steps, opacity = 12, 0.28
        else:
            steps, opacity = 8, 0.38
        banded = np.floor(axis * steps) / max(1, steps - 1)
        if horizontal:
            banded = np.broadcast_to(banded[None, :], (height, width))
        else:
            banded = np.broadcast_to(banded[:, None], (height, width))
        color_a = np.array(self.rng.choice([(236, 245, 255), (255, 232, 214), (232, 245, 233)]), dtype=np.float32)
        color_b = np.array(self.rng.choice([(130, 160, 230), (250, 180, 150), (180, 210, 185)]), dtype=np.float32)
        gradient = color_a * (1.0 - banded[..., None]) + color_b * banded[..., None]
        smooth_mask = self._smooth_region_mask(result)
        result = result * (1.0 - smooth_mask[..., None] * opacity) + gradient * (smooth_mask[..., None] * opacity)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_shadow_problem(self, image: np.ndarray, strength: str) -> np.ndarray:
        result = image.astype(np.float32).copy()
        height, width = result.shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)
        count = {"low": 1, "medium": 2, "high": 3}[strength]
        for _ in range(count):
            center = (self.rng.randint(width // 5, width * 4 // 5), self.rng.randint(height // 5, height * 4 // 5))
            axes = (self.rng.randint(width // 8, width // 3), self.rng.randint(height // 12, height // 4))
            angle = self.rng.randint(-35, 35)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX={"low": 8, "medium": 14, "high": 22}[strength])
        noise = self.np_rng.normal(0.0, 0.08, mask.shape).astype(np.float32)
        dirty = np.clip(mask + cv2.GaussianBlur(noise, (0, 0), sigmaX=4), 0.0, 1.0)
        opacity = {"low": 0.22, "medium": 0.34, "high": 0.48}[strength]
        result *= 1.0 - dirty[..., None] * opacity
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_reflection_problem(self, image: np.ndarray, material: str, strength: str) -> np.ndarray:
        result = image.astype(np.float32).copy()
        height, width = result.shape[:2]
        start_y = int(height * self.rng.uniform(0.45, 0.62))
        lower_h = height - start_y
        if lower_h <= 16:
            return image
        upper = image[max(0, start_y - lower_h):start_y]
        if upper.shape[0] != lower_h:
            upper = cv2.resize(upper, (width, lower_h), interpolation=cv2.INTER_LINEAR)
        reflected = np.flipud(upper).astype(np.float32)
        ripple = {"mirror": 0.4, "glass": 1.2, "water": 5.5, "asphalt": 2.2}.get(material, 1.0)
        reflected = self._warp_reflection(reflected, ripple)
        if material != "mirror":
            reflected = cv2.GaussianBlur(reflected, (0, 0), sigmaX={"low": 0.7, "medium": 1.5, "high": 2.4}[strength])
        opacity = {"low": 0.18, "medium": 0.30, "high": 0.44}[strength]
        fade = np.linspace(0.85, 0.10, lower_h, dtype=np.float32)[:, None, None]
        surface = result[start_y:].copy()
        target = surface * (1.0 - opacity) + reflected * fade * opacity
        if material == "asphalt":
            target *= 0.84
        result[start_y:] = target
        return np.clip(result, 0, 255).astype(np.uint8)

    def _smooth_region_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        edges = cv2.Canny(gray.astype(np.uint8), 60, 150).astype(np.float32) / 255.0
        texture = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), sigmaX=3)
        texture = texture / max(float(texture.max()), 1.0)
        mask = np.clip(1.0 - edges * 0.8 - texture * 1.3, 0.0, 1.0)
        return cv2.GaussianBlur(mask, (0, 0), sigmaX=7)

    def _warp_reflection(self, reflected: np.ndarray, ripple: float) -> np.ndarray:
        if ripple <= 0.2:
            return reflected
        height, width = reflected.shape[:2]
        y, x = np.indices((height, width), dtype=np.float32)
        x_offset = np.sin(y / 13.0) * ripple + np.sin((x + y) / 41.0) * ripple * 0.6
        y_offset = np.sin(x / 31.0) * ripple * 0.25
        map_x = np.clip(x + x_offset, 0, width - 1).astype(np.float32)
        map_y = np.clip(y + y_offset, 0, height - 1).astype(np.float32)
        return cv2.remap(reflected, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_name(value: str) -> str:
    value = value.replace("File:", "")
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value[:80].strip("_") or "image"


def _clean_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="backend/data/training")
    parser.add_argument("--target-examples", type=int, default=150)
    parser.add_argument("--max-sources", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    builder = DatasetBuilder(
        output_dir=args.output_dir,
        target_examples=args.target_examples,
        max_sources=args.max_sources,
        seed=args.seed,
    )
    print(json.dumps(builder.build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
