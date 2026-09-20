"""Turn a pile of selected images into an ai-toolkit dataset folder.

ai-toolkit expects a flat folder of images with an optional matching .txt
caption per image:

    datasets/my_first_lora/
        img_001.jpg
        img_001.txt
        img_002.jpg
        img_002.txt
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List

from PIL import Image, ImageOps

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MIN_USEFUL_SIDE = 512


@dataclass
class PrepResult:
    written: int = 0
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dataset_dir: Path = Path(".")


def build_caption(trigger: str, base: str) -> str:
    trigger = (trigger or "").strip()
    base = (base or "").strip()
    if trigger and base:
        return f"{trigger}, {base}"
    return trigger or base


def prepare(
    image_paths: Iterable[str],
    dataset_dir: Path,
    trigger_word: str,
    base_caption: str = "",
    max_side: int = 768,
    clean: bool = True,
    log: Callable[[str], None] = print,
) -> PrepResult:
    """Copy + normalise images into `dataset_dir` and write caption sidecars."""
    dataset_dir = Path(dataset_dir)
    if clean and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    result = PrepResult(dataset_dir=dataset_dir)
    caption = build_caption(trigger_word, base_caption)
    if not caption:
        result.warnings.append(
            "No trigger word set. Without one you have no reliable way to call "
            "the concept at inference time."
        )

    index = 0
    for raw in image_paths:
        src = Path(raw)
        if src.suffix.lower() not in SUPPORTED:
            result.skipped.append(f"{src.name} (unsupported type)")
            continue
        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                w, h = im.size
                if min(w, h) < MIN_USEFUL_SIDE:
                    result.warnings.append(
                        f"{src.name} is only {w}x{h}. Small inputs teach the model "
                        f"blur. Consider dropping it."
                    )
                longest = max(w, h)
                if longest > max_side:
                    scale = max_side / float(longest)
                    im = im.resize(
                        (max(1, round(w * scale)), max(1, round(h * scale))),
                        Image.LANCZOS,
                    )
                index += 1
                stem = f"img_{index:03d}"
                im.save(dataset_dir / f"{stem}.jpg", quality=95, subsampling=0)
        except Exception as exc:  # pillow raises a zoo of exceptions
            result.skipped.append(f"{src.name} ({exc})")
            continue

        # a per-image caption file; edit these by hand later if you want
        (dataset_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
        result.written += 1
        log(f"  + {src.name} -> {stem}.jpg")

    if result.written and result.written < 8:
        result.warnings.append(
            f"Only {result.written} images. That can work for a tight identity "
            f"LoRA, but 20-40 varied shots give a far more flexible result."
        )
    if result.written > 80:
        result.warnings.append(
            f"{result.written} images is a lot for an identity LoRA. Quality beats "
            f"quantity here; consider curating down to your best 40."
        )
    return result


def existing_count(dataset_dir: Path) -> int:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return 0
    return sum(1 for p in dataset_dir.iterdir() if p.suffix.lower() in SUPPORTED)
