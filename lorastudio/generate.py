"""Image generation with diffusers, with the trained LoRA applied.

The pipeline is cached between generations so that the second image is fast.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .settings import Settings
from . import presets


@dataclass
class GenResult:
    image_path: Path
    seed: int
    seconds: float


class Generator:
    """Holds one loaded pipeline. Reloads only when the model or LoRA changes."""

    def __init__(self, log: Callable[[str], None] = print) -> None:
        self.log = log
        self._pipe = None
        self._loaded_key: Optional[str] = None
        self._loaded_lora: Optional[str] = None

    # -- loading --------------------------------------------------------
    def _build_pipeline(self, preset_key: str):
        import torch
        from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

        p = presets.get(preset_key)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.log(f"Loading base model {p.base_model} (first run downloads several GB)...")

        if p.diffusers_pipeline == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                p.base_model, torch_dtype=torch.float16, use_safetensors=True,
                variant="fp16",
            )
        else:
            # Newer architectures land in diffusers under their own class names.
            # Let diffusers resolve it from the repo's model_index.json, and
            # allow remote code for very new repos that ship their own pipeline.
            try:
                pipe = DiffusionPipeline.from_pretrained(p.base_model, torch_dtype=dtype)
            except Exception as exc:
                self.log(f"Standard load failed ({exc}); retrying with trust_remote_code=True")
                pipe = DiffusionPipeline.from_pretrained(
                    p.base_model, torch_dtype=dtype, trust_remote_code=True
                )

        # 10GB survival kit: stream components to the GPU only while in use.
        try:
            pipe.enable_model_cpu_offload()
            self.log("Model CPU offload enabled (keeps peak VRAM low).")
        except (AttributeError, ValueError):
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        for attr, fn in (("vae", "enable_slicing"), ("vae", "enable_tiling")):
            obj = getattr(pipe, attr, None)
            if obj is not None and hasattr(obj, fn):
                try:
                    getattr(obj, fn)()
                except Exception:
                    pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except AttributeError:
            pass
        return pipe

    def ensure_loaded(self, s: Settings) -> None:
        lora = (s.lora_path or "").strip()
        if self._pipe is not None and self._loaded_key == s.preset_key and self._loaded_lora == lora:
            return

        if self._pipe is None or self._loaded_key != s.preset_key:
            self._pipe = self._build_pipeline(s.preset_key)
            self._loaded_key = s.preset_key
            self._loaded_lora = None

        # swap LoRA
        if self._loaded_lora:
            for fn in ("unfuse_lora", "unload_lora_weights"):
                try:
                    getattr(self._pipe, fn)()
                except Exception:
                    pass
            self._loaded_lora = None

        if lora:
            path = Path(lora)
            if not path.exists():
                raise FileNotFoundError(f"LoRA file not found: {lora}")
            self.log(f"Applying LoRA {path.name} at weight {s.lora_weight}")
            self._pipe.load_lora_weights(str(path.parent), weight_name=path.name,
                                         adapter_name="custom")
            try:
                self._pipe.set_adapters(["custom"], [float(s.lora_weight)])
            except (AttributeError, ValueError):
                self._pipe.fuse_lora(lora_scale=float(s.lora_weight))
            self._loaded_lora = lora
        else:
            self.log("No LoRA selected - generating from the base model only.")

    # -- generation ------------------------------------------------------
    def generate(self, s: Settings) -> GenResult:
        import torch

        self.ensure_loaded(s)
        p = presets.get(s.preset_key)

        seed = s.gen_seed if s.gen_seed and s.gen_seed >= 0 else random.randint(0, 2**31 - 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

        kwargs = dict(
            prompt=s.gen_prompt,
            num_inference_steps=int(s.gen_steps),
            guidance_scale=float(s.gen_guidance),
            width=int(s.gen_width),
            height=int(s.gen_height),
            generator=generator,
        )
        # Not every modern pipeline accepts a negative prompt.
        if s.gen_negative.strip():
            kwargs["negative_prompt"] = s.gen_negative

        self.log(f"Generating {s.gen_width}x{s.gen_height}, {s.gen_steps} steps, seed {seed}...")
        t0 = time.time()
        try:
            out = self._pipe(**kwargs)
        except TypeError as exc:
            if "negative_prompt" in str(exc):
                kwargs.pop("negative_prompt", None)
                out = self._pipe(**kwargs)
            else:
                raise
        elapsed = time.time() - t0
        image = out.images[0]

        out_dir = Path(s.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        stem = f"{s.project_name}_{stamp}_{seed}"
        img_path = out_dir / f"{stem}.png"

        meta = {
            "prompt": s.gen_prompt,
            "negative_prompt": s.gen_negative,
            "seed": seed,
            "steps": s.gen_steps,
            "guidance": s.gen_guidance,
            "size": [s.gen_width, s.gen_height],
            "base_model": p.base_model,
            "lora": s.lora_path,
            "lora_weight": s.lora_weight,
            "generated": stamp,
        }
        try:
            from PIL import PngImagePlugin

            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", json.dumps(meta))
            image.save(img_path, pnginfo=info)
        except Exception:
            image.save(img_path)
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self.log(f"Saved {img_path} in {elapsed:.1f}s")
        return GenResult(image_path=img_path, seed=seed, seconds=elapsed)

    def unload(self) -> None:
        self._pipe = None
        self._loaded_key = None
        self._loaded_lora = None
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
