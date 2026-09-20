"""Model presets.

Each preset bundles (a) the repo id of the base model, (b) the architecture
string ai-toolkit expects, and (c) the memory-saving knobs appropriate for a
given VRAM tier.

IMPORTANT: the `arch` strings below are the ones ai-toolkit uses in its own
example configs. ai-toolkit moves fast. If training dies immediately with
something like "unknown architecture", open
    <ai-toolkit>/config/examples/
find the example config for your model, and copy its `arch` value into the
matching preset here. That is the single most likely thing you will need to
edit in this whole project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Preset:
    key: str
    label: str
    arch: str
    base_model: str
    # memory knobs
    quantize: bool = True
    quantize_te: bool = True
    low_vram: bool = True
    gradient_checkpointing: bool = True
    optimizer: str = "adamw8bit"
    noise_scheduler: str = "flowmatch"
    dtype: str = "bf16"
    # Z-Image Turbo is step-distilled. Training a LoRA on it directly degrades
    # the 8-step behaviour, so ai-toolkit loads a temporary "de-distillation"
    # adapter during training and drops it at inference.
    training_adapter: Optional[str] = None
    # sampling defaults that suit this base model
    sample_steps: int = 9
    sample_guidance: float = 3.5
    default_resolutions: List[int] = field(default_factory=lambda: [512, 640])
    # rough seconds-per-iteration, used only for the pre-flight time estimate
    est_sec_per_it: float = 5.0
    extra_model: Dict = field(default_factory=dict)
    # inference
    diffusers_pipeline: str = "auto"
    notes: str = ""


PRESETS: Dict[str, Preset] = {
    "zimage_turbo_10gb": Preset(
        key="zimage_turbo_10gb",
        label="Z-Image Turbo 6B  -  10GB profile (RTX 3080 10GB)",
        arch="z-image-turbo",
        base_model="Tongyi-MAI/Z-Image-Turbo",
        training_adapter="ostris/zimage_turbo_training_adapter",
        default_resolutions=[512, 640],
        est_sec_per_it=6.0,
        sample_steps=9,
        sample_guidance=3.5,
        notes=(
            "Recommended for a 10GB card. Quantised transformer + text encoder, "
            "low-VRAM mode, 512/640 buckets. If it fits, try adding 768 to the "
            "resolution list for sharper faces."
        ),
    ),
    "zimage_turbo_12gb": Preset(
        key="zimage_turbo_12gb",
        label="Z-Image Turbo 6B  -  12GB profile (768px buckets)",
        arch="z-image-turbo",
        base_model="Tongyi-MAI/Z-Image-Turbo",
        training_adapter="ostris/zimage_turbo_training_adapter",
        default_resolutions=[512, 768],
        est_sec_per_it=3.0,
        sample_steps=9,
        sample_guidance=3.5,
        notes="The configuration people have reported working on 12GB cards.",
    ),
    "zimage_turbo_24gb": Preset(
        key="zimage_turbo_24gb",
        label="Z-Image Turbo 6B  -  24GB profile (rented 3090/4090)",
        arch="z-image-turbo",
        base_model="Tongyi-MAI/Z-Image-Turbo",
        training_adapter="ostris/zimage_turbo_training_adapter",
        quantize=False,
        quantize_te=False,
        low_vram=False,
        default_resolutions=[512, 768, 1024],
        est_sec_per_it=1.6,
        notes="No quantisation, full 1024px buckets. Use this on rented hardware.",
    ),
    "sdxl_10gb": Preset(
        key="sdxl_10gb",
        label="SDXL 1.0  -  10GB profile (safe fallback)",
        arch="sdxl",
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        quantize=False,
        quantize_te=False,
        low_vram=True,
        noise_scheduler="ddpm",
        dtype="bf16",
        training_adapter=None,
        sample_steps=25,
        sample_guidance=7.0,
        default_resolutions=[1024],
        est_sec_per_it=1.2,
        extra_model={"is_xl": True},
        diffusers_pipeline="sdxl",
        notes=(
            "The known-quantity fallback. Weaker anatomy than Z-Image but a huge "
            "ControlNet / inpainting ecosystem to fix it at inference time."
        ),
    ),
    "flux2_klein_4b_16gb": Preset(
        key="flux2_klein_4b_16gb",
        label="FLUX.2 klein 4B Base  -  16GB+ only (will not fit 10GB)",
        arch="flux2",
        base_model="black-forest-labs/FLUX.2-klein-base-4B",
        quantize=True,
        quantize_te=True,
        low_vram=True,
        default_resolutions=[512, 768],
        est_sec_per_it=8.0,
        sample_steps=28,
        sample_guidance=4.0,
        notes=(
            "Listed for completeness. Black Forest Labs put the LoRA floor at 12GB "
            "and community runs generally want 16-24GB. Expect OOM on a 3080 10GB."
        ),
    ),
}

DEFAULT_PRESET = "zimage_turbo_10gb"


def get(key: str) -> Preset:
    return PRESETS.get(key, PRESETS[DEFAULT_PRESET])


def labels() -> List[str]:
    return [p.label for p in PRESETS.values()]


def key_for_label(label: str) -> str:
    for p in PRESETS.values():
        if p.label == label:
            return p.key
    return DEFAULT_PRESET
