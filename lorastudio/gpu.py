"""GPU detection. Works with torch if installed, falls back to nvidia-smi."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GpuInfo:
    name: str = "unknown"
    total_mb: int = 0
    free_mb: int = 0
    source: str = "none"

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024.0

    @property
    def free_gb(self) -> float:
        return self.free_mb / 1024.0

    def summary(self) -> str:
        if self.source == "none":
            return "No NVIDIA GPU detected (or nvidia-smi is not on PATH)."
        if self.free_mb:
            return (
                f"{self.name}  |  {self.total_gb:.1f} GB total, "
                f"{self.free_gb:.1f} GB free  [{self.source}]"
            )
        return f"{self.name}  |  {self.total_gb:.1f} GB total  [{self.source}]"


def _from_smi() -> Optional[GpuInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    try:
        return GpuInfo(parts[0], int(float(parts[1])), int(float(parts[2])), "nvidia-smi")
    except (IndexError, ValueError):
        return None


def _from_torch() -> Optional[GpuInfo]:
    try:
        import torch  # noqa: WPS433
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        return GpuInfo(props.name, total // (1024 * 1024), free // (1024 * 1024), "torch")
    except Exception:  # pragma: no cover - driver quirks
        return None


def detect() -> GpuInfo:
    return _from_torch() or _from_smi() or GpuInfo()


def recommend_preset(info: GpuInfo) -> str:
    gb = info.total_gb
    if gb >= 22:
        return "zimage_turbo_24gb"
    if gb >= 11.5:
        return "zimage_turbo_12gb"
    if gb >= 9:
        return "zimage_turbo_10gb"
    return "sdxl_10gb"
