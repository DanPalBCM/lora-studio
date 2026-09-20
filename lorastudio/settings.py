"""Workspace paths and persisted user settings."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

APP_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = APP_ROOT / "workspace"
DATASETS_DIR = WORKSPACE / "datasets"
CONFIGS_DIR = WORKSPACE / "configs"
TRAIN_OUT_DIR = WORKSPACE / "training_output"
IMAGES_DIR = WORKSPACE / "generated"
LOGS_DIR = WORKSPACE / "logs"
SETTINGS_FILE = WORKSPACE / "settings.json"

ALL_DIRS = [WORKSPACE, DATASETS_DIR, CONFIGS_DIR, TRAIN_OUT_DIR, IMAGES_DIR, LOGS_DIR]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def default_toolkit_python(toolkit_dir: str) -> str:
    """Best guess at the python interpreter inside an ai-toolkit checkout."""
    if not toolkit_dir:
        return sys.executable
    base = Path(toolkit_dir)
    candidates = [
        base / "venv" / "Scripts" / "python.exe",
        base / "venv" / "bin" / "python",
        base / ".venv" / "Scripts" / "python.exe",
        base / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


@dataclass
class Settings:
    # --- environment -------------------------------------------------
    toolkit_dir: str = ""          # path to an ostris/ai-toolkit checkout
    toolkit_python: str = ""       # python.exe inside the toolkit venv
    hf_token: str = ""             # optional, for gated repos

    # --- project -----------------------------------------------------
    project_name: str = "my_first_lora"
    preset_key: str = "zimage_turbo_10gb"
    trigger_word: str = "ohwx person"
    base_caption: str = ""         # appended after the trigger word

    # --- dataset -----------------------------------------------------
    max_image_side: int = 768
    last_image_dir: str = ""

    # --- training ----------------------------------------------------
    steps: int = 2000
    rank: int = 16
    learning_rate: float = 1e-4
    save_every: int = 250
    sample_every: int = 250
    resolutions: List[int] = field(default_factory=lambda: [512, 640])
    layer_offload_percent: int = 0   # 0 = off. Raise to 40-60 if you OOM.

    # --- generation --------------------------------------------------
    gen_prompt: str = "ohwx person, candid photo, natural window light, 85mm lens, shallow depth of field"
    gen_negative: str = "blurry, deformed hands, extra fingers, watermark, text"
    gen_steps: int = 9
    gen_guidance: float = 3.5
    gen_width: int = 768
    gen_height: int = 1024
    gen_seed: int = -1
    lora_weight: float = 1.0
    lora_path: str = ""
    output_dir: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self) -> None:
        ensure_dirs()
        SETTINGS_FILE.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "Settings":
        ensure_dirs()
        s = cls()
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            known = {f for f in cls().to_json()}
            for k, v in data.items():
                if k in known:
                    setattr(s, k, v)
        if not s.output_dir:
            s.output_dir = str(IMAGES_DIR)
        if not s.toolkit_python:
            s.toolkit_python = default_toolkit_python(s.toolkit_dir)
        return s

    # helpers ----------------------------------------------------------
    @property
    def dataset_dir(self) -> Path:
        return DATASETS_DIR / self.project_name

    @property
    def config_path(self) -> Path:
        return CONFIGS_DIR / f"{self.project_name}.yaml"

    def env_for_subprocess(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
        return env
