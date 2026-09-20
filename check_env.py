#!/usr/bin/env python3
"""Sanity-check the environment before you waste an evening on a failed run.

    python check_env.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OK = "[ ok ]"
WARN = "[warn]"
BAD = "[fail]"


def line(tag: str, text: str) -> None:
    print(f"{tag} {text}")


def main() -> None:
    print("\nLoRA Studio environment check\n" + "-" * 40)

    line(OK if sys.version_info >= (3, 10) else BAD, f"Python {sys.version.split()[0]}")

    try:
        import tkinter  # noqa: F401
        line(OK, "tkinter available")
    except ImportError:
        line(BAD, "tkinter missing (Ubuntu: sudo apt install python3-tk)")

    try:
        import PIL
        line(OK, f"Pillow {PIL.__version__}")
    except ImportError:
        line(BAD, "Pillow missing -> pip install -r requirements.txt")

    try:
        import yaml  # noqa: F401
        line(OK, "PyYAML available")
    except ImportError:
        line(WARN, "PyYAML missing - configs will be written as JSON (still valid YAML)")

    try:
        import torch
        line(OK, f"torch {torch.__version__}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gb = props.total_memory / (1024 ** 3)
            line(OK, f"CUDA GPU: {props.name}, {gb:.1f} GB")
            if gb < 11:
                line(
                    WARN,
                    "Under 11GB. Use the 10GB profile, 512/640 buckets, rank 8-16, "
                    "and train with the desktop on another GPU if you can.",
                )
        else:
            line(BAD, "torch sees no CUDA device - training will not run")
    except ImportError:
        line(WARN, "torch not installed in THIS interpreter (fine if ai-toolkit has its own venv, "
                   "but you need it here for the Generate tab)")

    try:
        import diffusers
        line(OK, f"diffusers {diffusers.__version__}")
    except ImportError:
        line(WARN, "diffusers missing - the Generate tab will not work until you install it")

    if shutil.which("nvidia-smi"):
        line(OK, "nvidia-smi on PATH")
    else:
        line(WARN, "nvidia-smi not found on PATH")

    print("-" * 40)
    print("Training itself runs inside ai-toolkit's own environment, which this "
          "script does not inspect. Point at it on the Setup tab.\n")


if __name__ == "__main__":
    main()
