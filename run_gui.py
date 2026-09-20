#!/usr/bin/env python3
"""LoRA Studio - launch the GUI.

    python run_gui.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _fail(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70 + "\n")
    sys.exit(1)


def main() -> None:
    if sys.version_info < (3, 10):
        _fail(f"Python 3.10+ required, you have {sys.version.split()[0]}.")
    try:
        import tkinter  # noqa: F401
    except ImportError:
        _fail(
            "tkinter is missing.\n"
            "  Windows/macOS: reinstall Python from python.org (tkinter is bundled).\n"
            "  Ubuntu/Debian: sudo apt install python3-tk"
        )
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        _fail("Pillow is missing. Run:  pip install -r requirements.txt")

    from lorastudio.gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
