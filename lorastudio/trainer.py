"""Build an ai-toolkit job config, run it, and report live progress + ETA.

We shell out to ai-toolkit's own `run.py` rather than reimplementing training.
That keeps this project small and means you inherit every upstream fix.

Progress comes from parsing ai-toolkit's tqdm output, which looks like:

    my_lora: 26%|##5 | 519/2000 [24:43<1:10:54, 2.87s/it, lr: 2.0e-04 loss: 3.8e-01]

We parse the step counter and compute our own ETA from a rolling window, which
is steadier than tqdm's when the first few iterations include model loading.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from . import presets
from .settings import Settings

try:  # PyYAML is the normal path
    import yaml

    _HAVE_YAML = True
except ImportError:  # pragma: no cover
    _HAVE_YAML = False


PROGRESS_RE = re.compile(r"(?P<cur>\d+)\s*/\s*(?P<total>\d+)\s*\[")
RATE_RE = re.compile(r"(?P<rate>\d+(?:\.\d+)?)\s*(?P<unit>s/it|it/s)")
LOSS_RE = re.compile(r"loss:\s*(?P<loss>[\d.]+(?:[eE][+-]?\d+)?)")
OOM_MARKERS = ("out of memory", "CUDA error: out of memory", "OutOfMemoryError")

# ETA sanity limits. tqdm redraws the same step repeatedly, so a rate computed
# from too small a window is meaningless.
MIN_WINDOW_STEPS = 3
MIN_WINDOW_SECONDS = 5.0
MAX_PLAUSIBLE_SEC_PER_IT = 600.0


def format_duration(seconds: float) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN guard
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


@dataclass
class Progress:
    step: int = 0
    total: int = 0
    eta_seconds: Optional[float] = None
    sec_per_it: Optional[float] = None
    loss: Optional[float] = None
    elapsed: float = 0.0

    @property
    def fraction(self) -> float:
        return (self.step / self.total) if self.total else 0.0

    def eta_text(self) -> str:
        if self.eta_seconds is None:
            return "estimating..."
        finish = time.strftime("%H:%M", time.localtime(time.time() + self.eta_seconds))
        return f"{format_duration(self.eta_seconds)} left  (done around {finish})"


# ----------------------------------------------------------------------------
# config building
# ----------------------------------------------------------------------------

def build_config(s: Settings) -> Dict[str, Any]:
    p = presets.get(s.preset_key)
    resolutions = list(s.resolutions or p.default_resolutions)

    model_block: Dict[str, Any] = {
        "name_or_path": p.base_model,
        "arch": p.arch,
        "quantize": bool(p.quantize),
        "quantize_te": bool(p.quantize_te),
        "low_vram": bool(p.low_vram),
    }
    model_block.update(p.extra_model)
    if p.training_adapter:
        model_block["model_kwargs"] = {"train_adapter": p.training_adapter}
    if s.layer_offload_percent and s.layer_offload_percent > 0:
        model_block["layer_offloading"] = True
        model_block["layer_offloading_transformer_percent"] = round(
            s.layer_offload_percent / 100.0, 2
        )

    sample_prompts = [
        s.gen_prompt
        or f"{s.trigger_word}, photo, natural light",
        f"{s.trigger_word}, close-up portrait, soft daylight, 85mm",
        f"{s.trigger_word}, full body, standing outdoors, overcast day",
    ]

    process = {
        "type": "sd_trainer",
        "training_folder": str(Path(s.config_path).parent.parent / "training_output"),
        "device": "cuda:0",
        "network": {
            "type": "lora",
            "linear": int(s.rank),
            "linear_alpha": int(s.rank),
        },
        "save": {
            "dtype": "float16",
            "save_every": int(s.save_every),
            "max_step_saves_to_keep": 6,
            "push_to_hub": False,
        },
        "datasets": [
            {
                "folder_path": str(s.dataset_dir),
                "caption_ext": "txt",
                "caption_dropout_rate": 0.05,
                "shuffle_tokens": False,
                "cache_latents_to_disk": True,
                "resolution": resolutions,
            }
        ],
        "train": {
            "batch_size": 1,
            "steps": int(s.steps),
            "gradient_accumulation_steps": 1,
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": bool(p.gradient_checkpointing),
            "noise_scheduler": p.noise_scheduler,
            "optimizer": p.optimizer,
            "lr": float(s.learning_rate),
            "dtype": p.dtype,
            "ema_config": {"use_ema": False, "ema_decay": 0.99},
        },
        "model": model_block,
        "sample": {
            "sampler": p.noise_scheduler,
            "sample_every": int(s.sample_every),
            "width": 512,
            "height": 512,
            "prompts": sample_prompts,
            "neg": s.gen_negative,
            "seed": 42,
            "walk_seed": True,
            "guidance_scale": p.sample_guidance,
            "sample_steps": p.sample_steps,
        },
    }

    return {
        "job": "extension",
        "config": {"name": s.project_name, "process": [process]},
        "meta": {"name": s.project_name, "version": "1.0", "made_with": "LoRA Studio"},
    }


def write_config(s: Settings) -> Path:
    cfg = build_config(s)
    path = Path(s.config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAVE_YAML:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    else:
        # YAML is a superset of JSON, so a JSON body in a .yaml file parses fine.
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def preflight(s: Settings, image_count: int) -> Tuple[List[str], str]:
    """Return (blocking_errors, human readable time estimate)."""
    errors: List[str] = []
    p = presets.get(s.preset_key)

    if not s.toolkit_dir:
        errors.append("Set the ai-toolkit folder in the Setup tab.")
    else:
        run_py = Path(s.toolkit_dir) / "run.py"
        if not run_py.exists():
            errors.append(f"No run.py found in {s.toolkit_dir} - is that an ai-toolkit checkout?")
    if not Path(s.toolkit_python).exists():
        errors.append(f"Python interpreter not found: {s.toolkit_python}")
    if image_count == 0:
        errors.append("The dataset is empty. Add images on the Dataset tab first.")
    if not s.trigger_word.strip():
        errors.append("Set a trigger word - you need it to summon the concept later.")

    per_it = p.est_sec_per_it
    if s.layer_offload_percent:
        per_it *= 1 + (s.layer_offload_percent / 100.0) * 2.5
    if max(s.resolutions or p.default_resolutions) >= 1024:
        per_it *= 1.8
    total = per_it * s.steps
    estimate = (
        f"Rough estimate before we start: ~{per_it:.1f} s/step x {s.steps} steps "
        f"= {format_duration(total)}. The live ETA below replaces this within a "
        f"minute of the run starting."
    )
    return errors, estimate


# ----------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------

def _iter_output(stream) -> "object":
    """Yield lines from a binary stream, splitting on both \\n and \\r.

    tqdm redraws with carriage returns, so a plain readline() would block until
    the bar finished.
    """
    buf = bytearray()
    while True:
        chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
        if not chunk:
            break
        buf.extend(chunk)
        while True:
            idx = -1
            for i, byte in enumerate(buf):
                if byte in (10, 13):  # \n \r
                    idx = i
                    break
            if idx < 0:
                break
            line = bytes(buf[:idx]).decode("utf-8", errors="replace")
            del buf[: idx + 1]
            if line.strip():
                yield line
    if buf:
        tail = bytes(buf).decode("utf-8", errors="replace")
        if tail.strip():
            yield tail


class TrainingRun:
    """Runs ai-toolkit in a background thread and pushes events to a callback."""

    def __init__(
        self,
        settings: Settings,
        on_log: Callable[[str], None],
        on_progress: Callable[[Progress], None],
        on_finish: Callable[[bool, str], None],
    ) -> None:
        self.s = settings
        self.on_log = on_log
        self.on_progress = on_progress
        self.on_finish = on_finish
        self.proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._samples: Deque[Tuple[float, int]] = deque(maxlen=60)
        self._start_time = 0.0
        self.saw_oom = False

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        config_path = write_config(self.s)
        self.on_log(f"Config written to {config_path}")
        cmd = [self.s.toolkit_python, "run.py", str(config_path)]
        self.on_log("Launching: " + " ".join(cmd))
        self.on_log(f"Working directory: {self.s.toolkit_dir}")
        self.on_log("-" * 70)
        self._start_time = time.time()
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=self.s.toolkit_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=self.s.env_for_subprocess(),
                bufsize=0,
            )
        except (OSError, ValueError) as exc:
            self.on_finish(False, f"Could not start training: {exc}")
            return
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.on_log("Stopping... (checkpoints already written are kept)")
            try:
                self.proc.terminate()
            except OSError:
                pass

    @property
    def running(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    # -- internals ------------------------------------------------------
    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        log_file = Path(self.s.config_path).parent.parent / "logs" / f"{self.s.project_name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as fh:
            for line in _iter_output(self.proc.stdout):
                fh.write(line + "\n")
                if any(m.lower() in line.lower() for m in OOM_MARKERS):
                    self.saw_oom = True
                prog = self._parse(line)
                if prog is not None:
                    self.on_progress(prog)
                else:
                    self.on_log(line)
        code = self.proc.wait()
        if self.saw_oom:
            self.on_finish(
                False,
                "Ran out of VRAM. Fixes in order of preference: drop 768 from the "
                "resolution list, set rank to 8, then raise Layer offload % to 40-60.",
            )
        elif self._stop.is_set():
            self.on_finish(False, "Stopped by user.")
        elif code == 0:
            elapsed = time.time() - self._start_time
            self.on_finish(True, f"Training finished in {format_duration(elapsed)}.")
        else:
            self.on_finish(False, f"ai-toolkit exited with code {code}. See the log above.")

    def _parse(self, line: str) -> Optional[Progress]:
        m = PROGRESS_RE.search(line)
        if not m:
            return None
        try:
            cur = int(m.group("cur"))
            total = int(m.group("total"))
        except ValueError:
            return None
        if total <= 0 or cur > total:
            return None

        now = time.time()
        self._samples.append((now, cur))
        prog = Progress(step=cur, total=total, elapsed=now - self._start_time)

        per_it: Optional[float] = None

        # 1. rolling window - steadiest, but only once it spans real work.
        #    tqdm redraws the same step many times, so a naive first/last
        #    difference can divide by an almost-zero time span.
        if len(self._samples) >= 2:
            t0, s0 = self._samples[0]
            t1, s1 = self._samples[-1]
            if (s1 - s0) >= MIN_WINDOW_STEPS and (t1 - t0) >= MIN_WINDOW_SECONDS:
                per_it = (t1 - t0) / (s1 - s0)

        # 2. whatever tqdm itself reports on this line
        if per_it is None:
            rm = RATE_RE.search(line)
            if rm:
                rate = float(rm.group("rate"))
                if rate > 0:
                    per_it = rate if rm.group("unit") == "s/it" else 1.0 / rate

        # 3. crude average since launch (includes model load, so pessimistic)
        if per_it is None and cur >= MIN_WINDOW_STEPS and prog.elapsed > 0:
            per_it = prog.elapsed / cur

        if per_it and 0.0 < per_it < MAX_PLAUSIBLE_SEC_PER_IT:
            prog.sec_per_it = per_it
            prog.eta_seconds = per_it * (total - cur)

        lm = LOSS_RE.search(line)
        if lm:
            try:
                prog.loss = float(lm.group("loss"))
            except ValueError:
                pass
        return prog


# ----------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------

def find_checkpoints(settings: Settings) -> List[Path]:
    """All .safetensors produced for this project, newest last."""
    root = Path(settings.config_path).parent.parent / "training_output" / settings.project_name
    if not root.exists():
        return []
    files = [p for p in root.rglob("*.safetensors")]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files
