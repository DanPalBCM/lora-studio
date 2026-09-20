"""Tkinter front-end. Stdlib only, so there is no extra GUI dependency."""

from __future__ import annotations

import queue
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    Tk,
    X,
    Y,
    StringVar,
    DoubleVar,
    IntVar,
    filedialog,
    messagebox,
)
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import List, Optional

from . import dataset as ds
from . import gpu, presets, trainer
from .generate import Generator
from .settings import IMAGES_DIR, Settings, default_toolkit_python, ensure_dirs

PAD = 8


class App:
    def __init__(self, root: Tk) -> None:
        ensure_dirs()
        self.root = root
        self.s = Settings.load()
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.selected_images: List[str] = []
        self.run: Optional[trainer.TrainingRun] = None
        self.generator = Generator(log=lambda m: self.events.put(("gen_log", m)))
        self._preview_ref = None  # keep a reference or Tk garbage-collects it

        root.title("LoRA Studio")
        root.geometry("980x760")
        root.minsize(860, 640)

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill=BOTH, expand=True, padx=PAD, pady=PAD)
        self._build_setup_tab()
        self._build_dataset_tab()
        self._build_train_tab()
        self._build_generate_tab()

        self.status = StringVar(value="Ready.")
        ttk.Label(root, textvariable=self.status, relief="sunken", anchor="w").pack(
            fill=X, side="bottom"
        )

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._pump)
        self._refresh_gpu()

    # ------------------------------------------------------------------
    # tab 1: setup
    # ------------------------------------------------------------------
    def _build_setup_tab(self) -> None:
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="1. Setup")

        box = ttk.LabelFrame(f, text="Hardware")
        box.pack(fill=X, padx=PAD, pady=PAD)
        self.gpu_var = StringVar(value="detecting...")
        ttk.Label(box, textvariable=self.gpu_var).pack(anchor="w", padx=PAD, pady=(PAD, 2))
        self.gpu_hint = StringVar(value="")
        ttk.Label(box, textvariable=self.gpu_hint, foreground="#555", wraplength=880).pack(
            anchor="w", padx=PAD, pady=(0, PAD)
        )
        ttk.Button(box, text="Re-detect GPU", command=self._refresh_gpu).pack(
            anchor="w", padx=PAD, pady=(0, PAD)
        )

        box2 = ttk.LabelFrame(f, text="ai-toolkit (the training engine)")
        box2.pack(fill=X, padx=PAD, pady=PAD)
        ttk.Label(
            box2,
            text=(
                "LoRA Studio does not reimplement training; it drives ostris/ai-toolkit.\n"
                "Clone it, install its requirements, then point at the folder below."
            ),
            foreground="#555",
        ).pack(anchor="w", padx=PAD, pady=(PAD, 4))

        self.toolkit_var = StringVar(value=self.s.toolkit_dir)
        self.toolkit_py_var = StringVar(value=self.s.toolkit_python)
        self._path_row(box2, "ai-toolkit folder", self.toolkit_var, self._pick_toolkit)
        self._path_row(box2, "Python executable", self.toolkit_py_var, self._pick_python)
        ttk.Button(
            box2, text="Open ai-toolkit on GitHub",
            command=lambda: webbrowser.open("https://github.com/ostris/ai-toolkit"),
        ).pack(anchor="w", padx=PAD, pady=(0, PAD))

        box3 = ttk.LabelFrame(f, text="Project")
        box3.pack(fill=X, padx=PAD, pady=PAD)
        self.project_var = StringVar(value=self.s.project_name)
        self._entry_row(box3, "Project name", self.project_var, width=32)

        self.preset_var = StringVar(value=presets.get(self.s.preset_key).label)
        row = ttk.Frame(box3)
        row.pack(fill=X, padx=PAD, pady=4)
        ttk.Label(row, text="Base model / profile", width=22).pack(side=LEFT)
        combo = ttk.Combobox(
            row, textvariable=self.preset_var, values=presets.labels(),
            state="readonly", width=60,
        )
        combo.pack(side=LEFT, fill=X, expand=True)
        combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        self.preset_note = StringVar(value=presets.get(self.s.preset_key).notes)
        ttk.Label(box3, textvariable=self.preset_note, foreground="#555", wraplength=880).pack(
            anchor="w", padx=PAD, pady=(0, PAD)
        )

        self.token_var = StringVar(value=self.s.hf_token)
        self._entry_row(box3, "HF token (optional)", self.token_var, width=48, show="*")

        ttk.Button(f, text="Save settings", command=self._save_settings).pack(
            anchor="w", padx=PAD, pady=PAD
        )

    # ------------------------------------------------------------------
    # tab 2: dataset
    # ------------------------------------------------------------------
    def _build_dataset_tab(self) -> None:
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="2. Dataset")

        top = ttk.Frame(f)
        top.pack(fill=X, padx=PAD, pady=PAD)
        ttk.Button(top, text="Add images...", command=self._pick_images).pack(side=LEFT)
        ttk.Button(top, text="Add a whole folder...", command=self._pick_folder).pack(
            side=LEFT, padx=4
        )
        ttk.Button(top, text="Remove selected", command=self._remove_selected).pack(
            side=LEFT, padx=4
        )
        ttk.Button(top, text="Clear list", command=self._clear_images).pack(side=LEFT)

        self.listbox_count = StringVar(value="0 images selected")
        ttk.Label(top, textvariable=self.listbox_count).pack(side=RIGHT)

        listframe = ttk.Frame(f)
        listframe.pack(fill=BOTH, expand=True, padx=PAD)
        sb = ttk.Scrollbar(listframe)
        sb.pack(side=RIGHT, fill=Y)
        from tkinter import Listbox, EXTENDED

        self.listbox = Listbox(listframe, selectmode=EXTENDED, yscrollcommand=sb.set)
        self.listbox.pack(fill=BOTH, expand=True)
        sb.config(command=self.listbox.yview)

        opts = ttk.LabelFrame(f, text="Captions and sizing")
        opts.pack(fill=X, padx=PAD, pady=PAD)
        self.trigger_var = StringVar(value=self.s.trigger_word)
        self.caption_var = StringVar(value=self.s.base_caption)
        self.maxside_var = IntVar(value=self.s.max_image_side)
        self._entry_row(opts, "Trigger word", self.trigger_var, width=32)
        self._entry_row(opts, "Extra caption text", self.caption_var, width=60)
        self._spin_row(opts, "Max image side (px)", self.maxside_var, 512, 1536, 64)
        ttk.Label(
            opts,
            text=(
                "Tip: caption what varies (clothing, pose, background) and leave the "
                "constant subject to the trigger word. On a 10GB card keep this at 768."
            ),
            foreground="#555", wraplength=880,
        ).pack(anchor="w", padx=PAD, pady=(0, PAD))

        ttk.Button(f, text="Build dataset", command=self._build_dataset).pack(
            anchor="w", padx=PAD, pady=(0, PAD)
        )
        self.ds_log = ScrolledText(f, height=9, wrap="word")
        self.ds_log.pack(fill=BOTH, expand=False, padx=PAD, pady=(0, PAD))

    # ------------------------------------------------------------------
    # tab 3: train
    # ------------------------------------------------------------------
    def _build_train_tab(self) -> None:
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="3. Train")

        opts = ttk.LabelFrame(f, text="Training parameters")
        opts.pack(fill=X, padx=PAD, pady=PAD)

        grid = ttk.Frame(opts)
        grid.pack(fill=X, padx=PAD, pady=PAD)

        self.steps_var = IntVar(value=self.s.steps)
        self.rank_var = IntVar(value=self.s.rank)
        self.lr_var = StringVar(value=str(self.s.learning_rate))
        self.save_every_var = IntVar(value=self.s.save_every)
        self.res_var = StringVar(value=", ".join(str(r) for r in self.s.resolutions))
        self.offload_var = IntVar(value=self.s.layer_offload_percent)

        self._grid_spin(grid, 0, 0, "Steps", self.steps_var, 200, 10000, 100)
        self._grid_spin(grid, 0, 2, "LoRA rank", self.rank_var, 4, 128, 4)
        self._grid_entry(grid, 1, 0, "Learning rate", self.lr_var)
        self._grid_spin(grid, 1, 2, "Save every", self.save_every_var, 50, 1000, 50)
        self._grid_entry(grid, 2, 0, "Resolutions", self.res_var)
        self._grid_spin(grid, 2, 2, "Layer offload %", self.offload_var, 0, 90, 10)

        ttk.Label(
            opts,
            text=(
                "Out of memory? In this order: remove 768 from Resolutions, drop rank "
                "to 8, then raise Layer offload % to 40-60 (slower but it fits)."
            ),
            foreground="#555", wraplength=880,
        ).pack(anchor="w", padx=PAD, pady=(0, PAD))

        btns = ttk.Frame(f)
        btns.pack(fill=X, padx=PAD)
        self.start_btn = ttk.Button(btns, text="Start training", command=self._start_training)
        self.start_btn.pack(side=LEFT)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self._stop_training, state="disabled")
        self.stop_btn.pack(side=LEFT, padx=4)
        ttk.Button(btns, text="Write config only", command=self._write_config_only).pack(side=LEFT)
        ttk.Button(btns, text="Open output folder", command=self._open_output).pack(side=LEFT, padx=4)

        prog = ttk.LabelFrame(f, text="Progress")
        prog.pack(fill=X, padx=PAD, pady=PAD)
        self.progress = ttk.Progressbar(prog, maximum=100.0)
        self.progress.pack(fill=X, padx=PAD, pady=(PAD, 4))
        self.step_var = StringVar(value="Not started")
        self.eta_var = StringVar(value="")
        ttk.Label(prog, textvariable=self.step_var).pack(anchor="w", padx=PAD)
        ttk.Label(prog, textvariable=self.eta_var, font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w", padx=PAD, pady=(0, PAD)
        )

        self.train_log = ScrolledText(f, height=14, wrap="none")
        self.train_log.pack(fill=BOTH, expand=True, padx=PAD, pady=(0, PAD))

    # ------------------------------------------------------------------
    # tab 4: generate
    # ------------------------------------------------------------------
    def _build_generate_tab(self) -> None:
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="4. Generate")

        left = ttk.Frame(f)
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=PAD, pady=PAD)

        self.lora_var = StringVar(value=self.s.lora_path)
        self._path_row(left, "LoRA file", self.lora_var, self._pick_lora)
        row = ttk.Frame(left)
        row.pack(fill=X, pady=2)
        ttk.Button(row, text="Use latest checkpoint", command=self._use_latest_lora).pack(side=LEFT)

        self.outdir_var = StringVar(value=self.s.output_dir or str(IMAGES_DIR))
        self._path_row(left, "Save images to", self.outdir_var, self._pick_outdir)

        ttk.Label(left, text="Prompt").pack(anchor="w", pady=(PAD, 0))
        self.prompt_box = ScrolledText(left, height=4, wrap="word")
        self.prompt_box.insert("1.0", self.s.gen_prompt)
        self.prompt_box.pack(fill=X)

        ttk.Label(left, text="Negative prompt").pack(anchor="w", pady=(4, 0))
        self.neg_box = ScrolledText(left, height=2, wrap="word")
        self.neg_box.insert("1.0", self.s.gen_negative)
        self.neg_box.pack(fill=X)

        grid = ttk.Frame(left)
        grid.pack(fill=X, pady=PAD)
        self.gsteps_var = IntVar(value=self.s.gen_steps)
        self.gcfg_var = DoubleVar(value=self.s.gen_guidance)
        self.gw_var = IntVar(value=self.s.gen_width)
        self.gh_var = IntVar(value=self.s.gen_height)
        self.gseed_var = IntVar(value=self.s.gen_seed)
        self.lweight_var = DoubleVar(value=self.s.lora_weight)
        self._grid_spin(grid, 0, 0, "Steps", self.gsteps_var, 1, 100, 1)
        self._grid_spin(grid, 0, 2, "Guidance", self.gcfg_var, 0, 20, 0.5)
        self._grid_spin(grid, 1, 0, "Width", self.gw_var, 256, 2048, 64)
        self._grid_spin(grid, 1, 2, "Height", self.gh_var, 256, 2048, 64)
        self._grid_spin(grid, 2, 0, "Seed (-1 random)", self.gseed_var, -1, 2**31 - 2, 1)
        self._grid_spin(grid, 2, 2, "LoRA weight", self.lweight_var, 0, 2, 0.05)

        btns = ttk.Frame(left)
        btns.pack(fill=X)
        self.gen_btn = ttk.Button(btns, text="Generate image", command=self._generate)
        self.gen_btn.pack(side=LEFT)
        ttk.Button(btns, text="Unload model", command=self._unload_model).pack(side=LEFT, padx=4)
        ttk.Button(btns, text="Open folder", command=self._open_images).pack(side=LEFT)

        self.gen_log = ScrolledText(left, height=8, wrap="word")
        self.gen_log.pack(fill=BOTH, expand=True, pady=(PAD, 0))

        right = ttk.LabelFrame(f, text="Preview")
        right.pack(side=RIGHT, fill=Y, padx=PAD, pady=PAD)
        self.preview_label = ttk.Label(right, text="(no image yet)", width=44, anchor="center")
        self.preview_label.pack(padx=PAD, pady=PAD)

    # ------------------------------------------------------------------
    # small widget helpers
    # ------------------------------------------------------------------
    def _entry_row(self, parent, label, var, width=40, show=None):
        row = ttk.Frame(parent)
        row.pack(fill=X, padx=PAD, pady=3)
        ttk.Label(row, text=label, width=22).pack(side=LEFT)
        ttk.Entry(row, textvariable=var, width=width, show=show).pack(
            side=LEFT, fill=X, expand=True
        )
        return row

    def _spin_row(self, parent, label, var, lo, hi, step):
        row = ttk.Frame(parent)
        row.pack(fill=X, padx=PAD, pady=3)
        ttk.Label(row, text=label, width=22).pack(side=LEFT)
        ttk.Spinbox(row, from_=lo, to=hi, increment=step, textvariable=var, width=10).pack(side=LEFT)
        return row

    def _path_row(self, parent, label, var, command):
        row = ttk.Frame(parent)
        row.pack(fill=X, padx=PAD, pady=3)
        ttk.Label(row, text=label, width=22).pack(side=LEFT)
        ttk.Entry(row, textvariable=var).pack(side=LEFT, fill=X, expand=True)
        ttk.Button(row, text="Browse...", command=command).pack(side=LEFT, padx=4)
        return row

    def _grid_spin(self, parent, r, c, label, var, lo, hi, step):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=4, pady=3)
        ttk.Spinbox(
            parent, from_=lo, to=hi, increment=step, textvariable=var, width=12
        ).grid(row=r, column=c + 1, sticky="w", padx=4, pady=3)

    def _grid_entry(self, parent, r, c, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=4, pady=3)
        ttk.Entry(parent, textvariable=var, width=14).grid(
            row=r, column=c + 1, sticky="w", padx=4, pady=3
        )

    def _log(self, widget: ScrolledText, text: str) -> None:
        widget.insert(END, text.rstrip() + "\n")
        widget.see(END)

    # ------------------------------------------------------------------
    # actions: setup
    # ------------------------------------------------------------------
    def _refresh_gpu(self) -> None:
        info = gpu.detect()
        self.gpu_var.set(info.summary())
        if info.total_gb and info.total_gb < 11:
            self.gpu_hint.set(
                "10GB class card. Close your browser and any game launcher before "
                "training, and consider setting NVIDIA Control Panel > Manage 3D "
                "Settings > CUDA Sysmem Fallback Policy to 'Prefer No Sysmem "
                "Fallback' so an overflow fails fast instead of crawling."
            )
        elif info.total_gb:
            self.gpu_hint.set(f"Suggested profile: {presets.get(gpu.recommend_preset(info)).label}")
        else:
            self.gpu_hint.set("Training needs an NVIDIA GPU with CUDA drivers installed.")

    def _pick_toolkit(self) -> None:
        d = filedialog.askdirectory(title="Select your ai-toolkit folder")
        if d:
            self.toolkit_var.set(d)
            self.toolkit_py_var.set(default_toolkit_python(d))

    def _pick_python(self) -> None:
        p = filedialog.askopenfilename(title="Select python executable")
        if p:
            self.toolkit_py_var.set(p)

    def _on_preset_change(self, _event=None) -> None:
        key = presets.key_for_label(self.preset_var.get())
        p = presets.get(key)
        self.preset_note.set(p.notes)
        self.res_var.set(", ".join(str(r) for r in p.default_resolutions))
        self.gsteps_var.set(p.sample_steps)
        self.gcfg_var.set(p.sample_guidance)

    def _save_settings(self) -> None:
        self._collect()
        self.s.save()
        self.status.set("Settings saved.")

    def _collect(self) -> None:
        """Pull every widget value back into the Settings object."""
        s = self.s
        s.toolkit_dir = self.toolkit_var.get().strip()
        s.toolkit_python = self.toolkit_py_var.get().strip()
        s.hf_token = self.token_var.get().strip()
        s.project_name = (self.project_var.get().strip() or "my_first_lora").replace(" ", "_")
        s.preset_key = presets.key_for_label(self.preset_var.get())
        s.trigger_word = self.trigger_var.get().strip()
        s.base_caption = self.caption_var.get().strip()
        s.max_image_side = int(self.maxside_var.get())
        s.steps = int(self.steps_var.get())
        s.rank = int(self.rank_var.get())
        try:
            s.learning_rate = float(self.lr_var.get())
        except ValueError:
            s.learning_rate = 1e-4
            self.lr_var.set("0.0001")
        s.save_every = int(self.save_every_var.get())
        s.sample_every = int(self.save_every_var.get())
        s.layer_offload_percent = int(self.offload_var.get())
        res = []
        for token in self.res_var.get().replace(";", ",").split(","):
            token = token.strip()
            if token.isdigit():
                res.append(int(token))
        s.resolutions = res or presets.get(s.preset_key).default_resolutions
        s.gen_prompt = self.prompt_box.get("1.0", END).strip()
        s.gen_negative = self.neg_box.get("1.0", END).strip()
        s.gen_steps = int(self.gsteps_var.get())
        s.gen_guidance = float(self.gcfg_var.get())
        s.gen_width = int(self.gw_var.get())
        s.gen_height = int(self.gh_var.get())
        s.gen_seed = int(self.gseed_var.get())
        s.lora_weight = float(self.lweight_var.get())
        s.lora_path = self.lora_var.get().strip()
        s.output_dir = self.outdir_var.get().strip() or str(IMAGES_DIR)

    # ------------------------------------------------------------------
    # actions: dataset
    # ------------------------------------------------------------------
    def _pick_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select training images",
            initialdir=self.s.last_image_dir or str(Path.home()),
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self.s.last_image_dir = str(Path(paths[0]).parent)
            self._add_images(paths)

    def _pick_folder(self) -> None:
        d = filedialog.askdirectory(title="Select a folder of images")
        if not d:
            return
        found = [str(p) for p in sorted(Path(d).iterdir()) if p.suffix.lower() in ds.SUPPORTED]
        if not found:
            messagebox.showinfo("Nothing found", "No supported images in that folder.")
            return
        self.s.last_image_dir = d
        self._add_images(found)

    def _add_images(self, paths) -> None:
        added = 0
        for p in paths:
            if p not in self.selected_images:
                self.selected_images.append(p)
                self.listbox.insert(END, p)
                added += 1
        self._update_count()
        self.status.set(f"Added {added} image(s).")

    def _remove_selected(self) -> None:
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)
            del self.selected_images[i]
        self._update_count()

    def _clear_images(self) -> None:
        self.listbox.delete(0, END)
        self.selected_images.clear()
        self._update_count()

    def _update_count(self) -> None:
        self.listbox_count.set(f"{len(self.selected_images)} images selected")

    def _build_dataset(self) -> None:
        self._collect()
        if not self.selected_images:
            messagebox.showwarning("No images", "Add some images first.")
            return
        self.ds_log.delete("1.0", END)
        self._log(self.ds_log, f"Building dataset in {self.s.dataset_dir}")
        res = ds.prepare(
            self.selected_images,
            self.s.dataset_dir,
            self.s.trigger_word,
            self.s.base_caption,
            max_side=self.s.max_image_side,
            clean=True,
            log=lambda m: self._log(self.ds_log, m),
        )
        for w in res.warnings:
            self._log(self.ds_log, f"  ! {w}")
        for sk in res.skipped:
            self._log(self.ds_log, f"  - skipped {sk}")
        self._log(self.ds_log, f"Done: {res.written} images + captions written.")
        self.s.save()
        self.status.set(f"Dataset ready: {res.written} images.")

    # ------------------------------------------------------------------
    # actions: training
    # ------------------------------------------------------------------
    def _write_config_only(self) -> None:
        self._collect()
        path = trainer.write_config(self.s)
        self._log(self.train_log, f"Config written: {path}")
        self.status.set("Config written.")

    def _start_training(self) -> None:
        self._collect()
        self.s.save()
        count = ds.existing_count(self.s.dataset_dir)
        errors, estimate = trainer.preflight(self.s, count)
        if errors:
            messagebox.showerror("Not ready yet", "\n\n".join(errors))
            return
        self.train_log.delete("1.0", END)
        self._log(self.train_log, f"Dataset: {count} images in {self.s.dataset_dir}")
        self._log(self.train_log, estimate)
        self.eta_var.set("starting up (model load takes a few minutes the first time)...")
        self.progress["value"] = 0
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.run = trainer.TrainingRun(
            self.s,
            on_log=lambda m: self.events.put(("train_log", m)),
            on_progress=lambda p: self.events.put(("train_prog", p)),
            on_finish=lambda ok, m: self.events.put(("train_done", ok, m)),
        )
        self.run.start()
        self.status.set("Training...")

    def _stop_training(self) -> None:
        if self.run:
            self.run.stop()

    def _open_output(self) -> None:
        self._open_path(Path(self.s.config_path).parent.parent / "training_output")

    # ------------------------------------------------------------------
    # actions: generation
    # ------------------------------------------------------------------
    def _pick_lora(self) -> None:
        p = filedialog.askopenfilename(
            title="Select a trained LoRA",
            filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
        )
        if p:
            self.lora_var.set(p)

    def _use_latest_lora(self) -> None:
        self._collect()
        found = trainer.find_checkpoints(self.s)
        if not found:
            messagebox.showinfo("Nothing yet", "No .safetensors checkpoints found for this project.")
            return
        self.lora_var.set(str(found[-1]))
        self.status.set(f"Using {found[-1].name}")

    def _pick_outdir(self) -> None:
        d = filedialog.askdirectory(title="Where should generated images go?")
        if d:
            self.outdir_var.set(d)

    def _generate(self) -> None:
        self._collect()
        self.s.save()
        if not self.s.gen_prompt:
            messagebox.showwarning("No prompt", "Type a prompt first.")
            return
        self.gen_btn.config(state="disabled")
        self.status.set("Generating...")
        self._log(self.gen_log, "-" * 60)

        def work():
            try:
                result = self.generator.generate(self.s)
                self.events.put(("gen_done", result))
            except Exception as exc:
                self.events.put(("gen_error", f"{exc}\n\n{traceback.format_exc()}"))

        threading.Thread(target=work, daemon=True).start()

    def _unload_model(self) -> None:
        self.generator.unload()
        self._log(self.gen_log, "Model unloaded, VRAM released.")

    def _open_images(self) -> None:
        self._open_path(Path(self.outdir_var.get() or IMAGES_DIR))

    def _show_preview(self, path: Path) -> None:
        try:
            from PIL import Image, ImageTk

            with Image.open(path) as im:
                im = im.copy()
            im.thumbnail((360, 520))
            self._preview_ref = ImageTk.PhotoImage(im)
            self.preview_label.config(image=self._preview_ref, text="")
        except Exception as exc:
            self.preview_label.config(text=f"(preview unavailable: {exc})")

    # ------------------------------------------------------------------
    # event pump
    # ------------------------------------------------------------------
    def _pump(self) -> None:
        try:
            while True:
                evt = self.events.get_nowait()
                kind = evt[0]
                if kind == "train_log":
                    self._log(self.train_log, evt[1])
                elif kind == "train_prog":
                    p: trainer.Progress = evt[1]
                    self.progress["value"] = p.fraction * 100.0
                    loss = f"   loss {p.loss:.4f}" if p.loss is not None else ""
                    rate = f"   {p.sec_per_it:.2f} s/step" if p.sec_per_it else ""
                    self.step_var.set(
                        f"Step {p.step} / {p.total}  ({p.fraction*100:.1f}%)"
                        f"   elapsed {trainer.format_duration(p.elapsed)}{rate}{loss}"
                    )
                    self.eta_var.set(p.eta_text())
                elif kind == "train_done":
                    ok, msg = evt[1], evt[2]
                    self._log(self.train_log, msg)
                    self.eta_var.set(msg)
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status.set(msg)
                    if ok:
                        self.progress["value"] = 100
                        found = trainer.find_checkpoints(self.s)
                        if found:
                            self.lora_var.set(str(found[-1]))
                            self._log(self.train_log, f"Latest checkpoint: {found[-1]}")
                            self.nb.select(3)
                elif kind == "gen_log":
                    self._log(self.gen_log, evt[1])
                elif kind == "gen_done":
                    res = evt[1]
                    self._log(self.gen_log, f"Done in {res.seconds:.1f}s (seed {res.seed})")
                    self._show_preview(res.image_path)
                    self.gen_btn.config(state="normal")
                    self.status.set(f"Saved {res.image_path.name}")
                elif kind == "gen_error":
                    self._log(self.gen_log, "ERROR: " + evt[1])
                    self.gen_btn.config(state="normal")
                    self.status.set("Generation failed - see the log.")
        except queue.Empty:
            pass
        self.root.after(100, self._pump)

    # ------------------------------------------------------------------
    def _open_path(self, path: Path) -> None:
        import os
        import subprocess
        import sys

        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showinfo("Path", f"{path}\n\n({exc})")

    def _on_close(self) -> None:
        if self.run and self.run.running:
            if not messagebox.askyesno("Training is running", "Stop training and quit?"):
                return
            self.run.stop()
        try:
            self._collect()
            self.s.save()
        except Exception:
            pass
        self.root.destroy()


def main() -> None:
    root = Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()
