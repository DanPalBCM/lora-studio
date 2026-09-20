# LoRA Studio

A small Tkinter desktop app for the loop you described: pick images from your
file explorer → build a dataset → train a LoRA with a live ETA → type a prompt →
get a PNG saved to a folder you choose.

It is a **front-end**, not a training engine. Training is driven by
[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit), which is the tool
people are actually using to train Z-Image / FLUX.2 / Qwen LoRAs on consumer
cards. This app writes the job config, launches it, parses its progress output,
and then loads the resulting `.safetensors` for inference. That division keeps
this codebase small and means upstream fixes reach you for free.

Tuned by default for an **RTX 3080 10GB**.

---

## Read this first

I wrote this without a GPU to test on, so treat the first run as a shakedown.
Two specific things are most likely to need a one-line edit:

1. **The `arch` string in `lorastudio/presets.py`.** ai-toolkit changes its
   architecture identifiers between releases. If training dies immediately with
   something like *unknown architecture*, open
   `<ai-toolkit>/config/examples/`, find the example config for your model, copy
   its `arch:` value, and paste it into the matching preset. Everything else in
   the generated config is standard.
2. **The diffusers class for Z-Image on the Generate tab.** `generate.py` asks
   diffusers to resolve the pipeline from the repo itself and retries with
   `trust_remote_code=True`. If your diffusers version is too old it will fail
   with a clear message — `pip install -U diffusers` usually fixes it. Training
   is unaffected either way; you can always load the LoRA in ComfyUI instead.

---

## Install

### 1. This app

```bash
cd lora_studio
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

# torch, for the Generate tab only - match your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu124

python check_env.py
```

### 2. ai-toolkit (the trainer)

Clone it somewhere with plenty of disk (allow 100GB+ for model caches):

```bash
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then in LoRA Studio's **Setup** tab, point "ai-toolkit folder" at that clone.
The app will auto-detect its `venv` interpreter.

### 3. Run

```bash
python run_gui.py
```

---

## The four tabs

**1. Setup** — detects your GPU, points at ai-toolkit, picks a base model
profile. The default is `Z-Image Turbo — 10GB profile`.

**2. Dataset** — multi-select images from your file explorer (or add a whole
folder). It EXIF-rotates, converts to RGB, downsizes so the longest side is at
most 768px, and writes a `.txt` caption beside each image containing your
trigger word. Output lands in `workspace/datasets/<project>/`.

**3. Train** — writes the ai-toolkit YAML, launches the run, and shows:
- a progress bar and `step N / total`
- a **live ETA** computed from a rolling window of the last ~60 steps, plus the
  wall-clock time it expects to finish
- a **pre-flight estimate** printed before the run starts, from the profile's
  expected seconds-per-step
- s/step, loss, and the full ai-toolkit log (also written to `workspace/logs/`)

OOM is detected in the output and turns into an actionable message rather than
a stack trace. Checkpoints land in `workspace/training_output/<project>/`.

**4. Generate** — picks up the newest checkpoint automatically when training
finishes, takes your prompt, generates one image, saves it as PNG (with the
prompt/seed/settings embedded plus a `.json` sidecar) into whatever folder you
chose, and shows a preview. The pipeline stays loaded, so the second image is
much faster than the first.

---

## Suggested first run on a 3080 10GB

| Setting | Value |
|---|---|
| Profile | Z-Image Turbo — 10GB |
| Images | 20–40, varied angles / expressions / lighting |
| Max image side | 768 |
| Trigger word | something rare, e.g. `ohwx person` |
| Resolutions | `512, 640` |
| Rank | 16 |
| Learning rate | 0.0001 |
| Steps | 2000 |
| Save every | 250 |
| Layer offload % | 0 (raise only if you OOM) |

**Do a fit test before the real run.** Set Steps to 300 and Save every to 100.
A run that survives loading, a training step, a sample preview *and* a
checkpoint save is a run that fits. Previews and saves are what kill jobs at
step 250 after step 249 looked fine.

Expect roughly 4–8 s/step at these settings, so 2000 steps is a 3–5 hour
overnight job. At 12GB and 768px buckets people report 2–3 s/step.

### If you run out of VRAM

In this order:
1. Remove `640` (then `768`) from Resolutions.
2. Drop rank to 8.
3. Raise **Layer offload %** to 40–60. This streams transformer layers between
   CPU and GPU — it fits, but each step gets several times slower.

Also worth doing once, outside the app: move your display to the motherboard's
iGPU or the other PC so Windows isn't holding ~1GB of your 10GB, and set
NVIDIA Control Panel → Manage 3D Settings → **CUDA — Sysmem Fallback Policy** to
*Prefer No Sysmem Fallback* for python.exe, so an overflow fails fast instead of
silently crawling at 40 s/step.

---

## Picking the best checkpoint

The last checkpoint is usually **not** the best one. Over-trained identity LoRAs
go waxy and stop responding to prompts. Generate the same prompt and seed
against checkpoints at 750 / 1000 / 1250 / 1500 steps and pick by eye — the
Generate tab's fixed-seed option exists for exactly this.

## Hands, fingers, faces

Your LoRA teaches identity, not anatomy. The remaining artefacts are fixed at
inference time, not in training: generate, then inpaint the face and hands at
higher resolution (ADetailer in A1111/Forge, or a FaceDetailer node in ComfyUI),
then upscale. That single pass removes more mangled fingers than any amount of
LoRA tuning.

---

## Layout

```
lora_studio/
  run_gui.py            launch this
  check_env.py          pre-flight environment check
  requirements.txt
  lorastudio/
    gui.py              Tkinter UI, threading, event pump
    settings.py         persisted settings + workspace paths
    presets.py          base models and per-VRAM-tier knobs  <- edit arch here
    dataset.py          image import, resize, caption writing
    trainer.py          config builder, subprocess, ETA parsing
    generate.py         diffusers inference + LoRA loading
    gpu.py              GPU detection
  workspace/            created on first run
    datasets/  configs/  training_output/  generated/  logs/
```

## Licensing note

Base model licences differ and they follow your LoRA. Z-Image Turbo and
FLUX.2 klein 4B are Apache 2.0; FLUX.2 klein 9B and FLUX.2 dev are
non-commercial; SDXL is OpenRAIL++-M. If you plan to sell anything, check the
licence of the base you trained on. And if you are training on photographs of a
real person, get their consent.
