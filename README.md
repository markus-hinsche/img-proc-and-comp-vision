# Computer Vision with PyTorch — DSR 3-day course

A practical, notebook-first introduction to Computer Vision using PyTorch.

**Story arc**

- **Day 1 — How computers see images.** Tensors, image processing, convolutions, datasets, first CNN.
- **Day 2 — How CNNs learn representations.** Architectures, training, transfer learning, explainability.
- **Day 3 — How Transformers changed vision.** Attention, patch embeddings, ViTs, CNN vs ViT.

---

## Setup

Target: from zero to running notebooks in **under 10 minutes**, on Mac / Linux / Windows, CPU-only is fine.

### 1. Install `uv`

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager. It handles the Python version, the virtual environment, and dependencies — no `conda`, no `pyenv`, no manual `python -m venv`.

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```bash
uv --version
```

### 2. Clone this repo

```bash
git clone <repo-url>
cd img-proc-and-comp-vision
```

### 3. Install dependencies

From the repo root:

```bash
uv sync
```

This will:

- download Python 3.11 if you don't have it (no need to install Python yourself),
- create a `.venv/` inside the project,
- install PyTorch, torchvision, JupyterLab, matplotlib, and friends,
- install the local `cvcourse` helper package so `import cvcourse` just works.

First run takes ~2 minutes on a decent connection. Subsequent runs are cached.

### 4. Open a notebook

Two options — pick whichever you prefer.

**Option A — VS Code / Cursor (recommended if you already use it)**

1. Install the **Python** and **Jupyter** extensions (Cursor ships them; in VS Code, install from the Extensions panel).
2. Open the repo folder: `File → Open Folder…` → select `img-proc-and-comp-vision`.
3. Open `notebooks/01_images_as_tensors.ipynb`.
4. Top-right of the notebook, click **Select Kernel** → **Python Environments** → pick the one at `.venv/bin/python` (created by `uv sync`). Cursor/VS Code usually pre-selects this automatically.
5. Click **Run All**.

If `.venv` doesn't show up in the kernel list, reload the window (`Cmd/Ctrl+Shift+P` → "Developer: Reload Window") and try again.

**Option B — JupyterLab in the browser**

```bash
uv run jupyter lab
```

JupyterLab opens in your browser. Open `notebooks/01_images_as_tensors.ipynb` and run all cells.

---

Either way: if the last cell renders a grid of random images, **you're set up**.

---

## What's in the repo

```
img-proc-and-comp-vision/
├── pyproject.toml            # dependencies (uv-managed)
├── .python-version           # Python 3.11
├── notebooks/                # teaching notebooks (the main interface)
│   ├── 01_images_as_tensors.ipynb
│   ├── 02_convolutions.ipynb
│   ├── 03_datasets_and_augmentations.ipynb
│   ├── 04_first_cnn.ipynb
│   ├── 05_training_cnn.ipynb
│   ├── 06_transfer_learning.ipynb
│   ├── 07_gradcam.ipynb
│   ├── 08_attention_intro.ipynb
│   ├── 09_building_vit.ipynb
│   └── 10_cnn_vs_vit.ipynb
├── src/cvcourse/             # reusable helpers used by notebooks
├── slides/                   # short slide decks (Marp markdown) — see SLIDES.md
├── SLIDES.md                 # how to build/edit slides + Marp syntax reference
├── scripts/build_notebooks.py  # regenerates notebooks/ from source
└── data/                     # cached datasets (gitignored)
```

---

## Smoke test

If you want to confirm everything works without opening JupyterLab:

```bash
uv run python -c "import torch, torchvision, cvcourse; print('torch', torch.__version__)"
```

To execute a notebook headlessly (good for CI / re-runs):

```bash
uv run python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('notebooks/01_images_as_tensors.ipynb', as_version=4)
NotebookClient(nb, timeout=120, kernel_name='python3').execute()
print('OK')
"
```

---

## Slides

Short decks live in `slides/` as Marp markdown (e.g. `slides/intro.md`,
`slides/debugging.md`) and are built to PDF. See [SLIDES.md](SLIDES.md)
for build commands, frontmatter, and syntax reference.

---

## Hardware notes

- **CPU is fine** for everything in this course. We deliberately picked small datasets (FashionMNIST, CIFAR10) and small models.
- **Apple Silicon (M1/M2/M3):** PyTorch's MPS backend will be auto-detected by `cvcourse.get_device()`. You don't need to do anything.
- **NVIDIA GPU:** the default `pip` PyTorch wheel ships with CUDA support out of the box on Linux/Windows. If you want a specific CUDA version, edit the `torch` line in `pyproject.toml`.
- **No GPU:** still fine. The Day 1 CNN trains in ~2 minutes on CPU. Day 2 transfer-learning is the slowest piece; budget ~5–10 minutes.

---

## Troubleshooting

**`uv: command not found`** → reopen your shell, or `source ~/.zshrc` / `source ~/.bashrc`. `uv` installs to `~/.local/bin` and the installer adds it to your PATH.

**`uv sync` hangs / fails** → check your network. Behind a corporate proxy, set `HTTPS_PROXY`. Behind a strict firewall, you may need to pre-download wheels.

**`ModuleNotFoundError: No module named 'cvcourse'`** in a notebook → you opened JupyterLab from outside `uv run`. Always launch it with `uv run jupyter lab` from the repo root, so the right virtualenv is used.

**Kernel says "Python 3" but cells fail with import errors** → in JupyterLab, top-right corner shows the active kernel. It should point inside `.venv`. If not, run `uv run python -m ipykernel install --user --name cv-course` once.

**FashionMNIST download fails** → it auto-caches into `data/`. Delete a half-downloaded `data/FashionMNIST/raw/` and retry. The mirror occasionally hiccups.

**Notebook outputs look stale** → re-run all cells (`Kernel → Restart Kernel and Run All Cells`).

---

## For instructors — editing the course

Notebooks are **generated** from `scripts/build_notebooks.py`. Each notebook's cells live as plain Python lists of markdown/code blocks, which is much easier to diff and review than raw `.ipynb` JSON.

To edit a notebook's content:

1. Edit `scripts/build_notebooks.py` (the `N01`, `N02`, … lists).
2. Regenerate: `python3 scripts/build_notebooks.py`.
3. Re-execute to refresh outputs (optional but recommended before teaching):
   ```bash
   uv run python -c "
   import nbformat; from nbclient import NotebookClient
   nb = nbformat.read('notebooks/01_images_as_tensors.ipynb', as_version=4)
   NotebookClient(nb, timeout=600, kernel_name='python3').execute()
   nbformat.write(nb, 'notebooks/01_images_as_tensors.ipynb')
   "
   ```

If you'd rather edit `.ipynb` files directly in JupyterLab, that's fine too — just know the build script will overwrite them if you re-run it.

---

## Teaching philosophy

- **Visualize first, theory second.** CV is visual; tensors and feature maps should be on screen constantly.
- **No abstraction magic.** Explicit training loops, no Lightning. Once `loss.backward()` and `optimizer.step()` click, abstractions are convenience, not mystery.
- **Debug-friendly.** Tensor shapes, devices, normalization, `eval()` vs `train()` — taught explicitly, with intentional bugs. See `slides/debugging.md`.
- **CPU-first.** GPU helps but is never required.

## Prerequisites

- Comfortable with Python.
- Basic PyTorch (you've seen `nn.Module`, tensors, `optimizer.step()` before).
- No prior Computer Vision knowledge required.
