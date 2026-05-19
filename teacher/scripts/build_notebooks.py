"""Generate the course notebooks from compact Python sources.

Run with:  python3 teacher/scripts/build_notebooks.py

We keep the *authoring* form as plain Python lists of (kind, text) cells so the
content is easy to edit, diff, and review. The script writes one .ipynb per
notebook into ``notebooks/``. Students never see this script — they just open
the generated notebooks.

No external dependencies: stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def _cell_id(kind: str, text: str) -> str:
    """Stable 8-char id derived from cell kind + content.

    nbformat 4.5+ requires every cell to have an id. If we omit it, Jupyter
    assigns a random one on first open — which then changes again on the next
    rebuild, producing meaningless diffs. A content-hash id keeps unchanged
    cells stable across rebuilds.
    """
    h = hashlib.sha1(f"{kind}\0{text}".encode("utf-8")).hexdigest()
    return h[:8]


def md(text: str) -> dict:
    text = textwrap.dedent(text).strip("\n")
    return {
        "cell_type": "markdown",
        "id": _cell_id("markdown", text),
        "metadata": {},
        "source": _splitlines(text),
    }


def code(text: str) -> dict:
    text = textwrap.dedent(text).strip("\n")
    return {
        "cell_type": "code",
        "id": _cell_id("code", text),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _splitlines(text),
    }


def _splitlines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        pass  # nbformat does not require trailing newline
    return lines


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (cv-course)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, cells: list[dict]) -> None:
    path = NB_DIR / name
    path.write_text(json.dumps(notebook(cells), indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Day 1 — How computers see images                                            #
# --------------------------------------------------------------------------- #

N01 = [
    md(r"""
    # 01 — Images as tensors

    **Day 1 · Notebook 1 of 4**

    Goal: by the end of this notebook you can answer

    1. What *is* an image, as far as PyTorch is concerned?
    2. What does `(N, C, H, W)` mean and why does the order matter?
    3. What does normalization do to pixel values, and how do you undo it for display?

    Run every cell in order. Stop and inspect the **shape** of every tensor you create — that habit alone prevents most CV bugs.
    """),
    code("""
    import sys, pathlib
    # Make the local src/cvcourse package importable when running notebooks
    # from the notebooks/ folder.
    sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

    import torch
    import torchvision
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    from cvcourse import show_image, show_grid

    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    """),
    md("""
    ## 1. Load an image — three ways

    Pretend you have never used PyTorch. An image on disk is just bytes. Different
    libraries hand you different representations of the same picture.
    """),
    md("""
    Real life: you'll get an image as a `PIL.Image`, a numpy array, or a torch
    tensor — same picture, three representations. Before touching any real data,
    let's construct a *synthetic* image so we control every pixel.
    """),
    code("""
    # Build a synthetic 8x8 RGB image by hand so we can reason about every value.
    img_np = np.zeros((8, 8, 3), dtype=np.uint8)
    img_np[:, :4, 0] = 255     # left half is red
    img_np[:, 4:, 2] = 255     # right half is blue
    img_np[3:5, :, 1] = 255    # middle horizontal stripe is green
    print("numpy image shape (H, W, C):", img_np.shape)
    print("dtype:", img_np.dtype, "min:", img_np.min(), "max:", img_np.max())

    plt.imshow(img_np); plt.title("synthetic 8x8 RGB"); plt.axis("off"); plt.show()
    """),
    md("""
    ## 2. HWC vs CHW — the layout switch that bites everyone

    NumPy and PIL store images as **H × W × C**. PyTorch convolutions want **C × H × W**.
    Get this wrong and a model trains on garbage without ever raising an error.
    """),
    code("""
    img_t = torch.from_numpy(img_np)          # still HWC
    print("HWC:", img_t.shape)

    img_chw = img_t.permute(2, 0, 1)          # -> CHW
    print("CHW:", img_chw.shape)

    # Sanity check: the red channel should be 255 on the left and 0 on the right.
    print("red row 0:", img_chw[0, 0].tolist())
    """),
    md("""
    ### Mini-exercise

    Without running the cell, predict the shape after each step:

    ```python
    x = torch.zeros(28, 28)        # ?
    x = x.unsqueeze(0)             # ?
    x = x.unsqueeze(0)             # ?
    x = x.repeat(4, 1, 1, 1)       # ?
    ```

    Then run it and check.
    """),
    code("""
    x = torch.zeros(28, 28)
    print(x.shape)
    x = x.unsqueeze(0); print(x.shape)
    x = x.unsqueeze(0); print(x.shape)
    x = x.repeat(4, 1, 1, 1); print(x.shape)
    """),
    md("""
    ## 3. Batches: the leading dimension

    PyTorch models almost always expect a batch dimension up front: `(N, C, H, W)`.
    Even when you have one image, you wrap it in a batch of size 1.
    """),
    code("""
    batch = img_chw.unsqueeze(0).float() / 255.0     # 1 image, float in [0, 1]
    print("batch:", batch.shape, batch.dtype, "min/max:", batch.min().item(), batch.max().item())
    """),
    md("""
    ## 4. From uint8 [0, 255] to float [0, 1] to normalized

    Most pretrained networks were trained with inputs roughly centered around zero.
    The classic ImageNet normalization is

    ```
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    x_norm = (x - mean) / std
    ```

    Display works on the *un-normalized* image. Forgetting to invert normalization
    before plotting is a top-3 source of "why does my image look insane" bugs.
    """),
    code("""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    x = batch[0]                       # (3, 8, 8) in [0, 1]
    x_norm = (x - mean) / std
    x_back = x_norm * std + mean       # invert
    print("after normalize  -> mean/std per channel:",
          x_norm.mean(dim=(1, 2)).tolist(), x_norm.std(dim=(1, 2)).tolist())
    print("after un-normalize -> equals original?",
          torch.allclose(x_back, x, atol=1e-6))
    """),
    md("""
    ## 5. Resize, crop, the usual tricks

    `torchvision.transforms.v2` is the modern API. It works on PIL images **and**
    tensors. Use tensor transforms whenever you can — they're faster and
    GPU-friendly.
    """),
    code("""
    from torchvision.transforms import v2 as T

    big = torch.rand(3, 256, 256)
    tfm = T.Compose([
        T.Resize(64),                # shortest side -> 64
        T.CenterCrop(48),
        T.ToDtype(torch.float32, scale=True),
    ])
    out = tfm(big)
    print("input:", big.shape, "-> output:", out.shape)
    """),
    md("""
    ## 6. Visualize a batch

    Visualizing batches *constantly* is the #1 trick for CV. If you can't picture
    what's flowing through your model, you'll lose hours to bugs that a single
    `plt.imshow` would have caught.
    """),
    code("""
    fake_batch = torch.rand(8, 3, 32, 32)
    show_grid(fake_batch, cols=4)
    """),
    md("""
    ## Recap

    - Images are tensors. NumPy/PIL use **HWC**, PyTorch uses **CHW**.
    - Batches are **(N, C, H, W)**.
    - Always check `.shape`, `.dtype`, and value range.
    - Normalize for the network. Un-normalize for the eyes.

    Next up: **convolutions** — the operation that makes CNNs work.
    """),
]

# --------------------------------------------------------------------------- #

N02 = [
    md(r"""
    # 02 — Convolutions

    **Day 1 · Notebook 2 of 4**

    Goal: build the mental model that **a convolution is a small sliding pattern
    detector**. Once you see that, CNNs are no longer mysterious — they're just
    stacks of *learned* pattern detectors.

    We'll:

    1. Hand-build a few classical kernels (blur, edge) and run them on a real image.
    2. Use `torch.nn.functional.conv2d` directly so the mechanics are visible.
    3. Watch what stacking convolutions does to spatial size.
    """),
    code("""
    import torch
    import torch.nn.functional as F
    import torchvision
    from torchvision import transforms as T
    import matplotlib.pyplot as plt
    import numpy as np

    from cvcourse import show_image, show_grid
    """),
    md("""
    ## 1. A real image

    We'll grab a single FashionMNIST sample. (FashionMNIST downloads in seconds
    and works offline once cached.)
    """),
    code("""
    ds = torchvision.datasets.FashionMNIST(
        root="../data", train=True, download=True,
        transform=T.ToTensor(),
    )
    img, label = ds[0]
    print("image:", img.shape, "label:", label, "(", ds.classes[label], ")")
    show_image(img)
    """),
    md("""
    ## 2. A blur kernel by hand

    A box blur is a 3×3 kernel of `1/9`. Every output pixel = average of the 3×3
    neighborhood in the input.
    """),
    code("""
    blur = torch.full((1, 1, 3, 3), 1/9)
    print("kernel shape (out_c, in_c, kH, kW):", blur.shape)

    x = img.unsqueeze(0)                     # (1, 1, 28, 28)
    blurred = F.conv2d(x, blur, padding=1)
    print("in:", x.shape, "out:", blurred.shape)

    show_grid([img, blurred[0]], titles=["original", "blurred"], cols=2)
    """),
    md("""
    ## 3. Edge detection — Sobel

    Two kernels, one detects vertical edges, one horizontal. The magnitude of the
    combined response highlights *all* edges.
    """),
    code("""
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]]).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    mag = (gx**2 + gy**2).sqrt()

    show_grid(
        [img, gx[0].abs(), gy[0].abs(), mag[0]],
        titles=["original", "|Gx|", "|Gy|", "magnitude"],
        cols=4,
    )
    """),
    md("""
    ## 4. The shape rule

    For a kernel of size `k` with stride `s` and padding `p`,
    the output spatial size is

    `out = floor((in + 2p - k) / s) + 1`.

    Memorize this. Half of all CNN debugging is verifying this equation.
    """),
    code("""
    def out_size(in_size, k, s=1, p=0):
        return (in_size + 2*p - k) // s + 1

    for k, s, p in [(3, 1, 0), (3, 1, 1), (5, 2, 2), (7, 2, 3)]:
        print(f"k={k} s={s} p={p}  ->  28 -> {out_size(28, k, s, p)}")
    """),
    md("""
    ## 5. Learned filters preview

    Real CNNs start with *random* kernels and train them. Here we just glance at
    what a randomly initialized 8-filter conv layer "sees" before any training —
    mostly noise, but with interesting structure already.
    """),
    code("""
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)
    feat = conv(x)
    print("feature maps:", feat.shape)
    show_grid([feat[0, i].detach() for i in range(8)], cols=4)
    """),
    md("""
    ## Recap

    - A convolution slides a small kernel over the image, producing one number per location.
    - Hand-built kernels can blur, sharpen, detect edges.
    - CNNs learn these kernels from data — same operation, learned weights.
    - The shape rule is non-negotiable: `(in + 2p - k) / s + 1`.

    Next: **datasets and DataLoader** — feeding batches into a network without losing your sanity.
    """),
]

# --------------------------------------------------------------------------- #

N03 = [
    md(r"""
    # 03 — Datasets and augmentations

    **Day 1 · Notebook 3 of 4**

    By the end you can:

    1. Load a torchvision dataset and iterate over batches with `DataLoader`.
    2. Apply train-time augmentations and visualize them.
    3. Spot the difference between "transform applied to the image" and
       "transform applied to the model input."
    """),
    code("""
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import transforms as T
    import matplotlib.pyplot as plt

    from cvcourse import show_grid
    """),
    md("""
    ## 1. A `Dataset` is just an indexable thing

    Anything that supports `len()` and `dataset[i]` works. torchvision ships many.
    """),
    code("""
    train_ds = torchvision.datasets.FashionMNIST(
        root="../data", train=True, download=True,
        transform=T.ToTensor(),
    )
    test_ds = torchvision.datasets.FashionMNIST(
        root="../data", train=False, download=True,
        transform=T.ToTensor(),
    )
    print("train:", len(train_ds), "test:", len(test_ds))
    img, label = train_ds[0]
    print("first sample:", img.shape, img.dtype, "label:", train_ds.classes[label])
    """),
    md("""
    ## 2. `DataLoader` = batching + shuffling + parallel workers
    """),
    code("""
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    xb, yb = next(iter(train_loader))
    print("batch images:", xb.shape, "labels:", yb.shape)
    show_grid(xb[:16], titles=[train_ds.classes[i] for i in yb[:16]], cols=8)
    """),
    md("""
    ## 3. Augmentation: same image, more views

    Augmentations make the model see each training image many ways. Use them at
    **train time only** — never on the validation/test set.
    """),
    code("""
    aug = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
    ])

    # Reload one sample with augmentation, show 8 random versions of the same image.
    train_aug = torchvision.datasets.FashionMNIST(
        root="../data", train=True, download=False, transform=aug,
    )
    versions = [train_aug[0][0] for _ in range(8)]
    show_grid(versions, cols=8)
    """),
    md("""
    ### Pitfall: don't augment your validation set

    If you augment validation, the metric becomes noisy and you can't compare
    models. Keep val/test transforms deterministic — usually just resize +
    normalize.
    """),
    md("""
    ## 4. Normalization belongs in the transform pipeline
    """),
    code("""
    pipeline = T.Compose([
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),   # FashionMNIST mean/std
    ])
    train_ds_norm = torchvision.datasets.FashionMNIST(
        root="../data", train=True, download=False, transform=pipeline,
    )
    img, _ = train_ds_norm[0]
    print("after normalize -> mean:", img.mean().item(), "std:", img.std().item())
    """),
    md("""
    ## Recap

    - `Dataset` + `DataLoader` is the standard pattern.
    - Augment training data; keep validation deterministic.
    - Normalize inside the transform pipeline so it's part of the data, not the model.

    Next: **first CNN** — putting it all together.
    """),
]

# --------------------------------------------------------------------------- #

N04 = [
    md(r"""
    # 04 — Your first CNN

    **Day 1 · Notebook 4 of 4**

    We'll train a small CNN on FashionMNIST. The training loop is **fully
    explicit** — no Lightning, no `model.fit`. You'll see every step:

    `forward → loss → backward → step → zero_grad`.

    Target: > 85% test accuracy in a few minutes of CPU training.
    """),
    code("""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import transforms as T
    from tqdm.auto import tqdm

    from torchinfo import summary

    from cvcourse import get_device, count_params, show_grid

    # get_device() auto-picks cuda > mps > cpu.
    # To force a specific backend, replace with one of:
    #   device = torch.device("cpu")
    #   device = torch.device("cuda")   # NVIDIA GPU
    #   device = torch.device("mps")    # Apple Silicon
    device = get_device()
    print("device:", device)
    """),
    md("""
    ## 1. Data
    """),
    code("""
    pipeline = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    train_ds = torchvision.datasets.FashionMNIST("../data", train=True,  download=True, transform=pipeline)
    test_ds  = torchvision.datasets.FashionMNIST("../data", train=False, download=True, transform=pipeline)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)
    """),
    md("""
    ## 2. Model

    Two conv blocks, then a small classifier head. We keep it simple by using
    `nn.Sequential` — a straight-line stack of layers, no `forward()` method
    needed. Perfect for a first CNN.

    Use the shape rule to predict output sizes *before* you run the cell.
    """),
    code("""
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 28 -> 14
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 14 -> 7
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    summary(model, input_size=(1, 1, 28, 28), device=device)  # (batch, C, H, W)
    """),
    md("""
    ## 3. Sanity-check the forward pass

    Always run one batch through the model **before** training. Catches every
    shape bug in 5 seconds.
    """),
    code("""
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)
    print("input:", xb.shape, "-> logits:", logits.shape)
    print("loss on init:", F.cross_entropy(logits, yb).item())
    """),
    md("""
    ## 4. The training loop

    Read each line. No magic.
    """),
    code("""
    def train_one_epoch(model, loader, optimizer, device):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in tqdm(loader, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += xb.size(0)
        return loss_sum / total, correct / total


    @torch.no_grad()
    def evaluate(model, loader, device):
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += xb.size(0)
        return loss_sum / total, correct / total
    """),
    code("""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = 3

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  "
              f"|  test loss {te_loss:.3f} acc {te_acc:.3f}")
    """),
    md("""
    ## 5. Look at the mistakes

    Always inspect failures. They tell you *what* the model finds hard, which is
    much more informative than the aggregate metric.
    """),
    code("""
    model.eval()
    xb, yb = next(iter(test_loader))
    with torch.no_grad():
        preds = model(xb.to(device)).argmax(1).cpu()
    wrong = (preds != yb).nonzero(as_tuple=True)[0][:16]
    show_grid(
        xb[wrong],
        titles=[f"{test_ds.classes[yb[i]]}->{test_ds.classes[preds[i]]}" for i in wrong],
        cols=8,
    )
    """),
    md("""
    ## Top-5 first-CNN bugs (memorize)

    1. Forgot `.to(device)` for either the model or the batch.
    2. Forgot `optimizer.zero_grad()` — gradients accumulate, training looks broken.
    3. Forgot `model.eval()` during evaluation — BatchNorm/Dropout misbehave.
    4. Wrong shape after `Flatten` — the `Linear` layer size doesn't match.
    5. Used the same transform pipeline for train and test, including augmentations.

    ## You did it

    You trained a CNN from scratch. Day 2 picks up here: deeper architectures,
    transfer learning, and explainability.

    > **Next notebook (05):** we'll rewrite this same model as an `nn.Module`
    > subclass so we can branch the forward pass, expose intermediate features,
    > and attach hooks (needed for transfer learning and Grad-CAM).
    """),
]


# --------------------------------------------------------------------------- #
# Day 2 / Day 3 — skeletons only for now. Flesh out after Day 1 dry-run.      #
# --------------------------------------------------------------------------- #

def skeleton(title: str, day: str, intro: str, sections: list[str]) -> list[dict]:
    cells = [md(f"# {title}\n\n**{day}**\n\n{intro}\n\n> ⚠️ Skeleton — content to be filled in after Day 1 dry-run.")]
    for s in sections:
        cells.append(md(f"## {s}"))
        cells.append(code(f"# TODO: {s}"))
    return cells


N05 = [
    md(r"""
    # 05 — Training a CNN, properly

    **Day 2 · Notebook 1 of 3**

    Take the Day 1 model further: a real `nn.Module`, better architecture,
    augmentations, learning-rate scheduling, and a clean train/val/test split.

    ## Objectives

    - Rewrite yesterday's `nn.Sequential` model as an `nn.Module` subclass
      and understand *why* you'd bother.
    - Add BatchNorm and global average pooling.
    - Use augmentations that actually help on FashionMNIST/CIFAR.
    - Wire in a learning-rate schedule and checkpointing.
    """),
    md(r"""
    ## 1. From `nn.Sequential` to `nn.Module`

    Yesterday's model was a flat stack:

    ```python
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    ```

    `nn.Sequential` is great until you need to do *anything* non-linear in the
    control flow: a skip connection, two heads, returning intermediate features
    for Grad-CAM, conditional logic. For those you write an `nn.Module` subclass
    and put the wiring in `forward()`.

    **Mental model**

    - `__init__` declares the **layers** (the things with learnable weights).
    - `forward(x)` describes the **dataflow** — how an input becomes an output.

    PyTorch tracks `self.<layer>` attributes automatically so
    `model.parameters()`, `.to(device)`, and `state_dict()` all just work.
    """),
    code("""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class SmallCNN(nn.Module):
        def __init__(self, n_classes: int = 10):
            super().__init__()
            # Conv block 1
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            # Conv block 2
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            # Classifier head
            self.fc1 = nn.Linear(32 * 7 * 7, 64)
            self.fc2 = nn.Linear(64, n_classes)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28 -> 14
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14 -> 7
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)


    model = SmallCNN()
    print(model)
    """),
    md(r"""
    Same architecture as Day 1, same parameter count — but now you have a
    `forward()` you can edit. For example, returning the feature map *before*
    the classifier head (we'll need this for Grad-CAM in notebook 07):

    ```python
    def forward(self, x, return_features: bool = False):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        feats = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (N, 32, 7, 7)
        out = self.fc2(F.relu(self.fc1(feats.flatten(1))))
        return (out, feats) if return_features else out
    ```

    You can't do that with `nn.Sequential`.
    """),
    md(r"""
    ## 2. Split the data: train / val / test

    Yesterday we trained on `train_ds` and reported accuracy on `test_ds`.
    That's fine for a one-shot sanity run — but as soon as you start *tuning*
    (more epochs? augmentations? a different LR?) you're using the test set to
    make decisions. That leaks information and your reported accuracy becomes
    optimistic.

    The standard fix: carve a **validation** set out of the training data and
    keep the test set sealed until the very end of the notebook.

    - **train** → gradient updates
    - **val** → watch during training, pick the best checkpoint
    - **test** → look at *once*, at the end
    """),
    code("""
    from torch.utils.data import DataLoader, random_split
    import torchvision
    from torchvision import transforms as T

    from cvcourse import get_device

    device = get_device()
    print("device:", device)

    pipeline = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    full_train = torchvision.datasets.FashionMNIST(
        "../data", train=True, download=True, transform=pipeline,
    )
    test_ds = torchvision.datasets.FashionMNIST(
        "../data", train=False, download=True, transform=pipeline,
    )

    # 54k / 6k split, deterministic seed so val is reproducible.
    train_ds, val_ds = random_split(
        full_train, [54_000, 6_000],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")
    """),
    md(r"""
    ## 3. A real training loop, and the loss curves that come with it

    Same five steps as yesterday (`forward → loss → backward → step → zero_grad`),
    but now wrapped in two helpers and run for *enough* epochs to actually see
    overfitting. After training, we plot train-vs-val loss and accuracy. The
    shape of those curves is the single most useful debugging signal you have.

    **Predict before you run:** at what epoch do you think train and val
    accuracy will start to diverge?
    """),
    code("""
    from tqdm.auto import tqdm


    def train_one_epoch(model, loader, optimizer, device):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in tqdm(loader, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += xb.size(0)
        return loss_sum / total, correct / total


    @torch.no_grad()
    def evaluate(model, loader, device):
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += xb.size(0)
        return loss_sum / total, correct / total


    def fit(model, train_loader, val_loader, optimizer, epochs, device, scheduler=None):
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
            va_loss, va_acc = evaluate(model, val_loader, device)
            history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step()
            print(f"epoch {epoch:2d}  train {tr_loss:.3f}/{tr_acc:.3f}  "
                  f"val {va_loss:.3f}/{va_acc:.3f}  lr {history['lr'][-1]:.4f}")
        return history
    """),
    code("""
    import matplotlib.pyplot as plt

    from cvcourse import plot_history
    """),
    code("""
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history_base = fit(model, train_loader, val_loader, optimizer, epochs=10, device=device)
    plot_history(history_base, title="SmallCNN — no regularization")
    """),
    md(r"""
    Classic overfitting pattern: train loss keeps falling while val loss
    flattens (and eventually climbs). Train accuracy pulls ahead of val
    accuracy. The model is memorizing the training set.

    Sections 4–6 each introduce one tool to push that gap back down.
    """),
    md(r"""
    ## 4. Architecture upgrades: BatchNorm + global average pooling

    Two cheap changes to the architecture:

    - **BatchNorm** after each conv normalizes activations per mini-batch.
      Training is more stable, you can push the learning rate higher, and it
      acts as a mild regularizer.
    - **Global average pooling** replaces `Flatten + big Linear` with a single
      `AdaptiveAvgPool2d(1)`. The feature map collapses to one number per
      channel, and the head becomes a tiny `Linear(C, n_classes)`. Far fewer
      parameters, less prone to overfit.
    """),
    code("""
    from cvcourse import count_params


    class SmallCNN_v2(nn.Module):
        def __init__(self, n_classes: int = 10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.bn1   = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2   = nn.BatchNorm2d(32)
            self.gap   = nn.AdaptiveAvgPool2d(1)           # (N, 32, 7, 7) -> (N, 32, 1, 1)
            self.fc    = nn.Linear(32, n_classes)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
            x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
            x = self.gap(x).flatten(1)
            return self.fc(x)


    print("SmallCNN    params:", count_params(SmallCNN()))
    print("SmallCNN_v2 params:", count_params(SmallCNN_v2()))
    """),
    code("""
    model = SmallCNN_v2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history_v2 = fit(model, train_loader, val_loader, optimizer, epochs=10, device=device)
    plot_history(history_v2, title="SmallCNN_v2 — BatchNorm + GAP")
    """),
    md(r"""
    Fewer parameters, similar (often better) val accuracy, and a smaller
    train/val gap. GAP is doing a lot of the work — the model can no longer
    memorize via a fat fully-connected layer.
    """),
    md(r"""
    ## 5. Augmentations that actually help on FashionMNIST

    Augmentations show the model slightly different versions of each image
    every epoch. The training distribution effectively grows, the model can't
    memorize as easily, and the val/train gap shrinks.

    Pick augmentations that match the **invariances of the task**:

    - `RandomHorizontalFlip()` — a sneaker flipped left/right is still a
      sneaker. ✅
    - `RandomCrop(28, padding=2)` — pad then crop simulates small shifts. ✅
    - `RandomVerticalFlip()` — a bag upside-down is a weird bag. ❌
    - `ColorJitter(...)` — useless on single-channel grayscale. ❌

    The augmentations only apply to the *training* pipeline. Val and test stay
    deterministic.
    """),
    code("""
    train_pipeline = T.Compose([
        T.RandomCrop(28, padding=2),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])

    # Re-wrap the underlying dataset with the augmented pipeline. We rebuild
    # train_ds from scratch so we can change the transform; val_ds keeps the
    # plain pipeline.
    aug_full_train = torchvision.datasets.FashionMNIST(
        "../data", train=True, download=False, transform=train_pipeline,
    )
    aug_train_ds, _ = random_split(
        aug_full_train, [54_000, 6_000],
        generator=torch.Generator().manual_seed(0),  # same seed -> same split
    )
    aug_train_loader = DataLoader(aug_train_ds, batch_size=128, shuffle=True, num_workers=0)
    """),
    code("""
    model = SmallCNN_v2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history_aug = fit(model, aug_train_loader, val_loader, optimizer, epochs=10, device=device)
    plot_history(history_aug, title="SmallCNN_v2 + augmentations")
    """),
    md(r"""
    Train accuracy is *lower* now (the model sees harder, varied inputs) but
    val accuracy is up and the gap is smaller. That's the trade you want.
    """),
    md(r"""
    ## 6. Learning-rate schedules

    A single fixed learning rate is rarely optimal. The classic move: start
    high (fast progress), end low (fine adjustments). **Cosine annealing**
    decays smoothly from the initial LR to (near) zero over the training run;
    `StepLR` drops the LR by a factor every N epochs.

    Couple a schedule with what you already have and you'll typically see one
    or two more accuracy points for free.
    """),
    code("""
    EPOCHS = 10
    model = SmallCNN_v2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history_sched = fit(
        model, aug_train_loader, val_loader, optimizer,
        epochs=EPOCHS, device=device, scheduler=scheduler,
    )
    plot_history(history_sched, title="+ cosine LR schedule")

    plt.figure(figsize=(5, 3))
    plt.plot(range(1, EPOCHS + 1), history_sched["lr"], "-o")
    plt.xlabel("epoch"); plt.ylabel("lr"); plt.title("cosine annealing"); plt.show()
    """),
    md(r"""
    ## 7. Checkpoints — keep your best model, then look at test *once*

    During a long training run the best val accuracy often happens a few
    epochs before the end. Save the weights whenever val improves; at the
    end, reload that checkpoint and report the sealed test accuracy.

    We save `state_dict()` (the tensors only) — not the whole model object.
    `state_dict` checkpoints are portable across code refactors and are the
    standard way to persist a PyTorch model.
    """),
    code("""
    from pathlib import Path


    def fit_with_best(model, train_loader, val_loader, optimizer, epochs, device,
                     scheduler=None, ckpt_path="best.pt"):
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
            va_loss, va_acc = evaluate(model, val_loader, device)
            history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step()
            improved = va_acc > best_val_acc
            if improved:
                best_val_acc = va_acc
                torch.save(model.state_dict(), ckpt_path)
            print(f"epoch {epoch:2d}  train {tr_loss:.3f}/{tr_acc:.3f}  "
                  f"val {va_loss:.3f}/{va_acc:.3f}  "
                  f"{'*saved*' if improved else ''}")
        return history, best_val_acc


    EPOCHS = 10
    ckpt_path = Path("best_smallcnn_v2.pt")

    model = SmallCNN_v2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history_final, best_val_acc = fit_with_best(
        model, aug_train_loader, val_loader, optimizer,
        epochs=EPOCHS, device=device, scheduler=scheduler, ckpt_path=ckpt_path,
    )
    print(f"\\nbest val accuracy: {best_val_acc:.3f}")
    """),
    code("""
    # Reload the best checkpoint into a fresh model — this is what you'd do at
    # inference time, after training has finished.
    final_model = SmallCNN_v2().to(device)
    final_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_loss, test_acc = evaluate(final_model, test_loader, device)
    print(f"sealed test accuracy: {test_acc:.3f}")
    """),
    md(r"""
    That number — and *only* that number — is what you'd report. We touched
    the test set exactly once.

    ## What changed from notebook 04

    | | nb 04 | nb 05 |
    |---|---|---|
    | Model | `nn.Sequential` | `nn.Module` subclass |
    | Splits | train / test | train / **val** / test |
    | Loss curves | print only | plotted, used to spot overfitting |
    | Architecture | plain conv stack | + BatchNorm + GAP |
    | Augmentation | none | `RandomCrop` + `RandomHorizontalFlip` |
    | LR | fixed 1e-3 | cosine-annealed |
    | Best model | last epoch | checkpointed on val improvement |

    > **Next notebook (06):** instead of training a CNN from scratch, we'll
    > start from a pretrained backbone and adapt it — far less data, far more
    > accuracy.
    """),
]

N06 = [
    md(r"""
    # 06 — Transfer learning

    **Day 2 · Notebook 2 of 3**

    The single most useful skill in applied CV: take a tiny dataset, pick a
    pretrained backbone, swap the head, fine-tune. This is how 90% of real
    CV projects start.

    Today we'll classify **ants vs. bees** from 244 training images. From
    scratch, that's hopeless. With a ResNet18 pretrained on ImageNet, we'll
    cross 90% val accuracy in under a minute.

    ## Objectives

    - Load a pretrained model and replace its classifier head.
    - Run **feature extraction** (freeze the backbone, train only the head).
    - Run **fine-tuning** (unfreeze, lower LR for the backbone).
    - Recognize the 3 transfer-learning pitfalls that eat the most time.
    - Swap in any of `timm`'s 1000+ pretrained models with one line.
    """),
    md(r"""
    ## 1. The dataset — Hymenoptera (244 images, 2 classes)

    The canonical "transfer learning works, training from scratch doesn't"
    dataset, from the official PyTorch tutorial. ~50 MB, downloads in seconds.
    """),
    code("""
    import urllib.request, zipfile
    from pathlib import Path

    DATA_DIR = Path("../data/hymenoptera_data")
    if not DATA_DIR.exists():
        url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
        zip_path = Path("../data/hymenoptera_data.zip")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(zip_path.parent)
        zip_path.unlink()
    print("ready:", DATA_DIR)
    print("contents:", sorted(p.name for p in DATA_DIR.iterdir()))
    """),
    code("""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from torchvision.datasets import ImageFolder
    from tqdm.auto import tqdm

    from cvcourse import get_device, show_grid, plot_history

    device = get_device()
    print("device:", device)

    # A plain pipeline for the from-scratch baseline. We'll swap in the
    # pretrained-model preprocessing later.
    plain_pipeline = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    train_ds = ImageFolder(DATA_DIR / "train", transform=plain_pipeline)
    val_ds   = ImageFolder(DATA_DIR / "val",   transform=plain_pipeline)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    class_names = train_ds.classes
    print(f"train {len(train_ds)}  val {len(val_ds)}  classes {class_names}")
    """),
    code("""
    # One batch — make sure the data actually looks like what you think it does.
    xb, yb = next(iter(train_loader))
    show_grid(xb[:8], titles=[class_names[y] for y in yb[:8]], cols=4)
    """),
    md(r"""
    ## 2. Train/eval helpers (same as nb 05)

    Notebooks are independent — students might land here directly — so we
    redefine the loop helpers once. Nothing new vs. notebook 05.
    """),
    code("""
    def train_one_epoch(model, loader, optimizer, device):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in tqdm(loader, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += xb.size(0)
        return loss_sum / total, correct / total


    @torch.no_grad()
    def evaluate(model, loader, device):
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += xb.size(0)
        return loss_sum / total, correct / total


    def fit(model, train_loader, val_loader, optimizer, epochs, device):
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
            va_loss, va_acc = evaluate(model, val_loader, device)
            history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            print(f"epoch {epoch:2d}  train {tr_loss:.3f}/{tr_acc:.3f}  "
                  f"val {va_loss:.3f}/{va_acc:.3f}")
        return history
    """),
    md(r"""
    ## 3. Baseline — train from scratch (and watch it fail)

    Before pulling out pretrained weights, see what 244 images can do on their
    own. We use the same `SmallCNN_v2` shape from notebook 05, adapted to
    3-channel 224×224 inputs.
    """),
    code("""
    class TinyCNN(nn.Module):
        def __init__(self, n_classes: int = 2):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1   = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2   = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn3   = nn.BatchNorm2d(64)
            self.gap   = nn.AdaptiveAvgPool2d(1)
            self.fc    = nn.Linear(64, n_classes)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
            x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
            x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
            x = self.gap(x).flatten(1)
            return self.fc(x)


    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history_scratch = fit(model, train_loader, val_loader, optimizer, epochs=5, device=device)
    plot_history(history_scratch, title="TinyCNN — from scratch on 244 images")
    """),
    md(r"""
    Val accuracy hovers near chance. 244 images is nowhere near enough for a
    network to learn what an edge, a texture, or a wing is from pixels alone.

    A network trained on ImageNet has already learned all of that. We just
    need to teach it the *last* step — "of these features, which mean ant and
    which mean bee?".
    """),
    md(r"""
    ## 4. Load a pretrained ResNet18

    `torchvision.models` ships dozens of architectures. The modern API hands
    you the weights enum *and* the exact preprocessing the model was trained
    with — so you can't get normalization wrong.
    """),
    code("""
    from torchvision.models import resnet18, ResNet18_Weights
    from torchinfo import summary

    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()   # the correct resize + normalize for this model
    print(preprocess)

    model = resnet18(weights=weights).to(device)
    summary(model, input_size=(1, 3, 224, 224), device=device)
    """),
    md(r"""
    Two things to notice:

    - **~11M parameters** — vs. ~10k in our TinyCNN. All of them already trained.
    - The final layer is **`fc: Linear(512, 1000)`** — it predicts ImageNet's
      1000 classes. We have 2. The head has to go.
    """),
    md(r"""
    ## 5. Swap the head, swap the preprocessing

    Replace `fc` with a fresh `Linear(512, 2)`. The new layer is randomly
    initialized; everything below it keeps its ImageNet weights.

    Then rebuild the loaders with `preprocess` so inputs match what the model
    was trained on (ImageNet mean/std, 224×224 center crop).
    """),
    code("""
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
    print(model.fc)

    train_ds_pre = ImageFolder(DATA_DIR / "train", transform=preprocess)
    val_ds_pre   = ImageFolder(DATA_DIR / "val",   transform=preprocess)
    train_loader_pre = DataLoader(train_ds_pre, batch_size=32, shuffle=True,  num_workers=0)
    val_loader_pre   = DataLoader(val_ds_pre,   batch_size=32, shuffle=False, num_workers=0)
    """),
    md(r"""
    ## 6. Strategy A — Feature extraction (frozen backbone)

    Freeze every parameter, then unfreeze only the new head. Now
    `model.parameters()` still walks the whole network, but `requires_grad`
    on all but `fc` is `False`, so `optimizer.step()` only updates the head.

    **Why this is fast:** the backbone is computed once per batch (the
    forward pass) but no gradients flow through it.
    """),
    code("""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,}  /  total: {total:,}  ({trainable/total:.2%})")

    # Only pass the params we actually want to update.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    history_feat = fit(model, train_loader_pre, val_loader_pre, optimizer, epochs=5, device=device)
    plot_history(history_feat, title="ResNet18 — feature extraction (frozen backbone)")
    """),
    md(r"""
    > 1000 trainable parameters, ~5 epochs, **>90% val accuracy**. Compare to
    > the from-scratch run that couldn't beat chance.

    ## 7. Strategy B — Fine-tuning (unfreeze, two LRs)

    Feature extraction treats the pretrained features as fixed. Fine-tuning
    lets them *adapt* to your dataset — usually another 2–5 points of
    accuracy at the cost of a longer training run.

    The realistic recipe is **per-group learning rates**: the new head needs a
    big LR (it's random), the pretrained backbone needs a small one (its
    features are precious — too-large gradients erase them).
    """),
    code("""
    for p in model.parameters():
        p.requires_grad = True

    # Two param groups: head gets 1e-3, backbone gets 1e-4.
    head_params     = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    optimizer = torch.optim.AdamW([
        {"params": head_params,     "lr": 1e-3},
        {"params": backbone_params, "lr": 1e-4},
    ])

    history_ft = fit(model, train_loader_pre, val_loader_pre, optimizer, epochs=5, device=device)
    plot_history(history_ft, title="ResNet18 — fine-tuning (unfrozen, two LRs)")
    """),
    md(r"""
    ## 8. Compare all three runs on one chart
    """),
    code("""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for name, h in [
        ("from scratch",   history_scratch),
        ("feature extr.",  history_feat),
        ("fine-tuned",     history_ft),
    ]:
        plt.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], "-o", label=name)
    plt.xlabel("epoch"); plt.ylabel("val accuracy"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title("Hymenoptera — 244 train images"); plt.show()
    """),
    md(r"""
    ## 9. The pitfalls that eat the most time

    1. **Wrong normalization.** Torchvision pretrained models expect ImageNet
       mean/std. If you feed them your own dataset's stats (or raw `[0, 1]`),
       accuracy silently degrades. Always use `weights.transforms()` — it
       bakes in the right preprocessing for that specific checkpoint.

    2. **Wrong input size.** Most ImageNet models want **224×224**. Some
       (Inception, EfficientNet-B7) want 299, 600, or other sizes. Again,
       `weights.transforms()` handles it.

    3. **Backbone LR too high.** A pretrained backbone is the most valuable
       part of your model. Use `≤ 1e-4` on it (often `1e-5`). The head, being
       random, needs `1e-3` or higher. Per-group LRs are the standard tool.

    4. **BatchNorm running stats drifting on tiny batches.** When fine-tuning
       on a very small dataset with small batches, BN's running mean/var can
       wander away from the ImageNet statistics. Mitigation: keep the
       backbone in `.eval()` mode during the feature-extraction phase, or
       freeze BN layers explicitly.

    5. **Forgot to swap the head.** The new task has `K` classes; the
       pretrained `fc` predicts 1000. The error you'd get on a loss against
       2-class labels is `IndexError` or a bizarre loss value — easy to
       mistake for something deeper. Always re-print the model after the
       swap.
    """),
    md(r"""
    ## 10. `timm` — 1000+ pretrained models in one line

    [`timm`](https://github.com/huggingface/pytorch-image-models) ships
    practically every state-of-the-art image backbone (EfficientNet, ConvNeXt,
    Swin, BEiT, DINOv2, ...) behind a single `create_model` call. The same
    freeze/unfreeze recipe you just learned applies unchanged.

    `num_classes=` even handles the head swap for you.
    """),
    code("""
    import timm

    model_timm = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model_timm = model_timm.to(device)

    # timm models carry their own preprocessing recipe — use it.
    cfg = timm.data.resolve_model_data_config(model_timm)
    timm_preprocess = timm.data.create_transform(**cfg, is_training=False)
    print(timm_preprocess)

    # Quick sanity check — one forward pass on a real batch.
    xb, _ = next(iter(val_loader_pre))
    with torch.no_grad():
        logits = model_timm(xb.to(device))
    print("logits shape:", logits.shape)
    """),
    md(r"""
    To train it: same `fit(...)` call, same freeze/unfreeze pattern. Pick the
    catalog with `timm.list_models(pretrained=True)`.

    ## Wrap-up

    You took 244 images and a pretrained ResNet18 and built a >90% classifier
    in under a minute. That's the recipe for the vast majority of applied CV
    projects — not "design a new architecture", but "pick a backbone, swap
    the head, fine-tune".

    > **Next notebook (07):** Grad-CAM — *see* which pixels the model used to
    > decide "ant" vs "bee", and learn to spot when it's right for the
    > wrong reasons.
    """),
]

N07 = [
    md(r"""
    # 07 — Explaining what the CNN looks at

    **Day 2 · Notebook 3 of 3**

    Your fine-tuned ResNet18 says "bee" with 99% confidence. *Why?*

    Three techniques, increasing sophistication, decreasing magic:

    1. **Saliency** — the raw gradient of the score w.r.t. the input pixels.
    2. **Grad-CAM** — the workhorse. Weighted last-conv activations, ~20 lines.
    3. **Embedding viz** — t-SNE of the penultimate features to *see* class
       separation.

    Then the punchline: a model that's right for the *wrong* reason.

    ## Objectives

    - Use forward and backward **hooks** to capture activations and gradients.
    - Implement **Grad-CAM** from scratch (no `captum`, no `pytorch-grad-cam`).
    - Generate **saliency** maps with one autograd call.
    - Project the **penultimate embeddings** to 2D with t-SNE.
    - Spot a case where the model is **right for the wrong reason**.
    """),
    md(r"""
    ## 1. Reload the fine-tuned model

    N07 stands alone — we re-fine-tune ResNet18 on Hymenoptera in a few epochs
    so this notebook can be run independently. Same recipe as N06.
    """),
    code("""
    import urllib.request, zipfile
    from pathlib import Path

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import resnet18, ResNet18_Weights
    from tqdm.auto import tqdm

    from cvcourse import get_device, show_grid

    device = get_device()
    print("device:", device)

    DATA_DIR = Path("../data/hymenoptera_data")
    if not DATA_DIR.exists():
        url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
        zip_path = Path("../data/hymenoptera_data.zip")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(zip_path.parent)
        zip_path.unlink()

    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    train_ds = ImageFolder(DATA_DIR / "train", transform=preprocess)
    val_ds   = ImageFolder(DATA_DIR / "val",   transform=preprocess)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
    class_names = train_ds.classes
    print(f"train {len(train_ds)}  val {len(val_ds)}  classes {class_names}")
    """),
    code("""
    # Build ResNet18, swap head, fine-tune ~3 epochs.
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.fc.parameters(),                                   "lr": 1e-3},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": 1e-4},
    ])

    for epoch in range(1, 4):
        model.train()
        for xb, yb in tqdm(train_loader, leave=False, desc=f"ep{epoch}"):
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total   += xb.size(0)
        print(f"epoch {epoch}  val acc {correct/total:.3f}")
    """),
    md(r"""
    ## 2. Forward hooks — recording activations

    PyTorch doesn't store intermediate activations by default. A **hook** is a
    function you attach to a module that runs every time the module's
    forward (or backward) is called. Use it to peek inside without rewriting
    the model.

    For Grad-CAM we need the activations of the **last conv block**. On
    ResNet18 that's `layer4` — output shape `(B, 512, 7, 7)` for 224×224 input.
    """),
    code("""
    activations = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    handle = model.layer4.register_forward_hook(save_activation("layer4"))

    # One image, one forward pass.
    xb, yb = next(iter(val_loader))
    xb = xb.to(device)
    with torch.no_grad():
        logits = model(xb[:1])
    print("layer4 activations:", activations["layer4"].shape)  # (1, 512, 7, 7)
    handle.remove()  # always remove hooks when done
    """),
    md(r"""
    Bridge: now we also need gradients of the class score w.r.t. those
    activations. Same idea, **backward** hook.

    ## 3. Grad-CAM from scratch

    One-paragraph intuition: each of the 512 channels in `layer4` is a learned
    feature detector. The gradient of `score["bee"]` w.r.t. each channel tells
    us how *much* that channel contributed. **Average the gradient over the
    spatial dims to get one weight per channel**, take a weighted sum of the
    channel feature maps, ReLU it (we only care about positive contributions),
    and upsample to the input size. Done.
    """),
    code("""
    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.acts = None
            self.grads = None
            self.h1 = target_layer.register_forward_hook(self._save_acts)
            self.h2 = target_layer.register_full_backward_hook(self._save_grads)

        def _save_acts(self, module, input, output):
            self.acts = output.detach()

        def _save_grads(self, module, grad_input, grad_output):
            self.grads = grad_output[0].detach()

        def __call__(self, x, class_idx=None):
            self.model.eval()
            self.model.zero_grad()
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            score = logits[0, class_idx]
            score.backward()

            # weights: global-avg-pool the gradients -> one scalar per channel
            weights = self.grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
            cam = (weights * self.acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam, class_idx

        def close(self):
            self.h1.remove(); self.h2.remove()

    gradcam = GradCAM(model, model.layer4)
    """),
    code("""
    import numpy as np
    import matplotlib.pyplot as plt

    # ImageNet normalization (what preprocess applied) — invert for display.
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

    def denorm(t):
        # t: (3, H, W) tensor, normalized. -> (H, W, 3) numpy in [0, 1].
        img = t.cpu().numpy().transpose(1, 2, 0)
        img = img * IMAGENET_STD + IMAGENET_MEAN
        return np.clip(img, 0, 1)

    def overlay(img, cam, alpha=0.5):
        plt.imshow(img)
        plt.imshow(cam, cmap="jet", alpha=alpha)
        plt.axis("off")
    """),
    code("""
    # Pick a few val images and visualize Grad-CAM for the predicted class.
    xb, yb = next(iter(val_loader))
    n = 4
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        x = xb[i:i+1].to(device)
        cam, pred_idx = gradcam(x, class_idx=None)
        img = denorm(xb[i])
        true_name = class_names[yb[i].item()]
        pred_name = class_names[pred_idx]
        mark = "OK" if pred_idx == yb[i].item() else "WRONG"

        plt.sca(axes[0, i]); plt.imshow(img); plt.axis("off")
        plt.title(f"true: {true_name}")
        plt.sca(axes[1, i]); overlay(img, cam)
        plt.title(f"pred: {pred_name} [{mark}]")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Where does the heatmap land? For a correctly-classified bee it usually
    sits on the body or wings; for an ant, on the body and legs. When it lands
    on the *background* — flower petals, grass, dirt — that's a warning sign
    (we'll come back to that in §6).

    ## 4. Saliency — the simplest gradient

    Grad-CAM smooths over a 7×7 grid. Saliency is the opposite extreme:
    gradient of the class score w.r.t. **each input pixel**. Noisy but
    pixel-accurate.

    ```
    grad = ∂ logits[class] / ∂ image
    saliency = grad.abs().max(over channels)
    ```
    """),
    code("""
    def saliency(model, x, class_idx=None):
        model.eval()
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        model.zero_grad()
        logits[0, class_idx].backward()
        # x.grad: (1, 3, H, W) -> max over channels -> (H, W)
        sal = x.grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        return sal, class_idx

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for i in range(n):
        x = xb[i:i+1].to(device)
        sal, _      = saliency(model, x)
        cam, pred   = gradcam(x)
        img = denorm(xb[i])

        plt.sca(axes[0, i]); plt.imshow(img); plt.axis("off")
        plt.title(class_names[yb[i].item()])
        plt.sca(axes[1, i]); plt.imshow(sal, cmap="hot"); plt.axis("off")
        plt.title("saliency")
        plt.sca(axes[2, i]); overlay(img, cam); plt.title("Grad-CAM")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Saliency is *where* at the pixel level — Grad-CAM is *where* at the
    feature-map level. Saliency tends to look like edge noise; Grad-CAM tends
    to highlight semantically meaningful blobs. In practice Grad-CAM is the
    one you'll reach for.

    ## 5. Embedding visualization

    Drop the classifier head. The penultimate vector (512-d for ResNet18,
    after `avgpool`) is what the model thinks the *image* is — independent of
    the final ant/bee decision. If those vectors cluster cleanly by class,
    the model has learned a useful representation.
    """),
    code("""
    from sklearn.manifold import TSNE

    feats = {}
    def grab_feats(module, input, output):
        feats["x"] = output.detach()

    h = model.avgpool.register_forward_hook(grab_feats)

    all_feats, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb_, yb_ in val_loader:
            _ = model(xb_.to(device))
            all_feats.append(feats["x"].flatten(1).cpu())
            all_labels.append(yb_)
    h.remove()

    X = torch.cat(all_feats).numpy()    # (N, 512)
    y = torch.cat(all_labels).numpy()
    print("embeddings:", X.shape)

    Z = TSNE(n_components=2, perplexity=10, init="pca", random_state=0).fit_transform(X)

    plt.figure(figsize=(6, 5))
    for c, name in enumerate(class_names):
        plt.scatter(Z[y == c, 0], Z[y == c, 1], label=name, s=30, alpha=0.7)
    plt.legend(); plt.title("t-SNE of penultimate features (val set)"); plt.show()
    """),
    md(r"""
    Two clean clusters → the model has learned an ant-vs-bee axis in feature
    space. The classifier head is just drawing a line through it.

    ## 6. Right for the wrong reason

    Accuracy hides a lot. On Hymenoptera, the two classes have very different
    *backgrounds* — bees often on flowers, ants often on dirt or leaves. A
    model can shortcut: classify the **background**, not the insect.

    Let's scan the val set and find images where the model got it right but
    Grad-CAM lit up *outside* a centered crop — a rough proxy for "looking at
    the background".
    """),
    code("""
    # Crude heuristic: how much of Grad-CAM's mass sits inside the center 50% box?
    def center_mass_fraction(cam, frac=0.5):
        h, w = cam.shape
        ph, pw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
        center = cam[ph:h-ph, pw:w-pw]
        return center.sum() / (cam.sum() + 1e-8)

    suspects = []
    model.eval()
    for xb_, yb_ in val_loader:
        for i in range(xb_.size(0)):
            x = xb_[i:i+1].to(device)
            cam, pred = gradcam(x)
            if pred == yb_[i].item():            # correct prediction
                cmf = center_mass_fraction(cam)
                if cmf < 0.35:                    # heatmap mostly off-center
                    suspects.append((xb_[i].clone(), yb_[i].item(), pred, cam, cmf))
        if len(suspects) >= 4:
            break

    print(f"found {len(suspects)} suspicious 'correct' predictions")
    fig, axes = plt.subplots(1, min(4, len(suspects)), figsize=(3 * min(4, len(suspects)), 3))
    if len(suspects) == 1:
        axes = [axes]
    for ax, (img_t, true_c, pred_c, cam, cmf) in zip(axes, suspects[:4]):
        plt.sca(ax); overlay(denorm(img_t), cam)
        plt.title(f"{class_names[pred_c]}  center={cmf:.2f}")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    Look at those heatmaps. The model *predicted correctly* but the
    explanation says it looked at petals, leaves, or dirt — not the insect.
    That's a **shortcut**: the model learned a background prior, not the
    object. On out-of-distribution images (a bee on concrete, an ant on a
    flower) it would likely fail.

    Famous examples of the same failure mode:

    - **Husky vs. wolf** — classifier learned to detect *snow*, not the animal.
    - **Pneumonia X-ray** — model learned to read the hospital marker tag,
      because each hospital's prevalence differed.
    - **Skin lesion benign/malignant** — model learned that dermatologists
      put a **ruler** next to suspicious lesions.

    Accuracy alone wouldn't have caught any of those. Grad-CAM would have.

    ## Wrap-up

    Three views of one model: pixels (saliency), regions (Grad-CAM), and
    representations (t-SNE). They don't tell you *what* to fix — they tell you
    *where* to look. Use them on every model you ship.

    > **Next notebook (08 — Day 3):** **attention**. Where Grad-CAM bolts
    > "where is the model looking" on after training, attention bakes it
    > into the architecture itself. We'll build it from scratch on toy
    > tensors before touching ViT.
    """),
    code("""
    # Clean up hooks.
    gradcam.close()
    """),
]

N08 = [
    md(r"""
    # 08 — Introduction to ViT

    **Day 3 · Notebook 1 of 3**

    CNNs were the only show in town for image classification until 2020. Then
    the Vision Transformer (ViT, Dosovitskiy et al.) showed that pure
    attention — no convolutions — could match or beat ResNets *given enough
    data*.

    Today we build a ViT from scratch in PyTorch and train it on a
    cats-vs-dogs binary task carved out of CIFAR-10. The architecture is the
    same one running inside CLIP, DINOv2, SAM, and the visual towers of every
    multimodal model.

    ## Objectives

    - Cut an image into **patches** and embed each as a token.
    - Add **positional embeddings** so the model knows patch order.
    - Implement **scaled dot-product attention** on a toy tensor (the
      ~6 lines at the heart of every transformer).
    - Stack **transformer encoder blocks** with LayerNorm → Attention → MLP.
    - Train the ViT on CIFAR cats-vs-dogs and compare honestly to a small CNN.
    - Understand why a tiny ViT *loses* to a tiny CNN — and what changes that.
    """),
    md(r"""
    ## 1. Data — CIFAR-10 filtered to cats vs dogs

    The original TF notebook used `tensorflow_datasets.cats_vs_dogs`. PyTorch
    doesn't ship that one, but `torchvision` does ship CIFAR-10 — and
    classes 3 (cat) and 5 (dog) are exactly what we want. ~10k train and 2k
    test images after filtering. Plenty to train a 15k-parameter model.

    We resize to **72×72** so 8×8 patches yield 9×9 = 81 tokens — same
    geometry as the TF reference.
    """),
    code("""
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
    from torchvision.datasets import CIFAR10
    from torchinfo import summary
    from tqdm.auto import tqdm

    from cvcourse import get_device, show_grid, plot_history

    device = get_device()
    print("device:", device)
    """),
    code("""
    IMG_SIZE = 72

    pipeline = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])

    cifar_train = CIFAR10("../data", train=True,  download=True, transform=pipeline)
    cifar_test  = CIFAR10("../data", train=False, download=True, transform=pipeline)

    # CIFAR-10 class indices: cat=3, dog=5. Remap to {0: cat, 1: dog}.
    CAT, DOG = 3, 5
    LABEL_MAP = {CAT: 0, DOG: 1}
    class_names = ["cat", "dog"]

    class CatsVsDogs(Dataset):
        def __init__(self, base):
            self.base = base
            self.indices = [i for i, (_, y) in enumerate(base) if y in LABEL_MAP]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            x, y = self.base[self.indices[i]]
            return x, LABEL_MAP[y]

    train_ds = CatsVsDogs(cifar_train)
    val_ds   = CatsVsDogs(cifar_test)
    print(f"train {len(train_ds)}  val {len(val_ds)}")
    """),
    code("""
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

    xb, yb = next(iter(train_loader))
    show_grid(xb[:8], titles=[class_names[y] for y in yb[:8]], cols=4)
    """),
    md(r"""
    ## 2. Train/eval helpers (same as nb 05/06/07)

    Re-defined so this notebook stands alone.
    """),
    code("""
    def train_one_epoch(model, loader, optimizer, device):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in tqdm(loader, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += xb.size(0)
        return loss_sum / total, correct / total


    @torch.no_grad()
    def evaluate(model, loader, device):
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += xb.size(0)
        return loss_sum / total, correct / total


    def fit(model, train_loader, val_loader, optimizer, epochs, device):
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
            va_loss, va_acc = evaluate(model, val_loader, device)
            history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            print(f"epoch {epoch:2d}  train {tr_loss:.3f}/{tr_acc:.3f}  "
                  f"val {va_loss:.3f}/{va_acc:.3f}")
        return history
    """),
    md(r"""
    ## 3. CNN baseline

    Before reaching for transformers, set the bar with a tiny CNN. Two
    Conv→ReLU→Pool blocks, flatten, one Linear. ~18k parameters. This is the
    "default" any reasonable engineer would try first on a small dataset.
    """),
    code("""
    cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),   # 72 -> 35
        nn.Conv2d(32, 32, 3), nn.ReLU(), nn.MaxPool2d(2),  # 35 -> 16
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 2),
    ).to(device)

    summary(cnn, input_size=(1, 3, IMG_SIZE, IMG_SIZE), device=device)
    """),
    code("""
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    history_cnn = fit(cnn, train_loader, val_loader, optimizer, epochs=5, device=device)
    plot_history(history_cnn, title="Tiny CNN — cats vs dogs (CIFAR-10 subset)")
    """),
    md(r"""
    Tiny CNN, low epoch count, decent val accuracy. The CNN's convolutions
    *hard-code* two priors — translation equivariance and locality — that
    a transformer has to *learn from data*. Hold on to that intuition.

    # Vision Transformer

    ## 4. Patches — slice the image into a sequence

    An image isn't a sequence — until you cut it into one. For a 72×72 RGB
    image and patch size 8, we get a **9×9 = 81 patches** of shape `(3, 8, 8)`.
    Flattened, that's 81 tokens of dimension 3·8·8 = 192.

    **The PyTorch trick.** Instead of explicitly extracting patches and then
    projecting them with a `Linear` (the TF notebook's two-step), do both
    operations in one `Conv2d` with `kernel_size = stride = patch_size`. This
    is what timm, torchvision, and the original Dosovitskiy paper all do
    (figure 1, "linear projection of flattened patches" is *literally* a
    strided Conv2d).
    """),
    code("""
    class PatchEmbedding(nn.Module):
        def __init__(self, in_channels: int, patch_size: int, num_patches: int, embed_dim: int):
            super().__init__()
            self.num_patches = num_patches
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.pos_embedding = nn.Embedding(num_patches, embed_dim)
            # register a buffer of positions so .to(device) carries it along
            self.register_buffer("positions", torch.arange(num_patches))

        def forward(self, x):
            # x: (B, 3, H, W) -> (B, D, H/p, W/p)
            x = self.proj(x)
            # -> (B, D, N) -> (B, N, D)
            x = x.flatten(2).transpose(1, 2)
            # add learned positional embedding to every batch item
            x = x + self.pos_embedding(self.positions)
            return x

    patch_size  = 8
    num_patches = (IMG_SIZE // patch_size) ** 2   # 9*9 = 81
    embed_dim   = 16

    pe = PatchEmbedding(3, patch_size, num_patches, embed_dim).to(device)
    xb, _ = next(iter(train_loader))
    out = pe(xb.to(device))
    print(f"in {tuple(xb.shape)} -> tokens {tuple(out.shape)}")  # (B, 81, 16)
    """),
    code("""
    # Visualize the 81 patches of a single image.
    import matplotlib.pyplot as plt

    img = xb[0]  # (3, 72, 72)
    n = IMG_SIZE // patch_size
    fig, axes = plt.subplots(n, n, figsize=(6, 6))
    for i in range(n):
        for j in range(n):
            patch = img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            axes[i, j].imshow(patch.permute(1, 2, 0))
            axes[i, j].axis("off")
    plt.suptitle("81 patches (9x9 grid of 8x8 patches)")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 5. Attention — the 5-minute explainer

    Each of the 81 tokens asks every other token *"how relevant are you to
    me?"* and produces a weighted sum of their **values**. Three learned
    projections do all the work:

    - **Query** Q — what this token is looking for.
    - **Key** K — what each token *advertises* about itself.
    - **Value** V — what each token *carries* if attended to.

    The attention weights are softmax over `Q · Kᵀ / √d`. The output is those
    weights times V. That's it. Walk through it once on a toy tensor.
    """),
    code("""
    # Toy: 1 batch, 4 tokens, 8-d embeddings.
    torch.manual_seed(0)
    B, N, D = 1, 4, 8
    x = torch.randn(B, N, D)

    W_q = torch.randn(D, D)
    W_k = torch.randn(D, D)
    W_v = torch.randn(D, D)

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    scores  = Q @ K.transpose(-1, -2) / math.sqrt(D)   # (B, N, N)
    weights = scores.softmax(dim=-1)                   # rows sum to 1
    out     = weights @ V                              # (B, N, D)

    print("attention weights (rows sum to 1):")
    print(weights.squeeze(0).round(decimals=2))
    print("output shape:", out.shape)
    """),
    code("""
    plt.imshow(weights.squeeze(0).detach(), cmap="viridis")
    plt.colorbar()
    plt.title("Toy attention matrix (4 tokens)")
    plt.xlabel("attended-to token"); plt.ylabel("query token")
    plt.show()
    """),
    md(r"""
    That's **single-head** attention. **Multi-head** attention runs `h` of
    these in parallel on lower-dimensional projections (`D / h` per head) and
    concatenates the outputs. Each head can learn a different relationship
    pattern — e.g. one head specializes in "look at neighbors", another in
    "look at the center". `nn.MultiheadAttention` is the batteries-included
    PyTorch version.

    ## 6. The transformer encoder block

    The standard *pre-norm* recipe (more stable than the post-norm variant
    in the original "Attention Is All You Need" paper, and what every modern
    ViT uses):

    ```
    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
    ```

    > Sidebar: the TF reference notebook has a `# BUG FIX!` comment on the
    > residual — it's a real bug. If you forget to add the residual back in,
    > gradients can't flow past the attention layer and the network refuses
    > to train. Residual connections are non-negotiable in transformers.
    """),
    code("""
    class TransformerEncoder(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int, mlp_hidden: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.ln2 = nn.LayerNorm(embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, embed_dim),
            )

        def forward(self, x):
            # pre-norm + self-attention + residual
            h = self.ln1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn_out
            # pre-norm + MLP + residual
            x = x + self.mlp(self.ln2(x))
            return x
    """),
    md(r"""
    ## 7. Assemble the ViT
    """),
    code("""
    class ViT(nn.Module):
        def __init__(self, image_size=72, patch_size=8, in_channels=3,
                     embed_dim=16, num_heads=4, num_encoders=2, num_classes=2):
            super().__init__()
            num_patches = (image_size // patch_size) ** 2
            self.patch_embed = PatchEmbedding(in_channels, patch_size, num_patches, embed_dim)
            self.encoders = nn.Sequential(*[
                TransformerEncoder(embed_dim, num_heads, mlp_hidden=embed_dim * 2)
                for _ in range(num_encoders)
            ])
            self.head = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            x = self.patch_embed(x)         # (B, N, D)
            x = self.encoders(x)            # (B, N, D)
            x = x.mean(dim=1)               # global average pool over tokens -> (B, D)
            return self.head(x)             # (B, num_classes)

    vit = ViT().to(device)
    summary(vit, input_size=(1, 3, IMG_SIZE, IMG_SIZE), device=device)
    """),
    md(r"""
    ~15k trainable parameters — comparable to the CNN's 18k. Roughly head-to-head
    on capacity, but **all the inductive bias the CNN gets for free, the ViT
    has to learn**.

    ## 8. Train the ViT
    """),
    code("""
    optimizer = torch.optim.Adam(vit.parameters(), lr=1e-3)
    history_vit = fit(vit, train_loader, val_loader, optimizer, epochs=10, device=device)
    plot_history(history_vit, title="Tiny ViT — cats vs dogs (CIFAR-10 subset)")
    """),
    code("""
    # Both runs on one chart.
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(history_cnn["val_acc"]) + 1), history_cnn["val_acc"], "-o", label="tiny CNN")
    plt.plot(range(1, len(history_vit["val_acc"]) + 1), history_vit["val_acc"], "-o", label="tiny ViT")
    plt.xlabel("epoch"); plt.ylabel("val accuracy"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title("CIFAR cats vs dogs — same parameter budget"); plt.show()
    """),
    md(r"""
    ## 9. Why the tiny ViT loses

    On this dataset the CNN wins, and that's the expected result. Three
    reasons:

    1. **Inductive bias.** CNNs hard-code translation equivariance and
       locality — two priors that *are true* about natural images. ViT has
       neither baked in; it must rediscover them from data.

    2. **Data hunger.** The original ViT paper trained on JFT-300M (300M
       proprietary images) to match a ResNet. On 10k images, you're in the
       regime where the CNN's priors are worth more than the ViT's
       flexibility.

    3. **Regularization left on the table.** Real ViTs use augmentation,
       dropout, stochastic depth, mixup, weight decay, and warmup-cosine LR
       schedules. We omitted all of it to keep this notebook legible.

    The take-away of Day 3: for small-data CV, **CNNs are still the default
    choice**. ViTs shine when you have either a lot of data, or — much more
    common — a **pretrained** ViT to fine-tune. That's exactly the recipe
    you'll see in N09.

    ## Wrap-up

    You've built a Vision Transformer from scratch: patchify, embed with
    learned positions, stack `LayerNorm → MultiheadAttention → MLP` blocks,
    pool, classify. The same skeleton runs inside CLIP, DINOv2, SAM, and
    every vision tower of every modern multimodal model.

    > **Next notebook (09):** load a **pretrained** ViT via `timm` and
    > fine-tune it on a small dataset. The transfer-learning recipe from
    > N06, with attention instead of convolutions in the backbone.
    """),
]

N09 = skeleton(
    "09 — Building a tiny Vision Transformer",
    "Day 3 · Notebook 2 of 3",
    "Implement a small ViT end-to-end: patches, positional embeddings, transformer encoder, classifier head. Train on CIFAR10 to see it actually learn.",
    [
        "Patch embedding via a Conv2d trick",
        "Positional embeddings (learned vs sinusoidal)",
        "Transformer encoder block",
        "CLS token and classifier head",
        "Training on CIFAR10 — what to expect",
        "Visualizing attention maps",
    ],
)

N10 = skeleton(
    "10 — CNN vs ViT — when to pick which",
    "Day 3 · Notebook 3 of 3",
    "Head-to-head: compute, parameter count, data efficiency, attention vs feature maps, real-world tradeoffs.",
    [
        "Parameter and FLOP comparison at matched accuracy",
        "Data efficiency — small dataset behavior",
        "Inductive bias: locality vs global context",
        "Visual comparison: GradCAM vs attention rollout",
        "What ships in production today (and why)",
        "When to reach for SAM / DINO / CLIP instead",
    ],
)


def main() -> None:
    write("01_images_as_tensors.ipynb", N01)
    write("02_convolutions.ipynb", N02)
    write("03_datasets_and_augmentations.ipynb", N03)
    write("04_first_cnn.ipynb", N04)
    write("05_training_cnn.ipynb", N05)
    write("06_transfer_learning.ipynb", N06)
    write("07_gradcam.ipynb", N07)
    write("08_intro_to_vit.ipynb", N08)
    write("09_building_vit.ipynb", N09)
    write("10_cnn_vs_vit.ipynb", N10)


if __name__ == "__main__":
    main()
