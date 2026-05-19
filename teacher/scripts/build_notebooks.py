"""Generate the course notebooks from compact Python sources.

Run with:  python3 teacher/scripts/build_notebooks.py

We keep the *authoring* form as plain Python lists of (kind, text) cells so the
content is easy to edit, diff, and review. The script writes one .ipynb per
notebook into ``notebooks/``. Students never see this script — they just open
the generated notebooks.

No external dependencies: stdlib only.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> dict:
    text = textwrap.dedent(text).strip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _splitlines(text),
    }


def code(text: str) -> dict:
    text = textwrap.dedent(text).strip("\n")
    return {
        "cell_type": "code",
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

    Two conv blocks, then a small classifier head. Use the shape rule to predict
    output sizes *before* you run the cell.
    """),
    code("""
    class SmallCNN(nn.Module):
        def __init__(self, n_classes: int = 10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28 -> 14
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 14 -> 7
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = SmallCNN().to(device)
    print(model)
    print("trainable params:", count_params(model))
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


N05 = skeleton(
    "05 — Training a CNN, properly",
    "Day 2 · Notebook 1 of 3",
    "Take the Day 1 model further: better architecture, augmentations, learning-rate scheduling, and a clean train/val/test split.",
    [
        "Recap the Day 1 model and where it plateaus",
        "Architecture: more conv blocks, BatchNorm, GAP head",
        "Augmentations that actually help (RandomCrop, ColorJitter)",
        "Learning-rate schedules (cosine, step)",
        "Saving / loading checkpoints",
        "Validating without leaking the test set",
    ],
)

N06 = skeleton(
    "06 — Transfer learning",
    "Day 2 · Notebook 2 of 3",
    "Most real CV systems start from a pretrained backbone. Today: ResNet18 / MobileNet / EfficientNet, frozen vs. fine-tuned, and how to swap heads.",
    [
        "Why pretrained weights matter",
        "Loading ResNet18 from torchvision",
        "Freezing layers, replacing the classifier",
        "Fine-tuning vs feature extraction — when to pick which",
        "Small dataset case study (Cats vs Dogs subset or CIFAR10)",
        "Common pitfalls: input size, normalization stats, learning rate per group",
    ],
)

N07 = skeleton(
    "07 — Explaining what the CNN looks at",
    "Day 2 · Notebook 3 of 3",
    "Open the black box. GradCAM, saliency, and embeddings — three views of one model.",
    [
        "Recording activations with forward hooks",
        "GradCAM from scratch (one screen of code)",
        "Saliency / vanilla gradient",
        "Embedding visualization (PCA or t-SNE)",
        "Failure analysis: when explanations disagree with intuition",
    ],
)

N08 = skeleton(
    "08 — Attention, the gentle introduction",
    "Day 3 · Notebook 1 of 3",
    "Before ViTs make sense, attention has to make sense. We build it from scratch on tiny tensors so it's no longer a black box.",
    [
        "Why CNNs are local, and why that's a limit",
        "Queries, keys, values on a toy example",
        "Scaled dot-product attention by hand",
        "Multi-head attention — what 'head' actually means",
        "Sanity: implementing it in ~30 lines",
    ],
)

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
    write("08_attention_intro.ipynb", N08)
    write("09_building_vit.ipynb", N09)
    write("10_cnn_vs_vit.ipynb", N10)


if __name__ == "__main__":
    main()
