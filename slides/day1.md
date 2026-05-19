---
marp: true
theme: default
paginate: true
---

<!-- _class: invert -->
<!-- _paginate: false -->

# <!-- fit --> Image Processing & Computer Vision

A 3-day PyTorch course

Markus Hinsche · datascienceretreat.com · 2026

---

# Agenda

Over three days we'll cover:

- **Day 1** — images as tensors, convolutions, datasets & augmentations
- **Day 2** — building and training CNNs, transfer learning, Grad-CAM
- **Day 3** — attention, building a ViT from scratch, CNN vs ViT

Hands-on throughout. Every concept is paired with a notebook you run yourself.

Topics we can dig into on request:
mixed-precision training, multi-GPU, model ensembling, hyperparameter tuning,
data-centric vs model-centric iteration, deployment.

---

# Download slides and code

Clone the repo to follow along:

```bash
$ git clone https://github.com/markus-hinsche/img-proc-and-comp-vision.git
$ cd img-proc-and-comp-vision
$ uv sync
$ uv run jupyter lab
```

Notebooks live in `notebooks/`, slides in `slides/`.

---

# whoami

- **Markus Hinsche** — co-founder & CTO of [Thea Care](https://theacare.de)
  - B2B SaaS: AI skin analysis for cosmetic and pharma brands
  - Computer vision in production (segmentation + classification)
- Previously: ML freelancer — Welthungerhilfe, Charité radiology (both CV)
- Strengths:
  - Python (since 2009), Deep Learning (since 2018)
  - Productionizing models, MLOps, mentoring
- [markushinsche.de](https://markushinsche.de) · [github](https://github.com/markus-hinsche/)

---

<!-- _class: invert -->

# Why computer vision?

### "Vision is the killer app of deep learning."

Image classification was the first task where deep nets clearly beat
classical methods (AlexNet, 2012). The recipes you learn this week —
convolutions, transfer learning, attention — are the foundation for
everything from medical imaging to robotics to generative models.

<!--
Mention concrete examples: skin analysis (my own work), self-driving,
medical imaging, defect detection in manufacturing, satellite imagery.
-->

---

# What you'll be able to do by Friday

- Load any image dataset into PyTorch and apply augmentations
- Build a CNN from scratch and train it on GPU
- Fine-tune a pretrained model (ResNet, EfficientNet, ViT) on your own data
- Debug training failures (loss not decreasing, overfitting, wrong shapes)
- Explain *why* a model predicts what it predicts (Grad-CAM)
- Understand the bridge from CNNs to Vision Transformers

---

# Ground rules

- **Ask questions immediately.** If you're lost on slide 3, you'll be lost on slide 30.
- **Run the code.** Reading notebooks is not the same as running them.
- **Pair up.** Two brains debug faster than one.
- **It's OK if your model doesn't train.** Half the skill is figuring out why.

---

<!-- _class: invert -->

# Debugging CV code

Failure modes you'll meet first — a reference deck. Point back to it
any time you hit a bug. The goal: make *"what kind of bug is this?"*
a reflex.

---

## The five bugs that eat the most time

### 1. Wrong tensor shape

**Symptom:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied` or
"expected input of shape (N, 3, H, W) but got (N, H, W, 3)".

**Reflex:** print `.shape` at every step. Especially before and after
`Flatten`, `permute`, `view`, and the first `Linear`.

### 2. HWC vs CHW

**Symptom:** image looks like static, or accuracy stuck at random.

**Cause:** loaded an image as `(H, W, C)` and fed it to a Conv2d expecting
`(C, H, W)`.

**Reflex:** if shape `[..., 3]` ends in a small number, you probably have HWC.

### 3. Forgot `.to(device)`

**Symptom:** `Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0`.

**Fix:** every batch needs `xb, yb = xb.to(device), yb.to(device)`. The model
needs `model.to(device)` once. Do it explicitly — no auto-magic.

### 4. Forgot `optimizer.zero_grad()`

**Symptom:** loss curve looks bizarre, gradients explode.

**Cause:** PyTorch *accumulates* gradients by default. Without `zero_grad()`,
each step uses the sum of all previous gradients.

**Reflex:** the loop pattern is fixed —

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Don't reorder.

### 5. `train()` vs `eval()` mode

**Symptom:** validation accuracy randomly bounces; dropout/batchnorm misbehave.

**Cause:** stayed in `model.train()` during evaluation, or never switched back.

**Reflex:** any eval/inference block starts with `model.eval()` and
`with torch.no_grad():`.

---

## Three more, less common but nasty

### Normalization mismatch

If you fine-tune a torchvision model, you must use ImageNet mean/std. If you
trained from scratch on your dataset, use *its* mean/std. Mixing them
silently degrades accuracy.

### Augmenting validation

Augmentations belong only on the train set. Augmenting validation makes the
metric noisy and uncomparable.

### Wrong `Flatten` size

`nn.Linear(in_features, ...)` must match `C * H * W` after the last conv block.
Compute it on paper from the shape rule, or run one batch through and read the
shape off `Flatten` output.

---

## The 30-second sanity check

Before training anything:

```python
xb, yb = next(iter(train_loader))
xb, yb = xb.to(device), yb.to(device)
logits = model(xb)
print(xb.shape, "->", logits.shape)
print("loss on init:", F.cross_entropy(logits, yb).item())
```

If this runs without error and the loss is roughly `-log(1/num_classes)`,
you're set to train.

---

<!-- _class: invert -->

# Let's go.

Open `notebooks/01_images_as_tensors.ipynb`
