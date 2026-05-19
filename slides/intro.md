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

# Let's go.

Open `notebooks/01_images_as_tensors.ipynb`
