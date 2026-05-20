---
marp: true
theme: default
paginate: true
style: |
  section { font-size: 24px; }
  section pre, section code { font-size: 0.78em; }
  section h1 { font-size: 1.6em; }
  section h2 { font-size: 1.25em; }
  section h3 { font-size: 1.05em; }
  section li { margin: 0.15em 0; }
  section table { font-size: 0.85em; }
---

<!-- _paginate: false -->
<!-- _backgroundColor: #f4f6fa -->
<!-- _color: #0d47a1 -->

# <!-- fit --> Day 2 — CNNs

Building, training, transfer-learning, and explaining

---

## Deep Learning for Computer Vision — theory

Before we build, read **cs231n Module 2**:

https://cs231n.github.io/

It covers convolutional layers, pooling, spatial arithmetic, and the
classic architectures (AlexNet, VGG, ResNet). ~1–2 hours.

We'll work from these concepts all day — having them fresh makes the
hands-on parts click faster.

---

# Today's notebooks

- `04_first_cnn.ipynb` — build a CNN from scratch
- `05_training_cnn.ipynb` — full training loop, monitor loss & accuracy
- `06_transfer_learning.ipynb` — fine-tune a pretrained model
- `07_gradcam.ipynb` — see *what* the model is looking at

By the end of the day you'll have trained your own CNN and visualized
its attention on real images.

---

<!-- _backgroundColor: #f4f6fa -->
<!-- _color: #0d47a1 -->
<!-- _paginate: false -->

# <!-- fit --> Let's build.

Open `notebooks/04_first_cnn.ipynb`
