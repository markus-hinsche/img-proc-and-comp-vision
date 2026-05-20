# Slides (Marp)

Slides live in `slides/` as Marp markdown. Each `.md` file is one deck.

## Building

PDF is the only output format we ship. Source markdown + images live in `teacher/slides/`; the generated PDFs are written to `slides/` at the repo root so students can grab them without digging into the teaching materials.

```bash
# Run from the project root. <deck> is day1, day2, day3, ...
npx --yes @marp-team/marp-cli@latest teacher/slides/<deck>.md --pdf --allow-local-files -o slides/<deck>.pdf

# Or install once
brew install marp-cli
marp teacher/slides/<deck>.md --pdf --allow-local-files -o slides/<deck>.pdf

# Live-reload server while editing
marp -s teacher/slides/

# Build all three day decks at once
for d in day1 day2 day3; do marp teacher/slides/$d.md --pdf --allow-local-files -o slides/$d.pdf; done
```

`--allow-local-files` is needed when slides reference local images.

Best editor experience: install the **Marp for VS Code** extension — live preview in a side panel.

## Required frontmatter

Every deck starts with:

```markdown
---
marp: true
theme: default
paginate: true
---
```

`theme: default` | `gaia` | `uncover` are built in. `paginate: true` shows page numbers.

## Slide separator

A line containing only `---` starts a new slide.

```markdown
# Slide 1

content

---

# Slide 2
```

## Per-slide directives

HTML comments. Prefix with `_` to apply to **this slide only**; without `_` they apply from that slide onward.

```markdown
<!-- _class: invert -->            # dark theme, this slide only
<!-- _backgroundColor: #0d47a1 -->
<!-- _color: white -->
<!-- _paginate: false -->          # hide page number on this slide
<!-- _header: '' -->               # blank header on this slide
```

## Title / cover slide

```markdown
<!-- _class: invert -->
<!-- _paginate: false -->

# <!-- fit --> My Big Title

Subtitle, author, date
```

`<!-- fit -->` auto-sizes the heading to fill the slide width.

## Images

```markdown
![w:400 h:300](path.png)         # explicit size
![width:50%](path.png)           # percent of slide

![bg](url)                       # full-bleed background
![bg right:40%](url)             # right 40%, content on left
![bg left:30% blur](url)         # left 30%, blurred
![bg fit](url) / ![bg cover](url)
```

Multiple `![bg]` on one slide stack horizontally (split layout).

## Speaker notes

```markdown
## My slide

content

<!--
Speaker notes here.
Visible in presenter view (press `p` in the HTML deck), not on slides.
-->
```

## Math

Add `math: mathjax` to frontmatter, then:

```markdown
Inline: $ax^2 + bx + c$

$$
\int_0^\infty e^{-x}\,dx = 1
$$
```

## Code blocks

Standard fenced markdown — syntax highlighting is automatic.

````markdown
```python
import torch
x = torch.randn(3, 224, 224)
```
````

## Two columns

Marp has no built-in macro; use a small CSS grid:

```markdown
<div class="columns">
<div>

Left content (blank line before/after the inner markdown!)

</div>
<div>

Right content

</div>
</div>

<style>
.columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
</style>
```

## Coming from remark.js

| remark.js                    | Marp                              |
| ---------------------------- | --------------------------------- |
| `---` slide separator        | `---` ✅ same                      |
| `class: center, middle`      | `<!-- _class: invert -->` + CSS   |
| `???` speaker notes          | `<!--  ... -->` HTML comment      |
| `.scale[]` / `![:scale 50%]` | `![w:400](url)` or `![width:50%]` |
| `--` incremental             | not supported — split into slides |
| Config in JS                 | YAML frontmatter                  |

## Decks in this repo

- `slides/day1.md` — course intro, agenda, whoami, debugging reference
- `slides/day2.md` — CNNs: theory pointer, training, transfer learning, Grad-CAM
- `slides/day3.md` — Transformers for vision (attention, ViT, CNN vs ViT)
