# Teacher-only

Everything in this directory is for me (the instructor). Students don't
need any of it — the top-level repo `README.md` is the student-facing entry.

## Contents

- [`TEACHING_NOTES.md`](TEACHING_NOTES.md) — cohort-feedback notes, re-read before each course.
- [`SLIDES.md`](SLIDES.md) — how to build/edit Marp slides + syntax reference.
- [`scripts/`](scripts/) — `build_notebooks.py` regenerates notebooks from compact Python source.
- [`slides/`](slides/) — Marp markdown decks and rendered PDFs.
  - `day1.md`, `day2.md`, `day3.md` — one deck per day.
  - `MC-questions.md` / `MC-solutions.md` — multiple-choice quiz.
  - `images/` — diagrams used in the decks.

## Editing notebooks

Notebooks are **generated** from `scripts/build_notebooks.py`. Each
notebook's cells live as plain Python lists of markdown/code blocks —
much easier to diff and review than raw `.ipynb` JSON.

To edit a notebook's content:

1. Edit `scripts/build_notebooks.py` (the `N01`, `N02`, … lists).
2. Regenerate from repo root: `uv run python teacher/scripts/build_notebooks.py`.
3. Re-execute to refresh outputs (optional but recommended before teaching):
   ```bash
   uv run python -c "
   import nbformat; from nbclient import NotebookClient
   nb = nbformat.read('notebooks/01_images_as_tensors.ipynb', as_version=4)
   NotebookClient(nb, timeout=600, kernel_name='python3').execute()
   nbformat.write(nb, 'notebooks/01_images_as_tensors.ipynb')
   "
   ```

If you'd rather edit `.ipynb` files directly in JupyterLab, that's fine —
just know the build script will overwrite them if you re-run it.

## Editing slides

See [`SLIDES.md`](SLIDES.md). Quick reference:

```bash
# From teacher/
marp slides/day1.md --pdf --allow-local-files
marp -s slides/   # live-reload server while editing
```

## Pre-cohort checklist

- [ ] Re-read `TEACHING_NOTES.md`.
- [ ] Rebuild all slide PDFs (`marp slides/day*.md --pdf --allow-local-files`).
- [ ] Re-execute all notebooks headlessly to refresh outputs.
- [ ] Update the year/cohort line on `slides/day1.md` title slide.
