# Teaching notes — DSR Computer Vision course

Running list of student-feedback-driven adjustments. Re-read before each cohort.

## What recent cohorts want (~last year)

- **Fundamentals + practical application > deep theory.** Students prefer being able to *execute* what they learned afterwards, not chasing depth that overwhelms them. Bias content toward "you can use this on Monday" over "here is the full derivation".
- **Clear agenda up front.** Each day (and ideally each notebook) should state what they will learn and what they will be able to *do* afterwards. Objectives must be concrete and outcome-shaped, not topic-shaped.
- **Old TF material needs a refresh, not a port.** Trends move fast. Treat the old repo as a source of structure/examples, not a script.

## ChatGPT / vibe-coding habit

Students lean heavily on ChatGPT and vibe-coding. This is fine in moderation, but:

- Periodically remind them not to over-rely on it — especially during the explicit-train-loop sections, where the point is to internalize `forward → loss → backward → step → zero_grad`, not to ship working code.
- For debugging exercises (debugging section in `slides/day1.md`), have them diagnose **before** pasting into a chatbot.
- Encourage "explain the tensor shape mismatch yourself first, then verify with an LLM" as a workflow.

## Implications for this repo

- Each notebook opens with **Objectives** (3–5 bullets, action verbs: "build…", "diagnose…", "apply…").
- Prefer minimum-viable examples that they can copy into their own projects.
- Keep skipping Lightning / over-abstracted frameworks — explicitness aids retention.
- Day-end "what you can now do" recap, not "what we covered".
