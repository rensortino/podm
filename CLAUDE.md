# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A self-study course: implementing diffusion models from first principles, following *The Principles of Diffusion Models* (arXiv:2510.21890). The user is the learner. `ROADMAP.md` is the source of truth for module order, scope, book page references, and the learner's progress (Status column + Log) — read it at the start of a session and keep it updated when the user finishes or changes something.

## Working with the learner

- **Do not fill in `raise NotImplementedError("TODO")` stubs** in `exercises/*/<topic>.py` unless the user explicitly asks for a solution. Default to hints: point at the book section, the relevant equation in the docstring, or what the failing test checks.
- When asked to review an attempt, run that module's tests and explain failures in terms of the math, not just the code.
- New modules follow the existing layout (README with book pages / stub file / test file) and get a row in `ROADMAP.md`. Validate new tests against a reference solution kept outside the repo, then ship only stubs.
- The paper PDF is not committed. To consult it: `mkdir -p paper && curl -sL -o paper/podm.pdf https://arxiv.org/pdf/2510.21890` (`paper/` is git-ignored). Page numbers in READMEs/ROADMAP are PDF pages.

## Who handles what

Route learner requests to the right helper instead of answering everything inline:

| Request | Use | Notes |
|---|---|---|
| "I don't get X", "why does Y work", intuition for a concept (ELBO, Tweedie, PF-ODE, CFG, …) | **Mentor** — `ml-intuition-coach` agent | Tell it the module and book pages from `ROADMAP.md`. Have it write its README/example/interactive HTML into `exercises/NN_topic/intuition/` so materials sit next to the exercise. It must not implement the module's stubs. |
| Stuck on an exercise, "quiz me", "walk me through the derivation", checking understanding before/after a module | **Teaching assistant** — `learn` skill (Socratic tutoring) | Diagnose first, one step per turn, hints over answers. Use the module README's "Check your understanding" questions as quiz material. |
| "Check my attempt", "grade module NN", "why does this test fail" | **Grader** — `exercise-grader` agent (`.claude/agents/`) | Runs tests, gives a hint ladder, never writes the solution. Afterwards update the module's Status in `ROADMAP.md`. |
| Reordering/extending the course, adding a module | **Plan** agent, then implement here | New modules follow the layout below; ship stubs only. |
| Finding where something is defined in the book | Download the PDF (see above) and read the relevant pages | Page numbers in READMEs are PDF pages. |

At the start of a session: read `ROADMAP.md`, find the `doing` module, and offer the natural next step (TA quiz on the reading, mentor explainer, or grading).

## Git workflow: one branch and one PR per exercise

- Each module gets its own branch off up-to-date `main`: `exercise/NN-topic` (e.g. `exercise/01-generative-basics`), and its own PR titled `Module NN: <Topic>`.
- Commits on an exercise branch touch **only** that module: `exercises/NN_topic/` (implemented stubs, `intuition/` material, experiment scripts) plus that module's Status row / Log line in `ROADMAP.md`. Nothing else.
- Anything not specific to one exercise (`podm/`, `CLAUDE.md`, `pyproject.toml`, new or restructured modules, `.claude/`) goes on a separate `chore/<topic>` branch and PR. If such a change is needed mid-exercise, stash it, branch from `main`, and PR it separately.
- Stage files explicitly by path (never `git add -A` / `git add .`); check `git diff --cached --stat` before committing. `runs/`, `data/`, `paper/` are git-ignored and never committed.
- Open the PR with `gh pr create` once the module's tests pass (a draft PR earlier is fine). The body lists test status, experiments run (with key plots/numbers) and open questions.
- Push over HTTPS with gh credentials (SSH is unavailable): `git -c credential.helper= -c credential.helper='!gh auth git-credential' push -u https://github.com/rensortino/podm.git <branch>`.

## Commands

```bash
uv sync                                              # install (Python >= 3.11, torch, pytest)
uv run pytest                                        # all modules (expect failures for unimplemented stubs)
uv run pytest exercises/05_ddpm                      # one module
uv run pytest exercises/05_ddpm -k posterior         # one test
uv run python -m podm.data                           # sanity plot of toy datasets -> runs/
```

## Architecture

- `podm/` — shared infrastructure that is deliberately *not* a learning target: `data.py` (2D toy distributions incl. labeled ones, MNIST in [-1, 1]), `nets.py` (`TimeMLP`, `MLP`, `SmallUNet`, `EMA`; all nets take `(x, t[, y])`, `t_scale` controls the time-embedding scale: 1000 for t ∈ [0,1], 1 for integer steps; label `num_classes` is the CFG null label), `plot.py` (writes to `runs/`), `device.py` (cuda > mps > cpu).
- `exercises/NN_topic/` — one module per book chapter/section. Directories are not packages; pytest's default `prepend` import mode puts each test's directory on `sys.path`, so **every stub/test file basename must be unique across modules**. Modules are self-contained: they don't import from each other (tests re-implement any helper they need), so a learner can skip modules.
- Tests are fast CPU checks built on **closed-form oracles** — mostly Gaussian data, where the optimal ε/score/velocity/x₀-predictor and exact PF-ODE maps are analytic. This lets samplers and losses be verified without training. Keep new tests in that style (float64, seeded generators, statistical tolerances ~0.03 on 10–20k samples).

## Conventions that change across modules

- Modules 05 (DDPM): discrete `t ∈ {0..T-1}`, `alpha_bar[t] = prod_{s<=t}(1-beta_s)`; models called as `model(x_t, t_long)`.
- Modules 09, 15: continuous VP/VE SDEs, `t ∈ [0, 1]`, t = 0 is data.
- Modules 11 onward: `x_t = α_t x₀ + σ_t ε`, **t = 0 data, t = 1 noise** (opposite of many flow-matching papers); a schedule returns `(α, σ, α', σ')`.
- Module 17: consistency models use EDM/VE `x_t = x₀ + t z`, t ∈ [0.002, 80]; MeanFlow uses the linear path.
