# Learning roadmap

Follows *The Principles of Diffusion Models* (Lai, Song, Kim, Mitsufuji, Ermon — [arXiv:2510.21890](https://arxiv.org/abs/2510.21890)). Page numbers refer to the arXiv PDF (522 pp.).

**This file is yours to edit.** Reorder rows, mark modules `skip`, change time budgets, add your own modules (template at the bottom). Claude reads this file to know where you are, so keep the **Status** column and the **Log** current (or ask Claude to update them).

Status values: `todo` · `doing` · `done` · `skip`

## The arc in one paragraph

Generative modeling = fitting `p_θ ≈ p_data` (01). Diffusion = transporting noise to data by integrating a differential equation (02). The book reaches it three ways — **variational** (VAE → hierarchical VAE → DDPM: 03–05), **score-based** (EBM → score matching → NCSN → score SDE: 06–09), **flow-based** (normalizing flows → flow matching: 10–11) — then shows they are one model in different coordinates (12). Everything after is about using that model well: controlling it (14), sampling it in fewer steps with better solvers (15), and finally learning generators that need 1–4 steps (16–17).

## Modules

| #  | Module | Book | Core? | Est. | Status |
|----|--------|------|-------|------|--------|
| **Phase 0 — Foundations** |
| 01 | [Generative modeling basics](exercises/01_generative_basics) — MLE, KL directions, reparameterisation | Ch. 1, pp. 21–33 | core | 3h | todo |
| 02 | [ODEs, SDEs, density evolution](exercises/02_differential_equations) — solvers, Euler–Maruyama, change of variables, continuity eq. | App. A, B.1–B.2, pp. 436–466 | core | 4h | todo |
| **Phase 1 — Variational path: from VAEs to DDPM** |
| 03 | [VAE](exercises/03_vae) — ELBO, Gaussian KLs, tight bound with exact posterior | §2.1, pp. 44–54 | core | 4h | todo |
| 04 | [HVAE → diffusion](exercises/04_hvae_to_diffusion) — stacked latents, freeze the encoder, closed-form marginals | §2.1.4–2.2.1, pp. 52–58 | core | 4h | todo |
| 05 | [DDPM](exercises/05_ddpm) — schedules, posterior, ε-loss, ancestral sampling, VLB ↔ simple loss | §2.2, pp. 54–67 | core | 8h | todo |
| **Phase 2 — Score-based path: from EBMs to NCSN** |
| 06 | [Energy-based models](exercises/06_ebm) — CD, Langevin | §3.1, pp. 70–76 | core | 3h | todo |
| 07 | [Score matching](exercises/07_score_matching) — implicit, sliced, denoising, Tweedie | §3.2–3.3, pp. 76–91 | core | 5h | todo |
| 08 | [NCSN](exercises/08_ncsn) — noise ladder, annealed Langevin, NCSN ↔ DDPM | §3.4–3.5, pp. 91–97 | core | 4h | todo |
| **Phase 3 — Continuous time** |
| 09 | [Score SDE](exercises/09_score_sde) — VP/VE SDEs, reverse SDE, PF-ODE, exact likelihood | Ch. 4, pp. 98–137 (+ App. C) | core | 8h | todo |
| **Phase 4 — Flow-based path** |
| 10 | [Normalizing flows](exercises/10_normalizing_flows) — RealNVP, CNF motivation | §5.1, pp. 141–149 | core | 4h | todo |
| 11 | [Flow matching](exercises/11_flow_matching) — CFM, paths, minibatch OT, rectified flow | §5.2–5.4, pp. 149–181 | core | 6h | todo |
| **Phase 5 — Unification** |
| 12 | [Unified view + EDM](exercises/12_unified_view) — ε/x₀/score/v conversions, loss weights, preconditioning | Ch. 6, pp. 182–213; App. D.6 p. 508 | core | 8h | todo |
| 13 | [Diffusion & optimal transport](exercises/13_optimal_transport) | Ch. 7, pp. 214–248 | optional | 4h | todo |
| **Phase 6 — Sampling** |
| 14 | [Guidance](exercises/14_guidance) — classifier, CFG, DPS, Diffusion-DPO | Ch. 8, pp. 251–282 | core | 6h | todo |
| 15 | [Fast solvers](exercises/15_fast_solvers) — DDIM, DPM-Solver-2, DPM-Solver++(2M) | Ch. 9, pp. 283–348 | core | 6h | todo |
| **Phase 7 — Fast generators** |
| 16 | [Distillation](exercises/16_distillation) — progressive distillation, distribution matching | Ch. 10, pp. 350–365 | core | 5h | todo |
| 17 | [Consistency models & flow maps](exercises/17_consistency_flow_maps) — CD, CT, MeanFlow | Ch. 11, pp. 366–407 | core | 8h | todo |
| **Epilogue** |
| 18 | [Discrete (masked) diffusion](exercises/18_discrete_diffusion) | §12.1–12.2, pp. 409–434 | optional | 5h | todo |

Estimates include reading, exercises, and the main experiment. Core total ≈ 90h.

## Alternative tracks

- **Fast track to a working image model:** 01 → 03 → 04 → 05 → 11 → 12 → 15. The score path (06–09) can come after; module 12 will feel like a recap of things you haven't seen yet, so re-read it after 09.
- **Flow-first (modern practice):** 02 → 10 → 11 → 12 → 15 → 17, then go back to 03–09 for the "why".
- **Theory-heavy:** add App. C (Itô, Girsanov, pp. 472–486) after 09, Ch. 6.4 Fokker–Planck carefully, and 13.

## Dependency graph

```
01 ─┬─ 03 ── 04 ── 05 ─┐
    │                  ├── 09 ──┐
02 ─┼─ 06 ── 07 ── 08 ─┘        ├── 12 ─┬─ 14
    │                           │       ├─ 15 ── 16 ── 17
    └─ 10 ── 11 ────────────────┘       ├─ 13 (opt)
                                        └─ 18 (opt)
```

## Each module contains

- `README.md` — why the module exists, book pages, exercise list, experiments, self-check questions.
- `<topic>.py` — stubs with docstrings (equations, conventions, hints). Replace each `raise NotImplementedError("TODO")`.
- `test_<topic>.py` — fast checks (seconds, CPU). Most use *closed-form oracles* (e.g. the exact score of Gaussian data) so you can verify a sampler before training anything.
- `intuition/` (created on demand) — explainers and interactive demos from the mentor agent.
- Experiments are scripts you write yourself in the module folder (`train.py`, notebooks, …), using `podm/` for data, networks and plots.

## Getting help

- **Mentor** (`ml-intuition-coach` agent): "explain Tweedie's formula intuitively" → explanation, minimal example, interactive HTML demo in the module's `intuition/`.
- **Teaching assistant** (`learn` skill): "quiz me on module 05", "I'm stuck on q_posterior" → Socratic hints, one step at a time.
- **Grader** (`exercise-grader` agent): "grade module 05" → runs tests, hint ladder per failing function, no solutions.

## Log

<!-- One line per session: date — module — what you did / what clicked / open questions. -->

## Adding your own module

Copy this row into the table and create `exercises/NN_name/` with a README (book pages + goal), a stub file and a `test_*.py` (unique file names across modules — pytest imports them by basename):

```
| NN | [Title](exercises/NN_name) — one-line scope | §x.y, pp. a–b | core/optional | Xh | todo |
```

Ideas: latent diffusion on MNIST with your module-03 VAE · diffusion transformers (DiT) · Schrödinger bridges (§7) · Wasserstein gradient flows (App. B.3) · ParaDiGMs (§9.6) · CTM (§11.4).
