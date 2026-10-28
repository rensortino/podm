# 17 — Consistency models and flow maps

**Book:** Ch. 11 "Learning Fast Generators from Scratch" (pp. 366–407): §11.1 prologue (flow maps), §11.2 consistency models in discrete time, §11.3 continuous time, §11.4 consistency trajectory models (CTM), §11.5 MeanFlow.

## Why this module
Instead of learning the *velocity* and integrating it, learn the *flow map* `Ψ(x_t, t → s)` directly. A consistency model is the special case `s = ε` (always jump to data); CTM and MeanFlow learn general `t → s` jumps. The training signal is self-consistency along the PF-ODE — with a teacher (CD) or without (CT, MeanFlow).

## Exercises (`consistency.py`)
1. `karras_timesteps`, `cm_coeffs`, `consistency_fn` (boundary condition by construction).
2. `cd_loss` — test: zero for the exact consistency function of Gaussian data.
3. `ct_loss`
4. `multistep_consistency_sample`
5. `meanflow_target` — the MeanFlow identity via one `torch.func.jvp`. Test: the exact average velocity of a linear ODE is a fixed point.
6. `meanflow_loss`, `meanflow_sample`

```bash
uv run pytest exercises/17_consistency_flow_maps
```

## Experiments
- CD on 2D using your module-09/12 teacher (Heun step as `ode_step`), EMA target network. Compare 1-step and 2-step samples to the teacher at 50 steps.
- CT from scratch on 2D: compare to CD; study the effect of N (discretisation) and the curriculum (increasing N during training).
- MeanFlow from scratch on `checkerboard` with a `TimeMLP`-style net taking `(z, r, t)` (e.g. embed `t` and `t − r`). 1-step vs 2-step samples. Sample `r = t` with some probability (≈ plain FM) — why does that help?
- MNIST 1-step generation with MeanFlow (stretch).

## Check your understanding
- Why does CT's target bias vanish as N → ∞? Why does training get unstable at the same time?
- Show that MeanFlow with `r = t` reduces to flow matching.
