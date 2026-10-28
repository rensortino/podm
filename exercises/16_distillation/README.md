# 16 — Distillation: fewer steps from a trained teacher

**Book:** Ch. 10 "Distillation-Based Methods for Fast Sampling" (pp. 350–365): §10.1 prologue, §10.2 distribution-based distillation (VSD/DMD-style), §10.3 progressive distillation.

## Why this module
Solvers (module 15) hit a wall around 10 steps: the ODE is curved. Distillation trains a new network to *jump*: progressive distillation halves the step count repeatedly by matching two teacher steps with one student step; distribution-based methods instead match the generator's output distribution to the teacher's using score differences.

## Exercises (`distill.py`)
1. `ddim_step_x0`
2. `progressive_distillation_target` — test: one student step from the target lands exactly where two teacher steps do.
3. `pd_loss`

```bash
uv run pytest exercises/16_distillation
```

## Experiments
- Take a trained x₀-prediction teacher on `sample_2d` (cosine path; module 12) at 256 DDIM steps. Distil 256 → 128 → … → 4 → 2 → 1, initialising each student from the previous one. Plot samples at each stage.
- **Distribution matching (stretch):** a one-step generator `G(z)`, the frozen teacher score `s_real`, and an online "fake" score `s_fake` trained by DSM on G's samples. Update G with the gradient `E_t[w(t)(s_fake − s_real)(x_t)] · ∂x_t/∂θ` (§10.2). Compare with progressive distillation at 1 step.

## Check your understanding
- Why does the student need x₀-/v-prediction rather than ε-prediction at very few steps?
- Why does progressive distillation compound error, and how do distribution-based methods avoid that?
