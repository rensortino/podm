# 06 — Energy-based models and Langevin sampling

**Book:** §3.1 "Energy-Based Models" (pp. 70–76).

## Why this module
EBMs give the cleanest motivation for scores: the normaliser `Z_θ` kills maximum likelihood, but `∇ₓ log p` never sees it. Langevin dynamics turns a score into samples — the same sampler NCSN uses (module 08), and the reverse SDE in module 09 is its time-dependent generalisation.

## Exercises (`ebm.py`)
1. `score_from_energy`
2. `langevin_sample`
3. `contrastive_divergence_loss`

```bash
uv run pytest exercises/06_ebm
```

## Experiments
- Train an EBM (`podm.nets.MLP(2, 1)`) on `sample_2d("8gaussians")` using short-run Langevin (20–60 steps, step 0.01) from noise, or a persistent replay buffer. Plot `exp(-E)` on a grid. Expect instability — that is the point.
- Langevin with the *true* score of a 2-mode mixture with well-separated modes: start all chains in one mode. How long until mass is balanced? (Mixing failure → motivates noise annealing in module 08.)

## Check your understanding
- Why does CD need samples from the *current* model, and why is that the expensive/unstable part?
- If you only want to sample, why not learn the score directly and skip the energy? (→ module 07)
