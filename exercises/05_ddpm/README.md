# 05 — DDPM: Denoising Diffusion Probabilistic Models

**Book:** §2.2 "Variational Perspective: DDPM" (pp. 54–67). Proofs in Appendix D.1 (pp. 486–489).

## Why this module
This is the first full diffusion model. Everything is a Gaussian identity you can verify: the closed-form forward marginal, the tractable posterior `q(x_{t−1}|x_t,x_0)`, and the observation that the ELBO's KL terms reduce to an MSE on predicted noise.

## Exercises (`ddpm.py`) — mind the index convention in the module docstring
1. `linear_beta_schedule`, `cosine_beta_schedule`, `alpha_bar_from_betas`
2. `q_sample` — one-shot forward noising.
3. `q_posterior` — the test re-derives it by conditioning a joint Gaussian; derive it yourself via Bayes first.
4. `predict_x0_from_eps`
5. `ddpm_loss` — the "simple" ε-MSE.
6. `p_mean_var`, `ddpm_sample` — ancestral sampling. The test plugs in the *exact* optimal ε-predictor for Gaussian data (`gaussian_oracle_eps` in the test file — read it!) and checks the sampler reproduces the data distribution. This decouples "is my sampler right?" from "is my network trained?".
7. `vlb_term` — show the per-step ELBO KL is a re-weighted ε-MSE.

```bash
uv run pytest exercises/05_ddpm
uv run pytest exercises/05_ddpm -k posterior   # single test
```

## Experiments
1. **2D:** train `TimeMLP(t_scale=1.0)` on `sample_2d("moons")` (T = 1000, ~20k Adam steps, batch 512, lr 1e-3). Plot samples and `trajectories2d` of the reverse chain.
2. **MNIST:** `SmallUNet(t_scale=1.0)` + EMA (`podm.nets.EMA`), ~20–30 min on an M-series GPU (`get_device()` → mps). Save sample grids during training.
3. Compare `var_type="posterior"` vs `"beta"`, linear vs cosine schedule. Plot `alpha_bar` and the SNR `ᾱ/(1−ᾱ)` for both.
4. Train with the true ELBO weighting from `vlb_term` instead of the simple loss. Which gives better samples, which better likelihood? (Book pp. 64–66.)
5. Sample with only the last 50 / first 50 steps replaced by "predict x0 and jump". What does each phase of the chain decide (coarse layout vs. fine detail)?

## Check your understanding
- Where in the derivation does "the reverse of a small Gaussian step is Gaussian" matter? What breaks with T = 10?
- ε-prediction, x0-prediction and mean-prediction are related by fixed affine maps — write them out. (Module 12 generalises this.)
