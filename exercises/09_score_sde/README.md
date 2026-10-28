# 09 — Score SDE: the continuous-time unification

**Book:** Ch. 4 "Diffusion Models Today: Score SDE Framework" (pp. 98–137). §4.1 principles, §4.2 why forward/reverse SDE and PF-ODE share marginals, §4.3 training & sampling, §4.4 VE/VP instantiations, §4.5 perturbation kernels. Optional §4.6 + Appendix C (Fokker–Planck, Itô, Girsanov) for the reverse-SDE derivation.

## Why this module
DDPM and NCSN become two choices of `(f, g)` in one SDE. Once you have a score network you get, for free: a stochastic sampler (reverse SDE), a deterministic one (PF-ODE), and exact likelihoods (instantaneous change of variables from module 02).

## Exercises (`score_sde.py`)
1. `VPSDE` and `VESDE`: `drift`, `diffusion`, `marginal`, `prior_sample`, `prior_log_prob`. Derive `marginal` by solving the linear SDE's mean/variance ODEs.
2. `dsm_loss_sde`
3. `reverse_sde_sample` — Euler–Maruyama backwards.
4. `pf_ode_drift`, `pf_ode_sample` — Heun backwards.
5. `pf_ode_log_likelihood` — the test checks against the exact Gaussian log-density.

All sampler tests use the closed-form score of Gaussian data (`oracle_score` in the test).

```bash
uv run pytest exercises/09_score_sde
```

## Experiments
- Train one score net on `sample_2d("swissroll")` with both VP and VE. Sample with reverse SDE (1000 steps) and PF-ODE (50–200 steps). Plot trajectories: SDE paths are jittery, ODE paths smooth.
- **Same marginals, different paths:** plot histograms of `x_t` at t ∈ {0.2, 0.5, 0.8} from forward simulation, reverse SDE and PF-ODE — they should agree (§4.2).
- **Prior mismatch:** sample Gaussian data with the oracle score (copy `oracle_score` from the test) under `VESDE(0.01, σ_max)` for σ_max ∈ {2, 10, 50}, with both samplers. The PF-ODE's sample mean is biased by ≈ `s/σ_max`, the reverse SDE's much less. Why does the SDE forgive a wrong prior and the ODE not?
- PF-ODE as an encoder: map data → noise → data. Is it invertible numerically? How does that depend on steps?
- Implement a predictor–corrector sampler (reverse-SDE predictor + Langevin corrector, book §4.3). Does it beat plain EM at equal network calls?
- Compute test NLL in bits/dim on MNIST with Hutchinson divergence (stretch).

## Check your understanding
- Show DDPM's update is a discretisation of the VP-SDE as `T → ∞` (`β_t = β(t)/T`).
- Why does the PF-ODE have the *same* marginals as the SDE although its trajectories are deterministic? (Fokker–Planck, §4.2 / §6.4.)
