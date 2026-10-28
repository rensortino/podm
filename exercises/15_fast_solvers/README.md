# 15 — Fast ODE solvers: DDIM, DEIS, DPM-Solver(++)

**Book:** Ch. 9 "Numerical Solvers for Fast Diffusion Sampling" (pp. 283–348): §9.1 prologue, §9.2 DDIM, §9.3 DEIS, §9.4 DPM-Solver, §9.5 DPM-Solver++, optional §9.6 ParaDiGMs.

## Why this module
Generic ODE solvers waste steps on the stiff linear part of the PF-ODE. Writing it in log-SNR λ makes it semi-linear; solving the linear part exactly gives DDIM (first order) for free, and higher-order Taylor/multistep approximations of the remaining integral give DPM-Solver/DEIS. This is how 1000 steps became 10–20.

## Exercises (`fast_solvers.py`)
1. `timesteps` (uniform in t vs λ)
2. `ddim_step` — test: exact for point-mass data (why?), and with η = 1 recovers the right marginals.
3. `dpm_solver1_step` — test: identical to DDIM η = 0.
4. `dpm_solver2_step`, `sample_dpm_solver2`
5. `sample_ddim`, `sample_dpm_solver_pp_2m`

The convergence test compares against the *exact* PF-ODE map for Gaussian data and measures empirical order (DDIM ≈ 1, second-order solvers ≈ 2).

```bash
uv run pytest exercises/15_fast_solvers
```

## Experiments
- With a trained 2D or MNIST model (modules 05/09/12): quality vs. NFE curves for Euler (module 11), Heun, DDIM, DPM-Solver-2, DPM-Solver++(2M) at NFE ∈ {5, 10, 20, 50}. Use a sliced-Wasserstein or MMD metric for 2D.
- Uniform-t vs uniform-λ vs EDM's ρ=7 time grid.
- Why is the data-prediction form (++) better with CFG at high guidance scale? Try w = 8 on MNIST.
- Optional: implement DEIS with a polynomial extrapolation of ε in t.

## Check your understanding
- Derive the semi-linear form via variation of constants. Where exactly does DDIM's approximation enter?
