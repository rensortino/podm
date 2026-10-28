# 02 — ODEs, SDEs and how densities move

**Book:** Appendix A "Crash Course on Differential Equations" (pp. 436–452), Appendix B §B.1–B.2 "Change of Variable" and "Continuity Equation" (pp. 453–466). Optional: Appendix C (Itô, Girsanov) once you reach module 09.

## Why this module
The book's punchline (p. 409, "density-transport backbone") is that *every* diffusion model learns a velocity field and samples by integrating a differential equation. You will reuse these solvers, and the instantaneous change-of-variables formula is how the PF-ODE gives exact likelihoods (module 09) and how continuous flows work (module 10).

## Exercises (`diffeq.py`)
1. `euler_ode`, `heun_ode`, `rk4_ode` — the test measures empirical convergence order.
2. `euler_maruyama` — check it on the Ornstein–Uhlenbeck process, whose marginals are Gaussian with known mean/variance. (The VP-SDE in module 09 *is* an OU process.)
3. `pushforward_log_prob` — the discrete change-of-variables formula.
4. `divergence` + `ode_with_log_density` — the continuous version: `d log p/dt = -∇·f`.

```bash
uv run pytest exercises/02_differential_equations
```

## Experiments
- Simulate 10k OU paths starting from `sample_2d("moons")` with `dx = -x dt + √2 dW`. Plot snapshots at t = 0, 0.25, 1, 3. Where does the data "go"? This *is* the forward diffusion.
- Run your ODE solvers backwards on `dx/dt = -x`. Why is going backwards for an SDE not just "flip the sign of dt"? (Answer comes in module 09: the reverse SDE needs the score.)
