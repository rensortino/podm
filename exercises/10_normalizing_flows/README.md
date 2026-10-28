# 10 — Normalizing flows (and why we leave them)

**Book:** §5.1 "Flow-Based Models: Normalizing Flows and Neural ODEs" (pp. 141–149).

## Why this module
Flows give exact likelihoods via change of variables (module 02) but pay for it with architectural constraints (invertibility + cheap Jacobians). Continuous flows (neural ODEs) remove the constraints but make *maximum-likelihood training* need an ODE solve per step. Flow matching (module 11) keeps the ODE and throws away the expensive training — it is to CNFs what DSM was to score matching.

## Exercises (`flows.py`)
1. `AffineCoupling.forward` / `inverse` — test checks the log-det against an autograd Jacobian.
2. `RealNVP.forward`, `log_prob`, `sample` — test checks the density integrates to 1.

```bash
uv run pytest exercises/10_normalizing_flows
```

## Experiments
- Train RealNVP by maximum likelihood on `sample_2d("moons")` and `"checkerboard"`. Plot the density on a grid and samples. Where does it struggle? (Topology: a diffeomorphism can't create holes.)
- **CNF:** reuse `ode_with_log_density` from module 02 with `f = TimeMLP` to train a continuous flow by max likelihood on `moons`. Measure wall-clock per iteration vs. RealNVP. This pain is the motivation for module 11.

## Check your understanding
- Why does a coupling layer have a triangular Jacobian? What would a full-matrix affine layer cost?
- A CNF needs `div f`; how does Hutchinson make that `O(1)` backward passes, and what does it cost in variance?
