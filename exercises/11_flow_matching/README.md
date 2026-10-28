# 11 — Flow matching

**Book:** §5.2 "Flow Matching Framework" (pp. 149–162), §5.3 "Constructing Probability Paths and Velocities" (pp. 162–175), optional §5.4 "Properties of the Canonical Affine Flow" (pp. 175–181).

## Why this module
Regress a velocity field onto *conditional* velocities of a simple per-sample path; the minimiser is the marginal velocity, which transports noise to data. No ODE solves during training, no score, no SDE — yet module 12 shows it is the PF-ODE in different coordinates.

**Convention switch:** from here on, `t = 0` is data and `t = 1` is noise, `x_t = α_t x₀ + σ_t ε` (book's unified notation). Many FM papers use the opposite direction — be careful when reading them.

## Exercises (`flow_matching.py`)
1. `linear_schedule`, `cosine_schedule` — `(α, σ, α', σ')`.
2. `interpolate`
3. `cfm_loss` — the test also shows the *conditional* target is attainable only by a model that cheats with `(x₀, ε)`; the honest minimiser is the conditional expectation.
4. `sample_ode` (Euler / Heun) — tested with the exact marginal velocity for Gaussian data.
5. `minibatch_ot_pairing`

```bash
uv run pytest exercises/11_flow_matching
```

## Experiments
- Train `TimeMLP` (t ∈ [0,1], default `t_scale`) on `sample_2d("checkerboard")` with linear vs cosine schedules. Sample with 1, 2, 5, 10, 50 Euler steps; plot quality vs steps.
- Retrain with `minibatch_ot_pairing`. Plot trajectories: straighter? Does few-step sampling improve?
- **Rectified flow / reflow:** sample (ε, x̂₀) pairs from your trained model, retrain on those *fixed* pairs. Repeat once. Measure path straightness `E‖x₁ − x₀ − ∫v‖`.
- Train FM on MNIST with `SmallUNet` (default `t_scale`). It is usually simpler and faster than DDPM — compare.

## Check your understanding
- Prove `argmin cfm_loss = E[u_t | x_t]` (one line) and that this marginal velocity generates `p_t` (continuity equation, §5.2).
- What is the conditional velocity of the VP path in terms of `ε`? Notice anything about the DDPM loss? (→ module 12)
