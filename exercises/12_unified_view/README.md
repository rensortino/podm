# 12 — One model, many coordinates: the unified view

**Book:** Ch. 6 "A Unified and Systematic Lens on Diffusion Models" (pp. 182–213): §6.1 conditional tricks, §6.2 roadmap of training losses, §6.3 equivalences, §6.4 Fokker–Planck. Appendix D.6 "(Optional) Elucidating Diffusion Model (EDM)" (p. 508).

## Why this module
This is the payoff of Part B. DDPM (ε), NCSN (score), score SDE and flow matching (v) are the *same* object: a parameterisation of `E[x₀ | x_t]` under `x_t = α_t x₀ + σ_t ε`, trained with a weighted MSE, sampled by integrating one velocity field. The differences are (1) which affine coordinates the net outputs, (2) the loss weighting in SNR, (3) the schedule. EDM makes those three choices explicitly and well.

## Exercises (`unify.py`)
1. Conversions: `eps_to_x0`, `x0_to_eps`, `eps_to_score`, `score_to_eps`, `to_velocity`, `x0_eps_from_velocity`.
2. `velocity_from_score` — test: the PF-ODE drift built from the exact score *equals* the exact flow-matching velocity.
3. `snr`, `weight_eps_per_x0`, `weight_v_per_eps` — every loss is every other loss with a different weight.
4. `edm_coeffs`, `edm_denoiser` — test checks network inputs and targets have unit variance at all noise levels.

```bash
uv run pytest exercises/12_unified_view
```

## Experiments (the big one)
1. Train **one** x₀-predicting `TimeMLP` on `sample_2d("8gaussians")` with the cosine path. Then, from the same network, sample with: DDPM ancestral (module 05, via ε), reverse SDE (module 09, via score), and flow-matching Euler (module 11, via v). Same distribution?
2. Train four nets with ε-, x₀-, v- and EDM-parameterisation, all with matched effective SNR weighting. Compare. Then change only the weighting. Which matters more: parameterisation or weighting?
3. Plot the implied weighting `w(SNR)` of: DDPM simple loss, NCSN σ², flow matching (linear path), EDM. (Book §6.2's roadmap figure is the answer key.)

## Check your understanding
- Fill in the book's equivalence table (§6.3) from memory.
- Why does ε-prediction misbehave as `α_t → 0` and x₀-prediction as `σ_t → 0`? How does EDM's `c_skip` interpolate?
