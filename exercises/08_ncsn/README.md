# 08 — NCSN: multi-level denoising score matching

**Book:** §3.4 "Multi-Noise Levels of Denoising Score Matching (NCSN)" (pp. 91–96), §3.5 "A Comparative View of NCSN and DDPM" (pp. 96–97).

## Why this module
Module 07 left two problems: scores are inaccurate in low-density regions, and Langevin mixes badly between modes. Perturbing with a whole ladder of noise levels fixes both — and makes NCSN and DDPM visibly the same model with different bookkeeping (§3.5).

## Exercises (`ncsn.py`)
1. `geometric_sigmas`
2. `ncsn_loss` — σ²-weighted multi-level DSM. Show that at σᵢ the target `−ε/σᵢ` has variance `D/σᵢ²`, so the σ² weight equalises levels.
3. `annealed_langevin`
4. `ddpm_eps_to_score` — the NCSN ↔ DDPM dictionary.

```bash
uv run pytest exercises/08_ncsn
```

## Experiments
- Train on `sample_2d("8gaussians")` with `TimeMLP` conditioned on `log σ` (`t_scale≈1`), σ ∈ [0.01, 5], L = 10–20. Sample with annealed Langevin. Now do the mixture weights come out right? (Compare with module 07.)
- Vary L and `n_steps_each` at a fixed budget of total network calls.
- Fill in the comparison table from book §3.5 yourself: forward process, training target, sampler, noise-level parameterisation for DDPM vs NCSN.
- Optional: MNIST with `SmallUNet`, σ_max ≈ 50 (≈ max pairwise distance between training images; Song & Ermon 2020 heuristics).

## Check your understanding
- NCSN perturbs with `x + σε` (variance *explodes*), DDPM with `√ᾱ x + √(1−ᾱ) ε` (variance *preserved*). Both are discretisations of an SDE — which ones? (→ module 09)
