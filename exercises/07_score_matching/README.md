# 07 — Score matching: explicit → implicit → sliced → denoising

**Book:** §3.2 "From Energy-Based to Score-Based Generative Models" (pp. 76–80), §3.3 "Denoising Score Matching" (pp. 80–91). Proofs: Appendix D.2 (pp. 489–500).

## Why this module
You cannot regress onto `∇ log p_data` because you don't know it. Each objective here is a trick to remove that unknown, with a different cost. DSM is the one that scales — and it is *literally* the DDPM loss up to a reparameterisation (`ε = −σ · score`). Tweedie's formula is the bridge between "score" and "denoiser".

## Exercises (`score_matching.py`)
1. `_jacobian_trace` + `implicit_score_matching_loss` (Hyvärinen; integrate by parts yourself first).
2. `sliced_score_matching_loss` — Hutchinson-style projection.
3. `dsm_loss` — the test checks the minimiser is the score of the *noisy* density.
4. `tweedie_denoise`

```bash
uv run pytest exercises/07_score_matching
```

## Experiments
- Train `TimeMLP` (ignore `t`, pass `t=None`) with ISM, SSM and DSM(σ = 0.1) on `sample_2d("8gaussians")`. Time an epoch of each as you increase the data dimension with zero-padding (D = 2, 32, 256). Which scales?
- Plot each learned score field (`podm.plot.vector_field2d`) next to the data. Where is the score garbage? (Low-density regions between modes — book p. 90.)
- Run Langevin (module 06) with the learned score. Do the mixture weights come out right? (Book p. 91: no. Why?)

## Check your understanding
- Why does DSM's minimiser equal the noisy score? Re-derive in two lines via `E[∇ log q(x̃|x) | x̃] = ∇ log p_σ(x̃)`.
- Small σ: accurate but useless far from data. Large σ: covers space but blurs the distribution. What's the obvious fix? (→ module 08)
