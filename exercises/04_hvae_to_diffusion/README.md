# 04 — From hierarchical VAEs to diffusion

**Book:** §2.1.4 "From standard VAE to hierarchical VAEs" (pp. 52–54), §2.2.1 "Forward process (fixed encoder)" (pp. 54–58).

## Why this module
This is the conceptual bridge you asked for: start from a stacked VAE, then make three changes — same-dimension latents, a fixed noising encoder, many layers — and you have DDPM. Seeing the ELBO survive each change is what makes the DDPM loss feel inevitable rather than magic.

## Exercises (`hvae.py`)
1. `hvae_elbo` — the general Markovian HVAE bound (test: it lower-bounds an exactly computable `log p(x)`).
2. `fixed_noising_encoder` — replace the learned encoder by `N(√(1−β) z, β I)`.
3. `composed_marginal_coeffs` — the closed-form `q(z_i | x)`; prove it by induction first.

```bash
uv run pytest exercises/04_hvae_to_diffusion
```

## Experiments
1. **Learned HVAE.** Train a 2-layer HVAE on `sample_2d("moons")` with MLP encoders/decoders. Compare samples to a 1-layer VAE.
2. **Freeze the encoder.** Replace the encoders by `fixed_noising_encoder` with L = 2, then L = 10, then L = 100 (constant β chosen so `ᾱ_L ≈ 0`). Only decoders are learned now. Plot samples vs. L.
3. Notice with large L that training samples *every* layer each step is wasteful — the closed form from (3) lets you train a *single, randomly chosen* layer per step. That observation is exactly what module 05 exploits.

## Check your understanding
- Why does a learned hierarchical encoder tend to make upper latents useless (posterior collapse), and why can't that happen once the encoder is fixed?
- Why must the decoders' variance be small when β is small? (Hint: the reverse of a small Gaussian step is ≈ Gaussian, book p. 58.)
