# 18 — (Optional) Discrete diffusion: masked diffusion

**Book:** Epilogue, §12.1 "The Density-Transport Backbone" (pp. 411–415), §12.2 "Beyond Continuous States: Probability Evolution on Discrete Spaces" (pp. 415–434).

## Why this module
Text and other discrete data have no scores or velocities, but the density-transport view survives: a continuous-time Markov chain moves probability mass, and masked (absorbing-state) diffusion gives a clean ELBO that is just a time-weighted masked-LM cross-entropy.

## Exercises (`discrete.py`)
1. `mask_tokens` — forward process.
2. `masked_diffusion_loss` — the continuous-time ELBO.
3. `sample_masked` — reverse unmasking.

```bash
uv run pytest exercises/18_discrete_diffusion
```

## Experiments
- Binarised MNIST as sequences of 784 binary tokens (K = 2): small transformer or the `SmallUNet` with an embedding layer in front. Sample with 10, 50, 784 steps.
- Character-level text (e.g. a public-domain book): small transformer, generate with different step counts. Compare with an autoregressive model of the same size.

## Check your understanding
- Where does the 1/t weight come from? Relate to the ELBO weighting of DDPM (module 05).
- Why can't many tokens be revealed per step without quality loss? (Factorised reverse kernel.)
