# 03 — Variational Autoencoder

**Book:** §2.1 "Variational Autoencoder" (pp. 44–54). Supplementary proofs: Appendix D.1 (p. 486).

## Why this module
DDPM is a VAE in disguise (module 04 makes this literal). The ELBO, Gaussian KLs, the reparameterisation trick and the "encoder vs. decoder" view all carry over verbatim — only the encoder becomes fixed and the latent becomes as large as the data.

## Exercises (`vae.py`)
1. `reparameterize`
2. `gaussian_log_prob`, `bernoulli_log_prob`
3. `gaussian_kl`, `kl_to_standard_normal` — derive both on paper first.
4. `elbo` — the test uses a linear-Gaussian model where `log p(x)` is exact: with the *true* posterior the ELBO is tight, with a wrong one it is strictly below. Bonus: show that the fully Monte Carlo estimator `log p(x,z) − log q(z|x)` has *zero variance* under the true posterior, while the analytic-KL version does not. Work out why the gap equals `KL(q(z|x) || p(z|x))`.

```bash
uv run pytest exercises/03_vae
```

## Experiments (`train.py`, yours to write)
- Train a VAE on binarised MNIST (`mnist_loader(binarize=True)`, Bernoulli decoder, `podm.nets.MLP` encoder/decoder, z-dim 16). Plot reconstructions and prior samples.
- Swap to a Gaussian decoder with a fixed variance; vary that variance. Why do samples get blurry? (Book p. 52–53: the Gaussian decoder's MSE ⇒ posterior mean ⇒ blur.)
- Train with β·KL for β ∈ {0.1, 1, 4}. Track the per-dimension KL — which latents collapse?

## Check your understanding
- Write ELBO = log p(x) − KL(q(z|x) ‖ p(z|x)). What does each training signal push on?
- Why is a *single* latent layer with a simple Gaussian prior limiting? (Motivates the hierarchy → module 04.)
