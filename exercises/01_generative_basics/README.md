# 01 — Deep generative modeling basics

**Book:** Ch. 1 "Deep Generative Modeling" (pp. 21–33). Skim §1.2 (prominent models) — we revisit VAEs, EBMs and flows in depth later.

## Why this module
Every model in this course is trained by making a model distribution `p_θ` close to `p_data` under some divergence. Before touching diffusion, get fluent with: evaluating log-densities stably, estimating KL by Monte Carlo, and seeing *which* KL you minimise changes the answer.

## Exercises (`gen_basics.py`)
1. `gmm_log_prob` — Gaussian mixture log-density with `logsumexp`.
2. `gaussian_kl_1d` — derive the closed-form Gaussian KL on paper first, then code it.
3. `kl_monte_carlo` — the estimator you will use whenever no closed form exists.
4. `fit_gaussian` — fit one Gaussian to a bimodal target with forward KL (= MLE) and reverse KL (needs the **reparameterisation trick**, which reappears in the VAE).

```bash
uv run pytest exercises/01_generative_basics
```

## Experiments (write `run.py` yourself)
- Plot the two fitted Gaussians over the bimodal density. Explain *mode covering* vs *mode seeking* in one sentence each.
- Run reverse KL from `init_mu=0, init_sigma=1` and from `init_mu=1, init_sigma=0.5`. Why does the first get stuck? (Reverse KL is non-convex in (μ, σ); the symmetric start is a saddle.)
- Fit a K=8 GMM by gradient-ascent MLE to `podm.data.sample_2d("8gaussians", ...)`. What happens with K=4? With bad initialisation?

## Check your understanding
- Why can't you directly train with reverse KL on real images? What does each model family in Ch. 1's taxonomy (p. 32) do to get around intractable likelihoods?
