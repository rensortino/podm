# 14 — Guidance and controllable generation

**Book:** Ch. 8 (pp. 251–282): §8.1 prologue (Bayes for scores), §8.2 classifier guidance, §8.3 classifier-free guidance, §8.4 training-free guidance, §8.5 from RL to DPO for alignment.

## Why this module
Conditioning is just Bayes' rule on scores: `∇ log p(x|y) = ∇ log p(x) + ∇ log p(y|x)`. Guidance turns the `p(y|x)` knob past 1. CFG removes the classifier by noticing that the guidance term is a *difference of two scores the same network can output*.

## Exercises (`guidance.py`)
1. `classifier_guided_score` — test: at w = 1 you get the exact conditional score of a 2-class mixture.
2. `cfg_combine` — test: with exact scores, CFG at any w equals classifier guidance at the same w.
3. `drop_labels`
4. `dps_guidance` — gradient *through* the denoiser.
5. `diffusion_dpo_loss`

```bash
uv run pytest exercises/14_guidance
```

## Experiments
- Conditional FM or DDPM on `sample_2d_labeled("8gaussians")` with `TimeMLP(num_classes=8)`, labels dropped with p = 0.1 (null label = 8). Sample class 3 with w ∈ {0, 1, 2, 4, 8}. Watch diversity collapse and samples overshoot (book p. 262).
- Train a noisy classifier `MLP(3, 8)` on `(x_t, t)` and do classifier guidance with an unconditional model. Compare to CFG.
- MNIST with `SmallUNet(num_classes=10)` + CFG: the canonical demo.
- DPS: inpaint MNIST (mask = operator) with an unconditional model. Sweep the guidance scale.
- Stretch: implement Diffusion-DPO on 2D: "prefer" samples with x₁ > 0 and fine-tune.

## Check your understanding
- Show CFG with w > 1 samples from `p(x) p(y|x)^w` *only approximately* — where does the approximation enter for t > 0?
