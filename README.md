# Diffusion Models from First Principles

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

This project provides a hands-on exploration of diffusion models, derived from first principles and grounded in the foundational concepts presented in the [The Principles of Diffusion Models](https://arxiv.org/pdf/2510.21890). The goal is to implement key ideas step-by-step, ensuring deep understanding through code and going back to the basics id necessary.

## 📚 Overview

Diffusion models are a class of generative models that have gained significant attention for their ability to generate high-quality samples across various domains (e.g., images, audio, and molecules). This project breaks down the mathematics and algorithms behind diffusion models into digestible components, with implementations in PyTorch.



## 🚀 Getting started

```bash
uv sync                                   # install dependencies
uv run python -m podm.data                # sanity check: plots the toy datasets to runs/
uv run pytest exercises/01_generative_basics
```

Then follow [ROADMAP.md](ROADMAP.md): 18 incremental modules from maximum likelihood and VAEs, through DDPM, score matching, score SDEs and flow matching, to guidance, fast solvers, distillation and consistency models / MeanFlow. Each module in `exercises/` has a README (book pages, goals, experiments), a stub file to implement, and fast tests that check your implementation against closed-form oracles.
