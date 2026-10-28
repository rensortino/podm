# 13 — (Optional) Diffusion models and optimal transport

**Book:** Ch. 7 "(Optional) Diffusion Models and Optimal Transport" (pp. 214–248). Optional appendix: B.3 Wasserstein gradient flows (p. 466).

## Why this module
Straight, non-crossing trajectories make few-step sampling easy (modules 15–17). OT is the formal notion of "shortest" transport, and Schrödinger bridges are its entropic/stochastic version. This chapter tells you exactly when diffusion's ODE/SDE coincide with them — and why rectified flow and minibatch-OT coupling help.

## Exercises (`ot.py`)
1. `monotone_map_1d` — 1D OT is sorting.
2. `transport_cost`
3. `path_straightness`

```bash
uv run pytest exercises/13_optimal_transport
```

## Experiments
- 1D: train flow matching (module 11) on a bimodal 1D distribution. Compare the PF-ODE map noise→data to `monotone_map_1d`. (In 1D any monotone map is OT, and the PF-ODE map is monotone — why?)
- 2D: compare `transport_cost` and `path_straightness` of the noise→data pairing from: independent-coupling FM, minibatch-OT FM, one reflow iteration. Relate to §7.5 "Is diffusion model's ODE an optimal map?".

## Check your understanding
- State the Schrödinger bridge problem and how the diffusion SDE relates to it (§7.4).
