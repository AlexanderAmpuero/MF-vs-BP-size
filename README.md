# MF-vs-BP-size

> We replicate and test Mono-Forward, measure where and why it falls behind backprop at larger image sizes, and test one fix.

**Status:** skeleton + BP and MF models. No results yet. Everything below marked *TBD* is filled in only after the runs exist.

## Research question
Why does Mono-Forward (MF), a backprop-free method, lose ground to backprop (BP) as input size grows, and will our potential fix help? A clean null result counts, provided the cause isolation is rigorous.

## Plan
| Phase | What | Status |
|---|---|---|
| 0. Replicate | Reproduce published MNIST numbers for BP, MF (then FF). Sanity gate before anything else. | not started |
| 1. Dimension experiment | BP vs MF on MNIST upscaled to 28/56/112 px (784 / 3,136 / 12,544 inputs); locate where the gap opens and which layer loses information; test one reconstruction-penalty fix. | not started |
| 2. Sequential learning | BP, FF, MF on digits 0-4 then 5-9; forgetting, Task-IL vs Class-IL, per-layer probes. | not started |

Pre-declared tolerances and decision rules: [`docs/PROTOCOL.md`](docs/PROTOCOL.md). Citations and statuses: [`docs/CITATIONS.md`](docs/CITATIONS.md).

## Definitions
**Backprop baseline.** Hidden layers $a_i = \mathrm{ReLU}(W_i a_{i-1} + b_i)$, head $z = W_o a_L + b_o$, loss $\mathcal{L} = \mathrm{CE}(z, y)$, gradients flow through all layers.

**Forward-Forward goodness (Hinton, 2022).** For layer $i$ with activations $a_i$, $g_i(x) = \sum_j a_{i,j}^2$. Each layer is trained to give high goodness on positive data and low goodness on negative data (label embedded in the input; negatives use a wrong label). Prediction runs the network once per candidate label and picks the highest goodness.

**Mono-Forward local loss (Gong et al.).** Layer $i$ has a goodness matrix $M_i \in \mathbb{R}^{h_i \times C}$ giving class scores $s_i = a_i M_i$, trained with $\mathcal{L}_i = \mathrm{CE}(s_i, y)$. Inputs to each layer are detached, so no gradient crosses a layer boundary. Prediction: final layer's scores (primary) or the sum over layers (secondary).

**Update schedule.** Following the MF authors, each layer gets one local update per batch before moving to the next, so all layers train simultaneously. Layer freezing is *not* part of FF or MF; it is a protocol choice and is not used here.

## Repository layout (To be filled)
```
models/        base.py (interface), bp.py, mf.py   (ff.py planned)
common/        data.py (MNIST + upscaling), utils.py (seeds, env logging)
experiments/   replicate.py   (dim_sweep.py, sequential.py planned)
analysis/      marimo notebooks: probes and visuals; sequential results (planned)
tests/         locality, parameter accounting, shapes, toy-learning tests
docs/          PROTOCOL.md, CITATIONS.md
results/       JSON per run (config, seeds, versions, hardware, git commit)
```

## Quick start
```bash
pip install -r requirements.txt
pytest                                   # run first: checks MF locality and parameter parity
# Include tests for bp and mf here based on papers. We are replicateing their works
```

## Limitations
- MNIST is easy and upscaling adds no information, so real high-resolution data may behave differently.
- Class count is fixed at 10 on purpose: scaling sweeps can confound output dimensionality with fine-grained difficulty.
- In an MLP the goodness matrix scales with width, not input size, so the documented MF memory blow-up (a convolutional effect) is not tested here.

## Future work
Wake-sleep-style consolidation (FF-style local learning awake, backprop from the most recent layers downward in a sleep phase; needs stand-ins for old data); convolutional/high-resolution MF with pooling before the goodness matrix; testing SFFA's Task-IL clash with layer-wise goodness. EWC and replay baselines, CKA, and hidden-width sweeps are deliberately out of scope.

## Results
*TBD.*