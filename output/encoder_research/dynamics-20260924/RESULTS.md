# State dimensions and temporal stability: existing exports

Completed September 24, 2026. **19 representations analyzed; no new training,
encoder inference or simulation.** Eight current checkpoints yield 12 separate
encoder/projector exports; seven older representations use the dense matched
trajectory cohort. These are separate populations and observation protocols.

The latest high-LR MACE states concentrate variation in a few linear directions:
whole-export participation ranks **2.40 / 2.03**, movement participation ranks
**2.73 / 2.56** at 108 ps. Their corresponding lower-LR controls have state ranks
2.45 / 2.04 and movement ranks 2.70 / 2.56. The AP increase previously observed
at high LR therefore does not require a large increase in these effective
ranks. This is descriptive co-occurrence, not evidence about its cause. Small
variance directions can still carry predictive information.

The current four-snapshot data **cannot establish short-time smoothness**.
Each atom has four observations separated by 120, 108, 108 ps. Per-track centered
rank is consequently at most three, regardless of embedding width. The 108 ps
pairs sample later trajectory intervals than the 120 ps pairs; differences
between these two points also reflect time/phase populations, not only lag.
There are 480 / 240 evaluation pairs at 108 / 120 ps across 15 development roots.
Pre-onset-conditioned pairs can be absent at these coarse lags; coverage is in
the tables. No interpolation fills this missing evidence.

## Raw exported encoder/descriptor results

Ranks below are source-weighted participation ratios, not integer matrix ranks
or nonlinear intrinsic manifold dimensions. “Dataset” includes all exported
fit/tune/development observations (2,880 for each current model), whereas movement
uses development sources only. The dense dataset contains 32,040 evaluation and
420 fitting-reference observations with equal source weight, not equal total
weight for those two splits. Jump normalization uses fitting rows exclusively.
The table shows the shortest **requested** measured lag for each protocol.

| Representation | Dataset rank | Noncrystalline dataset rank | Lag (ps) | Movement rank | Movement d95 | Normalized RMS jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gf-mlp-cov1-factor1-s123-0035--encoder | 1.658 | 2.087 | 108 | 1.968 | 7 | 0.751 |
| gf-mlp-cov1-factor1-s456-0035--encoder | 2.758 | 2.553 | 108 | 3.032 | 6 | 0.789 |
| gf-visreg-mlp-factor1-s123-0035--encoder | 1.617 | 2.112 | 108 | 1.916 | 7 | 0.729 |
| gf-visreg-mlp-factor1-s456-0035--encoder | 1.582 | 2.020 | 108 | 1.835 | 8 | 0.735 |
| mace-lr1e-05-distance0-s20260923-4096--encoder | 2.450 | 2.665 | 108 | 2.697 | 4 | 0.849 |
| mace-lr1e-05-distance0-s20260924-4096--encoder | 2.041 | 3.641 | 108 | 2.556 | 5 | 0.809 |
| mace-lr0.0001-distance0-s20260923-4096--encoder | 2.401 | 2.934 | 108 | 2.729 | 5 | 0.850 |
| mace-lr0.0001-distance0-s20260924-4096--encoder | 2.035 | 3.711 | 108 | 2.556 | 6 | 0.811 |
| dense-v6--mace | 1.884 | 2.005 | 0.75 | 1.855 | 2 | 0.723 |
| dense-v6--gatr | 6.386 | 6.289 | 0.75 | 6.313 | 7 | 0.680 |
| dense-v6--tda | 1.172 | 1.373 | 0.75 | 1.825 | 5 | 0.371 |
| dense-v6--soap | 2.495 | 3.913 | 0.75 | 4.059 | 9 | 0.587 |
| dense-v6--bond_order | 1.120 | 1.277 | 0.75 | 2.475 | 4 | 0.278 |
| dense-v6--radial | 2.761 | 4.278 | 0.75 | 5.757 | 8 | 0.661 |
| dense-v6--angular | 1.532 | 3.139 | 0.75 | 3.187 | 7 | 0.513 |

The dense 0.75 ps results refer to the historical v6 MACE/GATr checkpoints.
They do not measure temporal behavior of the latest higher-LR MACE encoder.
On that older cohort, MACE movement PR is 1.86 and GATr 6.31; this difference
by itself says nothing about which representation retains better onset signals.

## Tables, plots and definitions

- [Stability and movement spectra](tables/stability.csv): each physical lag,
  whole/noncrystalline/temperature and available onset-conditioned domains,
  RMS and p50/p95/p99 jumps, uncentered movement versus centered fluctuation.
- [State ranks](tables/ranks.csv): entire available dataset, fitting reference,
  held-out evaluation and within-track covariance, with PR, entropy rank,
  d90/d95/d99, numerical rank and sample ceilings.
- [Each atom trajectory](tables/per-track.csv): direction, reversal, roughness,
  duration, observed lags and state/movement dimensions.
- [Complete eigenvalue spectra](tables/eigenvalues.csv).
- [Frozen formulas](tables/METRICS.md) and
  [implementation identities](technical/metric-contract.json).
- Full metrics and verified input identities are in `technical/<model>.json`.

![Current coarse-time screen](plots/coarse.png)

![Historical dense trajectory comparison](plots/dense.png)

No statistical uncertainty is estimated in this descriptive supplement. Use
whole-source comparisons before claiming a difference. Do not promote a low-rank
or quiet encoder without retained present information and onset responsiveness.

The [AP-focused experiment proposal](../../../docs/encoder_research/ap_experiments.md)
starts with AP-selected frozen heads, then ranking losses and controlled
history/context comparisons, before supervised encoder fine-tuning. Training is
proposed, not submitted. The new diagnostics are also attached automatically to
future evaluations that use the current shared snapshot predictor.
