# Does simulator-response supervision improve an atomistic predictor?

The previous oracle pilots verified fixed-MACE derivatives and the toy controls
favored response supervision. They did not train an atomistic encoder. This
experiment tests that missing step with a complete observation of a small cell.

Three arms train the same native full-cell MACE128 plus nonlinear prediction head:
values8, values+responses8 on the same stochastic paths, and values32. Three paired
seeds give nine fits.32 perturbed-FCC configurations train,8 select and16 new
configurations evaluate. Old numerical-development geometries remain training
only. All configurations share an FCC prototype; this is a conditional synthetic
mechanism experiment, not independent-liquid generalization or an Al64 resplit.

The fixed MACE-MPA-0 BAOAB experiment observes20/100fs smooth radial-statistic
changes through256 fixed joint-prefix Fourier features. The response target is
the derivative of those simulator features with respect to the complete initial
configuration, in two zero-translation directions, averaged over8 branches.
Every branch is fresh. Test value and derivative streams are independent.

Primary comparison: independent-configuration future-feature error and directional
response error, corrected for finite-shot noise, with paired configuration
intervals. All checkpoints select by the same32-shot validation feature likelihood.
Report actual acquisition/training costs;32 value shots are an extra-label control,
not an exact cost match. No active acquisition is introduced in this first test.

Input responses must pass full-student finite differences and produce encoder
parameter gradients. Physical AD must pass common-random-number finite differences
at the actual100fs horizon. Numerical checks stay local. All scientific fits are
online, and interrupted branches/optimizer states resume from durable receipts.

The complete-cell student requires a differentiable periodic graph. The existing
local graph helper is decorated with no_grad and cannot supply the required input
Jacobian. Only graph membership/image selection is detached in the new consumer;
radial/angular features and the encoder/head stay differentiable. The simulator
potential remains frozen and distinct from the learned native encoder.

[Recipe](../../configs/response_atlas/atomistic_training_20261001.json) ·
[Definitions](../../docs/metrics/response_training.md) ·
[Execution](../../docs/response_atlas.md#atomistic-response-training).

## Completed results, October 2, 2026

All56 configurations and nine fits completed. On the16 held-out configurations,
averaged over three training seeds, the full256-feature noise-corrected errors are:

| Supervision | Future-feature MSE | Directional-response MSE |
| --- | ---: | ---: |
| Constant training prior / zero response | 0.950059 | 0.686926 |
| Values8 | 0.033864 | 0.522076 |
| Responses8 | 0.008677 | 0.148839 |
| Values32 | 0.033521 | 0.520655 |

Responses8 reduces future-feature error by74.4% and response error by71.5%
relative to Values8. It also improves both errors relative to Values32 and wins
at each of the three paired training seeds. The paired full-feature difference
versus Values8 is -0.025187 (95% configuration-bootstrap interval
[-0.043503,-0.010417]); the response-error difference is -0.373237
([-0.521869,-0.225844]). Improvements occur in both the20fs block and the joint
20/100fs block. Intervals condition on this fixed synthetic training set and
average the three training seeds before resampling held-out configurations.

These are results for the declared200-update training budget. Eight of nine
selected checkpoints are at epoch191 or later; convergence is not established.
Response fits use about20.2 minutes of optimization/selection each, versus1.52
minutes for value-only fits. A longer value-only optimization control, including
a matched training-compute comparison, is therefore a useful next check before
attributing the entire improvement to information unavailable to value training.
Values32's small improvement alone does not establish that further value-only
training cannot close the gap. Independent liquid states and longer physical
horizons remain necessary to connect this result to crystallization prediction.

The frozen exports are in
`output/response_atlas/atomistic-training-20261001/analyses/comparison-v1/`:
`tables/summary.csv`, `tables/paired-contrasts.csv`, `tables/parent-errors.csv`,
`tables/costs.csv`, and `plots/response-comparison.png`. All nine saved prediction
and selected-checkpoint hashes were verified when reporting these results.
