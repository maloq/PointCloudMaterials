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
