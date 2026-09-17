# Native supervised onset diagnostic — protocol v1

This is the supervised reference, distinct from the label-free physical-mean
encoder. Seed20260919; six hazards for bins ending0.75/3/9/24/48/96 ps.
See the shared [population/metric definitions](local_predictability.md).

The frozen 38,400-row physical index is filtered to at-risk rows, retaining 11,738
training rows. Source/center/anchor and validation-half identity is checked against
both the raw loader and separate label/index artifacts. Raw inputs are strictly
causal. Current x/v, 12-ps real history and repeated-current-frame history have the
same native architecture, radius17 Å, cutoff5 Å, width16 and output128. Each child
receives an identical trained snapshot parent and exactly K additional updates;
snapshot is also continued K. Optimizers and samplers reset consistently.

The onset head is linear in exported state128 and seven known conditions. The
seven conditions (five temperature indicators, time/600 ps and its square) use
frozen mean/std from all physical training rows. Loss is source-balanced joint
discrete-time hazard likelihood. Each effective batch draws eight independent
uniform source choices, then a uniform eligible row from each. No event enrichment.

Selection uses up to64 uniformly selected eligible windows per validation-selection
source, fixed before training. Checkpoints minimize equal-source event-time NLL;
calibration/test are not used for selection. Exports score every eligible row in
selection/calibration/test on the fixed native origin grid. Window-FPR thresholds
use calibration only. Log-loss/Brier/AP, whole-source bootstrap and horizons match
the shared definitions. No dense0.75-ps alarm metric is inferred from sparse scores.

Source/release/condition/label/small-fit and implementation hashes are stored with
checkpoint identity. Best and latest checkpoints preserve weights, optimizer,
sampler and Python/NumPy/Torch/CUDA RNG states. Native model mean physical fitting
checks are a separate training-only architecture gate, not a pretrained core parent.


Table export: 2026-09-17T14:56:03.576464+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
