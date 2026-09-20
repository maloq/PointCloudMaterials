# MACE order-preserving regularization comparison

Launched 2026-09-20. One seed; existing native Al Lee-MEAM data only.
16 core fits plus three development-selected longer continuations. Every encoder
gets matched frozen linear/MLP crystallization readouts; three descriptor/condition
baselines add six readouts (44 probe fits total).

- Regularizers: SIGReg, VICReg variance/covariance, and EpiJEPA-inspired geometric reservoir.
- Projectors: nonlinear, linear and identity, with declared normalization variants.
- New anchors: q4/q6/w4/w6, averaged q6, coherence, local density and coordination.
- Most fits adapt a common development-selected width64 checkpoint; two start from scratch.
- Batch 512, compiled BF16, 768 updates per core fit; selected continuations add 1,536 updates.

Eight workers started: node53 (three H100), node59 (two RTX PRO 6000), node61
(one RTX PRO 6000), node50 (two L40S). New one-GPU jobs 1001284–1001287 have four-hour
allocations. Existing jobs 1000616 and 1000818 provide the original four GPUs.
Workers share a locked queue and checkpoint before allocation expiry.

[Scientific protocol](../../../experiments/neighborhood_jepa_regularization_20260920/README.md)
· [Operations](../../../docs/neighborhood_jepa_regularization.md)
· [Prior large-run results](../large-20260920/RESULTS.md)
· [Launch records](technical/launches.json)
· [Exact tasks](technical/tasks.json)

Validation: 23 new objective/geometry/replay tests plus 27 existing regression tests
passed. Compiled B512 VICReg and Epi smoke fits completed training, validation,
checkpoint saving and metrics export. A compiled checkpoint extraction passed
with the frozen crystallization adapter and protected-source ancestry check.
These smoke tests establish execution, not prediction quality.

Update this report after all fits/probes complete. Training metrics appear in
`tables/`, and frozen prediction metrics in `crystallization/tables/`.
`CRYSTALLIZATION.md` updates as full assays finish. Missing results are pending.

The existing encoder already uses per-observation LayerNorm, so there is no
BatchNorm correction in this study. The Epi method is a geometric adaptation,
with scale-controlled features and a frozen random MACE reservoir.
The historical `geometry-baseline` includes velocities; the added
`geometry-only-baseline` supplies the positions/order-only comparison.
