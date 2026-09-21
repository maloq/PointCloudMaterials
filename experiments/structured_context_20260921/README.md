# Symmetric spatial context for frozen MACE and GATr forecasting

Question: can a consistent, spatially balanced arrangement of local observations
improve local crystallization and structural-trajectory prediction?

The previous context contained the center plus three representatives in each of
two annuli. The sampler maximized coverage without antipodal balance; its smooth
25 Å weighting could nearly erase selected outer representatives. This experiment
replaces that observation layout and the context head together.

## Observation and predictor

The tracked center has 24 surrounding queries: 12 cuboctahedral directions at
10 Å and the same 12 directions at 20 Å. Directions are permutations of
`(±1, ±1, 0)/sqrt(2)`. Each shell is exactly antipodally symmetric and has an
isotropic second moment. These are **query positions**; real liquid atoms cannot
be made symmetric. Minimum-cost unique assignment chooses a real atom for each
query at every observed frame, with a maximum allowed displacement of 4 Å.
Every local encoder still receives an unmodified, atom-centered neighborhood.

The predictor receives the query, actual displacement and frozen feature for
every slot. Two blocks alternate within-frame spatial attention and causal
temporal attention along each query slot, then pool. Temporal correspondence is
between spatial query slots, not necessarily the same neighboring atom. The
tracked central atom identity is constant. Center, inner shell and outer shell
each have equal total attention/pooling weight; there is no cutoff taper on these
discrete query slots. Each local encoder retains its own smooth atomic support.

The stencil stays fixed in the simulation-box frame. Cubic rotations permute
slots; the head is invariant when geometry and query frame rotate together.
Re-sampling a rotated cloud with a lab-fixed stencil is not generally SO(3)
invariant. Avoid claiming otherwise. Query reassignment can still introduce
changes as atoms move across assignment boundaries.

## Controlled population and fits

Existing 150 independent Al trajectories, 16 tracked centers each, original
90 training / 15 selection / 15 calibration / 30 test ancestry split. No new
simulations. At-risk origins stay on the 3 ps grid and use only observations at
−48, −12, −3 and 0 ps. The existing observed descriptor-history and shell-count
auxiliary inputs are supplied to both backbones; these are not pure-embedding
readouts. Future observation membership or features never enter a causal input.

Eight fits: two frozen backbones × direct whole-path, deterministic autoregressive,
event-stratified mixture and diffusion. All predictor heads start from scratch.
One seed; batch 128; LR 1e-4; one warmup epoch then cosine decay; 36-epoch cap,
eight unimproved selection epochs of patience after at least six epochs. Head
width 128, four attention heads, two alternating spatial/temporal blocks. Remaining
method settings come from the prior validation-selected family promotions.

MACE uses the original transfer checkpoint. GATr uses **exactly step 3072** from
`output/structural_pretraining/gatr-vicreg-20260917/technical/encoder.pt`, the source
of `output/structural_static/gatr-vicreg-step3072-al-20260918`. Its historical class
source is recovered from commit `32b4ca88`, checked against the checkpoint's recorded
source hash and replayed against that static analysis. It is not loaded into the
newer GATr architecture. Both pretraining ancestry checks protect calibration/test.

The checkpoints have different local supports: MACE 7.94 Å; this GATr 16.87 Å.
The spatial query layout is shared, but this is not an architecture-only comparison.
Query centers can be as far as 24 Å away; total atomic reach includes each
encoder's local support. Preserving the specifically requested GATr is intentional.

## Evaluation and interpretation

Predict 32 structural states at 3 ps intervals through 96 ps, plus the local onset
CDF at 0.75 ps resolution. Each backbone predicts its own 128-dimensional latent,
the same 128 physical channels, eight bond-order channels and PTM crystallinity.
Compare shared physical/event outcomes across backbones; their raw latent MSEs
are not comparable. Use the original source-weighted likelihood/Brier/AP/calibration,
timing with misses, physical MSE/CRPS and sampled-center spatial metrics, including
the ≤12 ps block. All selection/evaluation forecasts are fully open-loop.

Historical irregular-context fits are reference results, not a pure matched
ablation: the new head, initialization and common 48 ps history also change.
One seed does not characterize optimization uncertainty; the previously inspected
test population is not a fresh confirmatory test. All eight fits and evaluations
are now complete; the linked run report retains the original window-based scores.

Recipe: `configs/crystallization_transfer/symmetric_mace_gatr_20260921.json`.
Results: `output/crystallization_transfer/symmetric-mace-gatr-20260921/RESULTS.md`.
Execution and exact-resume instructions: [workflow](../../docs/structured_context.md).

## Event-aligned follow-up analysis

Motivation: following the same events from progressively earlier forecast origins
separates loss of predictive information with lead time from a changing event mix.

The completed eight fits are visualized using shared source-separated examples,
physical/Brier curves, separate training-fitted MACE/GATr UMAPs and actual symmetric
query assignments. The fixed-event diagnostic reuses one local onset and one seeded
same-source control at every nominal lead in 3/6/9/12/18/24/36/48 ps. Origins round
down to the saved 3 ps grid. Cases require predictions at all offsets; controls must
be at risk at those origins and have no onset by the reference case's onset time.
All selection uses identities/labels/availability, never model scores.

AP uses the probability by the reference event time, with balanced case/control
weights and equal-source averaging. Its 50% prevalence differs from the original
window population. Timing error includes every retained event and uses the 96 ps
restricted mean, retaining survival mass rather than dropping missed alarms.
Whole-source paired bootstrap intervals preserve the dependence across origins,
events and reused controls. See [exact definitions](../../docs/metrics/structured_figures.md).

Reproduce with:

```bash
conda run -n pointnet-torch214 python -m src.research.structured_context.figures --config configs/analysis/structured_context_figures.json
```

Gallery and captions: `output/crystallization_transfer/symmetric-analysis-20260921/README.md`.
This analysis does not fit predictors or change their selected checkpoints.

The matched cohort contains 287 distinct local onsets from 27 sources; 118 onsets
lack an eligible origin at one or more requested offsets and five lack a control.
For MACE direct, matched AP falls from 0.851 at nominal 3 ps to 0.764 at 12 ps,
0.654 at 24 ps and 0.538 at 48 ps; chance prevalence is 0.5. GATr and other heads
show the same broad loss of discrimination with lead. These are retrospective
matched-cohort diagnostics, not natural-prevalence event classification scores.
The all-event restricted timing MAE is substantially larger than the historical
detected-only short-horizon timing error, exposing broad predicted timing and
survival probability. No best-of-model or best-of-sample event selection is used.
