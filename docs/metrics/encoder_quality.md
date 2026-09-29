# Native MACE quality assessment v1

This evaluation freezes the latest selected supervised native MACE encoders and
the fixed epoch-12 VICReg/Epi-variance exports before fine-tuning. It does not
train or select encoders. The six supervised checkpoints retain their original
minimum-validation-NLL selection after at least 12 epochs of a 24-epoch run.
All encoders use the recorded 128-channel, 128-D projected-residual export,
two MACE blocks, 5 Å edges, nearest-80 candidates cropped at 8 Å, constant atom
channel and no additional halo. The native `ir_mul` fused cuEquivariance runtime
is compiled; inference uses float32 without TF32. No temperature/age/time
covariates enter any new predictor or control. Material length normalization
is preprocessing only. Encoder and predictor contexts are recorded separately.

## Input identity, geometric controls and static structure

Checkpoint SHA-256 and native tensor-producer hashes must agree with the frozen
training implementation; state dictionaries load strictly. Existing fixed-cohort
features are reused only for the exact checkpoint and sample IDs, with numerical
replay on 64 fitting and 64 test observations. Replay and geometric agreement
use absolute/relative tolerance 2e-4. These are numerical verification controls,
not scientific fitting runs, and create no W&B runs.

Five controls use 128 patches: repeat, proper rotation, neighbor permutation
(center remains index zero), translation followed by recentering, and a periodic
image representation followed by minimum-image recentering in a 64 Å box.
The latter is a local coordinate-convention check, not a validation of full-cell
trajectory extraction. Export maximum absolute and RMS differences. Padded
disconnected atoms carry zero attributes/weights. Every batch has capacity
256 × 80, including the last partial batch; unused exports are discarded.

Additional noise uses the existing `robust_onset.metrics.perturb_patch` producer:
3D RMS fractions 0.001/0.005/0.01/0.03 of the mean center-to-twelve-neighbor spacing,
fixed center, seed 20260926. Edges/radius masks are rebuilt, but the 80 candidates
remain fixed. `geoframe_evolution.metrics.perturbation` gives median/p95 response
normalized by the same 128 clean observations' independent-pair RMS. This differs
from the all64 fitting-reference noise metric retained in original evaluations.
No ranking/AP-loss function from that historical module is invoked.

Nine static frames (three each Al/Ta/Zr) retain the September23 outcome-independent
4096-anchor sample plus matched neighbor pairs, spatial split with an explicit
buffer, and 1024 topology anchors. Missing reference artifacts were regenerated
with the unchanged reference producer and assay recipe. Before evaluating a
model, original source checksums, 80-neighbor coordinates, and reference class
counts must match the saved native-screen records. New label arrays receive new
checksums; old result artifacts are not overwritten.

Equal-distance neighbor ties reordered 24 of the73728 regenerated patches.
Coordinates agree after a minimum-cost bijection within each affected patch
(absolute/relative tolerance2e-6); matching neither replaces nor drops points.
Inference uses the original saved native coordinates, and permutations have a
separate numerical invariance check. A per-frame agreement receipt is retained.

`geoframe_evolution.evaluate.frame_metrics` supplies PTM/context/fault probes,
distance-matched boundary AUROC with shuffled/collapsed controls, K=7 held-out
contingencies and nonbulk AMI. `encoder_parameter_search.metrics.frame` supplies
fit-normalized liquid-order/topology ridge errors, raw-Euclidean 10-neighbor
physical error, density controls and nonbulk classification. Their definitions
are unchanged; see those original metric contracts. Table summaries average
defined values across three frames per material, not independent-source effects.
Undefined classes remain undefined. Al fault labels do not transfer to HCP Zr.

All static configurations are relaxed. Observed-trained encoders are measured
in transfer, explicitly distinct from their deployed observed prediction inputs.
The fixed material length normalization is exactly the latest training recipe's
Al reference divided by its material-specific scale. Generating potentials of
these static snapshots remain unknown. Static transductive/single-configuration
evidence must not be called independent dynamic generalization.

Spatial plots keep exactly eight times the earlier slab sample and marker area
1.25 points². Dense points only change visualization; metric rows remain fixed.
From the28 September plotting repair, native evaluation retains the actual
K-means estimator returned by `frame_metrics`, saves its cluster centers, and
uses that estimator to assign dense points. It no longer refits K-means for
rendering: even a fixed seed can produce a different local optimum when the
numerical thread context changes. The anchor-label equality check remains and
prediction uses the same single-thread numerical context as the metric fit.
Clustering parameters, metric populations and numerical score definitions are
unchanged. Existing completed exports keep their original contracts.
K-means fits the original spatial fitting side. UMAP fits the original 4096
anchors for visualization; it is not a temporal-distance or clustering metric.

## Fixed Al64 predictive information

Use release `e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d`,
all64 track, unchanged roles and all 126545 rows: 43523 train, 20883 selection,
16848 calibration and 45291 test. No model-specific row drops or source splits.
Observed and relaxed predictors remain separate. Original-MD first sustained
FCC/HCP/BCC onset is the outcome; the ancestry audit indicates predominantly
existing-crystal arrival. This does not test regional nucleus birth.

Fresh frozen readouts use the established `supervised_onset.evaluate.readout`:
linear or one-hidden-layer 128-unit SiLU hazard, AdamW lr 0.001, weight decay
1e-4, clipping 5, natural-source importance weights, 24 full passes, batch 256,
and minimum selection NLL among eligible evaluations from epoch 12 onward.
Evaluate every 128 updates; 4104 updates total. Input normalization fits train
sources only. Inputs are z (128), physical (32), or joint (160). Physical features
are the existing 24 radial counts, 2 weighted counts and 6 l=2/4/6 bond-order
powers, computed on the same input domain. They are evaluation readouts/controls,
not physical-reconstruction encoder pretraining. Descriptor controls are fitted
once per domain. Diagnostic probes retain local progress, summaries and selected
checkpoints; they create no W&B runs. Original encoder training runs/checkpoints
remain unchanged.

The constant control is the source-weighted fitting event-frequency distribution
(categorical maximum likelihood), evaluated on all roles without additional inputs.

`score` applies the existing shared increasing calibration map, fitted only on
calibration sources at 3/6 ps. Report raw and calibrated log loss, Brier and AP
at 3/6/12 ps on selection/calibration/test, with counts. AP is diagnostic only.
Also report raw/calibrated categorical event NLL, reconstructing six event-bin
probabilities by cumulative differences and clipping observed probabilities at
1e-12 before the logarithm.
Whole-source paired proper-score intervals use 1000 bootstrap draws of the
30 test roots, seed 20260926; compute within-root means then average roots.
Delta is candidate minus baseline (negative is better). Comparisons are z minus
physical, joint minus z, and joint minus physical, for matched readout capacity
and input domain. Intervals condition on fitted models; no seed or search
uncertainty is implied. No AP/bootstrap selector is used.

## Native embedding neighbors and future outcomes

For each selection/calibration/test row, retrieve fitting-only neighbors by raw
embedding Euclidean distance. The physical-distance control uses its 32 channels
standardized with source-weighted fitting means/scales, minimum scale 1e-5.
Distances use float32 tensor arithmetic. Query/reference sources are disjoint.
Fitting rows receive a prior placeholder and never contribute to assessment.

Neighbor votes have inverse fitting-source row-count weights rescaled to mean
one over the full fitting population. Add eight equivalent fitting-prior votes
to smooth each cumulative probability. Candidate k values are 16/64/256;
select the neighbor readout by selection-source six-category event NLL. The
six categories are the five original hazard bins plus no event by 12 ps.
Score/calibrate on the same population and with the same proper-score methods
as neural readouts. This is a supervised evaluation readout; it never selects
an otherwise self-supervised encoder. The uncalibrated comparison most directly
measures native-neighbor outcome organization; calibration is separate.

## Relationship to actual temporal changes

Observed-input models additionally encode eight outcome-independent origin
frames per test source, with all 64 fixed center atoms and their next observations
at exactly 0.75 ps. Use the existing dense-observed cache and draw seed 20260926,
sampling origins without replacement from frames with an available next frame.
The final set contains 512 pairs/source and 15360 pairs over 30 sources. Save
frame IDs and observation checksums. This is not a new source split.

At each endpoint compute the same 32 current-geometry descriptors. Latent jump
norms use sqrt(2 × fitting-state covariance trace) as their reference; descriptor
changes use fitting-source means/scales from the all64 at-risk population.
Report per-source Spearman association of latent-jump magnitude with RMS
descriptor change, then the mean of defined source correlations, plus per-source
jump and descriptor-change RMS. This is an association diagnostic. It does not
match atomic displacement, distinguish all reversible motions from rearrangements,
or prove causal sensitivity. Nearest-neighbor identities may change between
frames; no coordinate-slot difference is called atomic displacement.
Dense relaxed trajectories are unavailable and are never replaced by observed
trajectories for a deployed relaxed-input model. Existing full dense spectra,
noise and temporal diagnostics remain separately linked, with original definitions.

## Artifacts and limitations

Each evaluation retains checkpoint/source identities, input contracts, metric
JSON, predictions with full sample IDs, selected readout checkpoints and local
tracking receipts. New encoder feature caches share the existing globally bounded six-entry
LRU with active leases; checkpoints/predictions/metrics are not evicted. Historical
durable feature exports are read and verified rather than copied or deleted.

This study has one training seed per latest recipe and reused historical test
sources. A failed/missing stage cannot count as completion. Regional-emergence
labels and richer raw-geometry add-back predictors are not available in this
release; the joint probe tests the declared 32 descriptor add-back only. Classical
references, density controls, probabilistic readouts and geometry controls answer
different questions; there is no weighted universal encoder score.


## Mechanism queue extension (2026-09-26)

Mechanism runs also load explicit initial/adapted checkpoints and carry exact training-context metadata. Existing quality scores and default128-unit readout equations are unchanged.


Tracking revision (2026-09-26): diagnostic frozen readouts and per-checkpoint
evaluations keep their logs and results locally. Associated final scores update
a recorded scientific training run through the API, without creating or
restarting runs. Scientific training remains online. This changes logging and
validates identity/hash before cached readout reuse; objectives, selectors,
metric calculations and historical exported definitions are unchanged.


## Execution refactor

The code-cleanup revision consolidates artifact export, preparation, checkpoint
and execution helpers. Scientific formulas, rows, weights, fitting populations
and selectors are unchanged. New table exports include a per-table hash and
definition binding. Historical exported definitions and frozen source snapshots
remain authoritative; changed implementation hashes require a new export revision.
