# Native MACE training-data amount

Maintenance note (16 September 2026): shared motion math and trajectory IO moved
from the discarded frozen-map package into `mace_velocity/motion.py` and
`mace_velocity/sequence_data.py`, with identical function bodies. Calculations,
model architecture and historical run exports are unchanged. Exact old source is
preserved in the [retirement archive](../discarded_frozen_encoder_maps.md).

Protocol `mace_native_data_amount_v1` trains MACE message-passing parameters and
the coordinate/velocity encoder end to end. No fitted map follows the pooled
embedding. Original MLIP pretraining is common to all runs; these are conditional
local-state adaptation learning curves, not training-from-scratch scaling laws.

## Data and controls

Use the 150 independent-melt preparation lineages already assigned to 90 train,
30 validation and 30 development-test sources. Exclude shared-preparation legacy
branches from this independent-source count. All are Al with measured velocities.
Each contributes four tracked centers, nine consecutive native frames (0.75 ps
cadence), and 36 local observations. Source/frame/center identity and selected
position/velocity arrays are verified against the producer of the cached labels.
Atom clouds are rebuilt from trajectories, not from frozen embeddings.

The train counts 10/25/45/90 have 2/5/9/18 sources at each of 400/450/500/510/520 K.
A deterministic ten-source core is shared by all subsets and both replicates.
Each replicate independently orders the remaining sources within temperature;
subsets are nested. Two seeds change those extensions, branch/head initialization
and batch sampling. They are two repeats, not a precise seed-uncertainty estimate.
The ten-source result varies initialization/sampling only because its core is fixed.

All normalization uses only this common core: label means/std, with a single RMS
std per TDA family; odd velocity labels have mean zero and RMS scales. Initial
foundation features use channel means and within-source/frame channel std, floored
at 5% of their median std. These constants are frozen and identical across fits.
The selected data never includes held-out normalization fitting or initialization
from an encoder trained on all current sources. Read-only label reuse does not
reuse the previously trained embeddings.

## Model, objective and optimization

Embedding is the encoder's native 304 channels: 256 smooth-inner pooled scalar
MACE structure channels, 32 velocity-even activity and 16 velocity-odd flow.
The structural block is coordinate-only at inference. Both geometry and relative
velocities affect the motion channels; all tasks can update their shared MACE
backbone. There is no history input, time/source ID input or forecasting target.
Halo message passing and smooth inner pooling retain the existing implementation.
Auxiliary physical heads predict the current 169 labels: bond order 0:16,
instantaneous TDA H0 16:32, H1 32:96, H2 96:160, even motion 160:166 and odd motion
166:169. Structural targets are decoded from the main 256 channels alone.

Each update samples two distinct sources uniformly and one of seven consecutive
three-frame starts per source, retaining all four centers (24 clouds). This
weights middle frames more often; it is identical across counts. All fits use
the same fixed update budget, batch size, learning rates and regularization.
Equivalent passes = cloud presentations/(36 * selected source count); report
actual unique source/observation coverage. The primary table evaluates **last.pt
at the fixed final update**. Also preserve best.pt chosen by validation-only
score; it does not replace the matched-update comparison.

Physical loss is the mean of the four structural family normalized MSEs. Motion
loss is (2 * even MSE + odd MSE)/3. For current batch structure features, subtract
each source/frame's four-center mean and compute channel population variances V.
Temporal loss is mean squared consecutive increment / (2 sum V). Bending is mean
squared time-adjusted second difference / (2 sum V); use exact physical times,
as defined in the consecutive-motion protocol. Denominators remain in the encoder
gradient. A channel variance floor is mean relu(0.5-sqrt(V+1e-6))². This study uses
whole-batch motion penalties; low-order quality is a separate reported audit.

A shared 256->32->(256*8) direction head sees only the detached current structure
state. Reduced QR produces eight orthonormal directions. Direction loss is mean
unexplained increment energy / mean total increment energy. Encoder gradients use
detached bases; basis fitting uses detached states and increments. Basis fitting
cannot smooth a frozen encoder. The basis is auxiliary and absent from deployed
embeddings. Unlike the frozen-map protocol this pilot has no physical-neighbor
basis-consistency term. Numerically clamp variance/energy denominators to 1e-12;
evaluation fails on collapsed reference spread.

Total = physical + motion_weight * motion + ramp * (temporal_weight * temporal +
curvature_weight * bending + direction_weight * direction) + variance_weight *
variance_floor + basis_fit. The validation selection score excludes variance floor
and basis_fit and always uses full regularization weights. Warmup/ramp and weights
are in the immutable run config. AdamW weight decay 1e-4; gradient norm caps 1 for
MACE, 5 for velocity branches plus physical heads, 5 independently for directions.
Exact gradient replay accumulates microbatches without changing the full objective.
An actual H100 check compares direct and replayed gradients and requires an actual
MACE parameter update, then measures final-objective update time and symmetries.

## Exported quality and uncertainty

Physical columns are normalized MSE, averaged within each source and then equally
over eligible sources. They are not percentages or distances in Angstroms. Report
both all observations and low-order observations (true local qbar6 <0.30), a
disordered proxy rather than a PTM phase label. Edges/triples require every
endpoint to meet the population restriction. Counts include eligibility.

For **each fitted encoder**, re-encode the same common training core. Ordinary
reference trace is sum of population channel variances over eligible core groups.
Within-context trace averages eligible source/frame variance traces with at least
two centers. This second denominator removes between-source/frame separation.
These reference populations differ from earlier pair audits; numeric jumps are
not automatically interchangeable with the previously reported 0.45.

`jump_rms` = sqrt(equal-source mean squared 256-channel increment / (2 * reference
trace)); `jump_p95`/`jump_max` use individual eligible edges. Native lag is 0.75 ps.
`within_context_jump_rms` uses the within-context trace instead. Low-order scores
fit their reference spread only to low-order core observations. `bend_rms` uses
source-mean squared bend with the ordinary structure trace. `full_embedding_jump_rms`
uses all 304 channels and their matching core trace; these velocity channels are
not directly required to be slow. Reference normalization makes uniform shrinking
insufficient to improve these scores.

`learned_rank8_explained_energy` = 1 - equal-source mean residual energy divided
by equal-source mean increment energy, using directions predicted from each
current state. This is held-out direction prediction, not a proven intrinsic
dimension or manifold. Jump confidence intervals resample whole eligible sources
500 times and report the 2.5/97.5 percentiles; they do not include seed variability.
The same bootstrap seed pairs source resamples where populations match. Keep both
replicate curves visible. Train/validation/development-test scores expose gaps.

Previously inspected held-out sources are development evidence. This short study
does not establish convergence, all-material transfer, phase classification,
cluster spatial coherence or a new blind generalization result. Learning-curve
plots update after each completed fit; a partial table is not a completed sweep.
