# Local crystallization forecast assay

This assay concerns the exact tracked center atom of each local embedding trajectory.
It does **not** use the first simulation-wide nucleus as that atom's transition.
The 2026-09-13 protocol draws 64 of the existing 1,024 embedded centers per source,
without replacement, with `SeedSequence([seed, preparation_seed])`. Sources retain
whole-simulation train/validation/test assignments (74/24/27). Units are ps;
frames are 0.75 ps apart. All metrics weight eligible centers/windows equally.
Uncertainty resamples entire simulation sources, preserving correlated atoms and
windows; individual frames are not independent replicates.

## Physical target

The independent structural assay is OVITO polyhedral template matching (PTM),
RMSD cutoff 0.1, at the center of the same 80-nearest-atom environment used for
embedding. Labels 1/2/3 (FCC/HCP/BCC) count as crystal; other labels do not. Neighbors
are drawn from the full periodic simulation. The maintained patch PTM implementation
analyzes all particles in separated clouds, then reads only centers. It must not
use `only_selected`, which also filters the neighbor search in the installed OVITO.
The local 80-atom assay matched full-cell periodic PTM for all 512 preflight centers.
This is a finite validation, not proof for every neighborhood.

Positions come from the retained float16 trajectories, with float32 box bounds and
exact atom IDs/timesteps. Labels and embeddings use those same artifacts. Agreement
with full-precision original positions was not measured in this assay.

The primary onset is the **start** of the first fully observed run of three crystal
frames. This entails 1.5 ps of confirmation after onset. Sensitivity rules require
five/nine frames (3/6 ps of confirmation). This defines a local crystalline episode,
not irreversible crystallization. A trajectory with no such run is right censored;
code represents its onset by the trajectory length, outside all valid frames.
Already-crystalline initial trajectories have onset zero. Windows reserve enough
end frames for all sensitivity confirmations, share origins 6–585 ps, and forecast
all 12 next samples through 9 ps.

## Physical readout and model separation

A class-balanced two-output ridge classifier maps observed local embeddings to PTM
state. It uses every eighth frame of training sources. Training-only means, sample
standard deviations (floored at 0.01 times their median), and inverse-class-frequency
weights define a regularized least-squares fit. The bias is unpenalized. Alpha is
selected from 0.1/10/1000 by validation macro F1; test data only measures the selected
probe. This reuses `smooth_temporal_encoder.evaluate.structure_fit` unchanged.
The readout score is crystal minus noncrystal output, **not a calibrated probability**.
Projection of the restored raw predicted embedding through this frozen linear
readout produces future crystal scores. Forecast normalization is undone exactly
algebraically; model inference takes observed embeddings only, never future labels
or embeddings. Forecasts run in float32; readout projection uses float64 and saved
margins use float32. Checkpoint identities, epochs and file hashes are retained.

Comparators are the last embedding, the arithmetic mean of the nine observed
embeddings, and a least-squares linear slope of observed embeddings extrapolated
from the last observation. Because the readout is affine, they can be computed
from observed readout margins without constructing 256-dimensional future arrays.
`observed_future` reads the true future embeddings: it diagnoses readout limitations
and is **not an available forecasting method**.

## State and transition classification

Future-state metrics use actual PTM state at exactly +3/+6/+9 ps at all matched
origins, including already-crystalline regions. Each method/horizon selects a score
threshold by maximum validation positive-class F1. An exact F1 tie chooses the
highest threshold. The default zero-margin state accuracy is also retained.

Transition metrics use only origins before that atom's first sustained onset, with
all three latest observed physical frames noncrystalline. This is a conditional
assessment of predictions from noncrystalline environments, not an end-to-end
inferred-current-state detector. An event is positive when true onset falls in
`(origin, origin + horizon]`. Never-crystallizing trajectories remain negative.
For each method/horizon/persistence, the maximum predicted crystal margin over the
future prefix is thresholded. The threshold maximizes validation event F1, with the
same tie rule. Test sources never select a probe, model, threshold or persistence.
The predicted onset is the first future frame crossing that threshold. The physical
persistence rule defines retrospective truth; the alert itself requires one crossing.
The interval classifier therefore does not enforce persistence in the forecast.

TP/FP/FN/TN are window counts. Accuracy = (TP+TN)/N; precision = TP/(TP+FP);
recall = TP/(TP+FN); specificity = TN/(TN+FP); false-positive rate = FP/(FP+TN);
balanced accuracy = (recall+specificity)/2; F1 = 2TP/(2TP+FP+FN).
AUROC uses continuous margins (the maximum margin for events). Average precision
is sklearn's non-interpolated positive-class average precision. Positive prevalence
is the random-ranking AP baseline; always-negative accuracy is one minus prevalence.
Zero denominators and undefined one-class AUROC are recorded as null/blank, not zero.

## Timing and lead-time analysis

Timing error = predicted onset minus actual onset in ps: negative means early.
MAE, bias, median/p90 absolute error, and fractions within 0.75/1.5/3 ps condition on
true positives. Missed events remain in FN and recall. `timed_within_1_5_ps_recall`
divides correctly detected-and-timed events by **all** true events, penalizing misses.

The fixed-lead analysis uses one origin exactly 3/6/9 ps before each eligible local
onset, using the same validation threshold selected above. It excludes events without
sufficient observed history or with a crystalline frame in the recent three-frame
history. All examples are true events here: recall and timing are useful; precision
and overall accuracy are degenerate and not interpreted as population performance.
Local examples are selected by the AR 9 ps outcome: lowest timing error, median
detected-event timing error, and first missed event when present. They are illustrative,
not a random sample. Shading shows instantaneous actual PTM state.

Intervals are percentile 95% intervals from 2,000 source bootstrap replicates with
fixed validation thresholds and model/probe fits. Empty-event sources are retained.
Replicates with undefined denominators are excluded only for that metric; valid
replicate counts are explicit. Timing intervals use per-source sums and TP counts.
These intervals do not include training-seed or threshold-fit uncertainty. Paired
F1 differences against history mean use the same resampled test sources for both
methods. Temperature tables preserve actual source counts; there are no validation
sources at 520 K, so that temperature uses thresholds selected at other temperatures.

The event census also counts onsets with nine observable frames starting at onset
and how many remain continuously crystalline throughout those nine frames. The
median initial episode length counts consecutive crystal frames until the first
noncrystalline frame or the trajectory end; end-truncated episode lengths are lower
bounds. This diagnoses transient versus long-lasting local crystalline episodes.


Table export: 2026-09-13T15:11:51.269136+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
