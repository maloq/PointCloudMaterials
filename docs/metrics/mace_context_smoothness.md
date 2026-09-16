# MACE context smoothness follow-up

This is an analysis of the seven completed feature extractions from the
[context pilot](mace_context.md), with no additional encoder or readout fitting.
It independently reproduces the original temporal and boundary scores and writes
a separate run suffixed `-smoothness`, preserving the historical exports.

The exact producer shapes are 5,760 anchor embeddings by 256 channels,
144 temporal tracks by 17 frames by 256 channels, and four crossing parameters
by 72 patches by two sides by 256 channels. All differences and reductions are
computed in float64 from retained float32 features. Both trajectory producers
request 0.75 ps cadence. Lags are the configured 1/2/4/8/12/16 frame offsets.
The six temporal sources must match the held-out probe split, with 24 tracks per
source. These sources were already examined in the exploratory pilot.

`raw_increment_mse` averages squared feature differences over channels, available
time pairs, tracks, and sources. The six sources have equal track counts, so this
equals the original all-pair mean. `source-increments.csv` retains the mean for
each source and lag. Overlapping time pairs are not independent observations.
Raw embedding units differ between methods and should not be compared as if
they shared a calibrated scale.

`train_total_variance` is mean channel variance over the 3,456 training anchors,
using population variance (ddof=0). `train_within_frame_variance` first subtracts
each training context's mean of its 64 anchor embeddings, then averages squared
residuals over all anchors and channels. No held-out observations set either
scale. `relative_increment_mse_total` and `relative_increment_mse_within_frame`
divide the same raw temporal increment energy by their respective scales. The
latter removes between-frame differences from the normalization, not from the
temporal trajectories. Both are invariant to uniform embedding rescaling, but
neither is invariant to arbitrary channel rescaling. Lower values indicate
less temporal change relative to the chosen representation spread; they do not
isolate numerical noise, a Lipschitz constant, or forecasting performance.

`paired-comparisons.csv` reports candidate/reference ratios of normalized mean
squared increments. `architecture` compares each alternative with the original
mean in the same frozen/trained state. `continuation` compares each trained
representation with its own frozen version. `rms_increment_ratio` is the square
root of the squared-increment ratio; percentage changes in these two quantities
are different. Percent reduction in squared change is 100*(1-ratio).

Percentile 95% ratio intervals use the configured 4,000 paired resamples of the
six whole test sources with replacement, retaining all tracks, windows and lags
from each source. The same draws are used for all comparisons. Training scales
and fitted encoders stay fixed. These are pointwise exploratory intervals, not
simultaneous bands or estimates of training-seed or normalization uncertainty.

`boundary.csv` retains all four original controlled crossing parameters
(0.1/0.01/0.001/0.0001 Angstrom). `raw_crossing_mse` averages squared differences
between the two input configurations over patches and channels;
`relative_crossing_mse` divides by total training variance, and
`fraction_of_075ps_energy` divides by that representation's natural 0.75 ps
increment energy. Quadratic decrease with the perturbation distinguishes a
continuous response from the hard-80 finite-jump plateau. The smallest values
can approach numerical precision. These perturbations test a specific membership
boundary, not every possible point-cloud motion.

The PNG/PDF shows the three matched trained representations using both temporal
normalizations and the controlled crossing curve. Frozen representations and
paired uncertainty remain available in the tables. Source input checksums and
the reproduced-metrics flag are in `technical/provenance.json`; the exported
definitions and implementation hashes accompany every table set.
The optional combined center/inner graph output used by the later recovery
experiment does not alter these retained single-readout feature arrays or scores.


Implementation note (2026-09-15): the shared context graph now retains original
input indices for aligned velocity fields and optionally exposes pre-pooling atom
features. Default structural pooling and all metrics above are unchanged; the
velocity extension is a separate `mace_local_phase_space_v1` protocol. Historical
exported definitions and implementation fingerprints remain untouched.
