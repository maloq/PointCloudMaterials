# Completed spatial VICReg study: saved-output summary

This is a post-completion descriptive review, not a changed training protocol,
new checkpoint selection, or a new encoder/probe/cluster fit. Its source inputs
are the immutable completed checkpoint metrics, cluster assignments and per-source
readout errors of the `spatial_vicreg_bias` family. Their hashes are retained in
`technical/input-hashes.json`; original numerical definitions remain beside them.

The primary displayed population is the all-phase uniform held-out track:
24,960 observations from 30 fixed test sources, including 12,266 observations
with no PTM-detected crystal among their 80 consumed atoms. Other all64/legacy16
tracks remain available in each original bundle; they are not pooled here.

Readout skill is the saved error reduction relative to the fitting-source mean
predictor on the same population, not standard R² against a fitted test mean.
The TDA family uses its original active-feature mask and training normalization.
The figure displays arithmetic means and min–max ranges over seeds 17/29/43,
not confidence bands. Pair distances retain the original training-variance
normalizer. Results include both encoder/projector outputs and K=3/6/7/10.

Cluster occupancy is counted from retained assignments with exact matching
source/frame/atom identity. Largest-cluster fraction and occupied-cluster count
are reported for all uniform test rows, strict-clear A inputs, and noncrystalline
centers whose A inputs include crystal. No labels or cluster centers are refit.

Paired readout contrasts use the saved float32 per-source error arrays. Features
are averaged within each family (excluding density identity targets); seeds are
then averaged with identical source ordering. Skill differences divide the mean
error difference by the common fitting-mean baseline error. Confidence intervals
resample all 30 source IDs with replacement 10,000 times, with the same draw for
both members and the denominator, percentile 2.5/97.5%, RNG seed 20260930. These
intervals condition on the three trained seeds; they do not estimate variability
over all possible training seeds. Epoch4→24 is a descriptive trajectory contrast;
the predeclared primary endpoints remain 12 and 24. No multiplicity-adjusted
significance claim, AP optimization or future-prediction claim is made.

The phase-field/q6 controls retain their actual privileged inputs and descriptor
overlap. Similar scores establish that broad cluster–descriptor associations
are not unique to a learned intermediate state. A positive finite-readout gain
does not establish conditional mutual information, a metastable liquid state or
precursor prediction. Global K-means occupancy does not determine how much
continuous information the original embedding retains.


Table export: 2026-09-29T10:25:20.422553+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/spatial_vicreg_results.json` relative to the analysis root.
