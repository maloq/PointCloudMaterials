# Spatial approach on fixed snapshots (v1)

Producers: `src/research/spatial_approach/`. Distance is in Å; no conversion to
ps or assumed probe velocity. Keep all 126,545 original Al64-v1 sample IDs and
90/15/15/30 source roles. This is the historical currently-liquid, pre-first-onset
population, not an unbiased spatial-volume sample. Only the outcome changes.

## Target and fitting

Reference crystal atoms belong to periodic components with at least 64 atoms
and an audited ancestor confirmed by that snapshot. PTM, connectivity and
confirmation follow the frozen Al64 ancestry audit. Future backfilled roots
cannot establish a crystal before its confirmation. Distance is the minimum
periodic distance from the probe atom to a reference crystal atom, **not** a
fitted thermodynamic interface or distance to a grain centroid. No crystal gives
infinity. The six bins are d≤4, 4<d≤8, 8<d≤12, 12<d≤20, 20<d≤32 and d>32 Å.

New spatial readouts use categorical NLL, with the existing five-logit hazard
factorization parameterizing six spatial bins. All epochs visit every training
row with inverse source-size weights. Selection minimizes source-weighted NLL
after at least twelve full epochs; no AP or warning-distance selection.
The common frozen observed MACE was fitted with temporal-onset NLL on training
sources. This is supervised transfer, not new encoder training or SSL evidence.

NLL, Brier and AP for distance within each threshold use equal-source weights.
AP is diagnostic. The training-prior reference uses train-only bin proportions,
floored at 1e-12 for its NLL. Calibration tables and distance profiles are
unweighted row summaries. Infinity belongs in the final profile bin; empty
bins have blank statistics. No model-specific sample dropping is allowed.

## Actual predictor inputs

- Geometry MLP: 32 local radial/count/bond-power descriptors from the same
  observed 80-candidate, <8 Å patch; a readout, not reconstruction pretraining.
- Local MACE: only the focal patch's exported 128-D z.
- Symmetric invariant: shared z128 from 25 patches plus relative geometry.
- Vector messages: same plus l=1 fields.
- Harmonic hierarchy: same plus l=1/2 and geometric l=4/6 bond fields.

The encoder has 128 channels, two interaction blocks, a 5 Å edge cutoff and one
constant atom channel. Its original 80-nearest-candidate, <8 Å patches are
truncated observations with no external message-passing halo. Context queries
at 0/10/20 Å have actual representatives displaced at most 4 Å; total input
support is bounded by 32 Å. Normalization uses train rows only; equivariant
fields receive channel RMS scaling without componentwise centering.

Temperature, age/time, material IDs, path direction/index, true distance, PTM
and ancestry never enter predictor tensors. No velocities, temporal history,
relaxation or teacher is used. Scanning order enters only the causal alarm rule.
`visible_local/context` records whether any actual input atom is in a reference
crystal; `ptm_local/context` also counts other PTM FCC/HCP/BCC atoms. These are
label-side diagnostics. Detection with crystal visible is contextual recognition;
detection without it is a candidate liquid-structure signal, not causal proof.

## Scan sampling and alarms

Calibration/test sources use all available fixed observation snapshots per
source and up to two randomly chosen fixed centers with distance 36–60 Å.
Toward paths follow the minimum-image ray to the nearest crystal atom, at 2 Å
nominal waypoint spacing. Snap to nearest atoms, remove consecutive duplicates
and include the endpoint. Away controls follow the opposite ray for the same
nominal length; retain only paths staying beyond 32 Å. Record all exclusions.
These are conditional known-target diagnostics, not autonomous navigation,
random-path incidence or motion through evolving MD. Models see no target ray.

Alarm score is P(d≤8 Å). Two consecutive observations must strictly exceed the
threshold; the alarm occurs at the second, with no forward peek. Set the
threshold to the higher empirical 95th percentile of maximum two-position
scores on calibration away paths. This targets ≤5% empirical path-level false
alarms there, not a guaranteed population rate. Report control counts and the
held-out false-alarm rate.

Warning distance is reference distance at first alarm; blank means no alarm.
Recall at D divides paths first alarming at distance ≥D by **all toward paths**,
including misses. Median warning distance is conditional on detection and must
accompany recall/miss results. An alarm without visible crystal requires both
contributing inputs to be clear. The early-alert fraction also requires d>8 Å.

Report pooled-path recall and equal-source mean recall separately. Bootstrap
per-source means (500 draws) for NLL and source-mean recall; CIs do not describe
the pooled-path statistic. Fewer than two sources gives an undefined interval.
One seed does not quantify training-seed uncertainty.
