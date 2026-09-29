# Material-specific CD-MACE128-D6 fine-tuning

Both fits initialize the entire completed CD-MACE128-D6-075nominal model,
including its temporal fusion and distance head. MACE and head remain trainable.
Fresh AdamW state, encoder LR 1e-5, head LR 5e-5, twelve epochs, seed 20260927.
Selection uses the inherited predictive objective at epoch twelve, never AP.
Global batch 1024, microbatch 256 and four accumulation steps on one GPU explicitly
retain the parent batch instead of the general batch-256 default.

The proper objective is censored zero-inflated lognormal distance NLL plus twice
the Bernoulli log losses for distances within 8/12/20/32 Al-equivalent Angstrom,
weighted .05/.15/.40/.40. Censoring/point-error cap is 64 Al-equivalent Angstrom.
No new weighting, AP objective, physical reconstruction or representation loss.

## Inputs and population

The spatial encoder consumes nearest-80 centered coordinates with smooth radius
8 and edge cutoff 5 in normalized units, two message-passing blocks, a constant
atom channel, width/export 128. There is no additional halo or surrounding-patch
predictor. Six per-frame exports and adjacent differences feed the temporal MLP
and current-state residual. No temperature, explicit time/cadence, velocity,
species/material ID, scale, relaxation or training-only teacher is an input.
Material lengths only normalize coordinates and target distance in preprocessing.

Al fitting uses only the original 90 native train sources: 4,561,920 windows.
The 15 native selection sources retain 190,080 windows. Every observation interval
is exactly 0.75 ps, span 3.75 ps. The Al64-v1 identity and all64 evaluation rows
are unchanged, including calibration/test exclusions from fitting. Its fixed test
contains 45,291 rows across 30 sources; scan tests have 495 approaches and 292
far-controls from 28 sources. The unchanged parent's existing predictions are
reused only after checkpoint, geometry-manifest and exact row/target checks.

Ta fitting uses all six earlier external Ta trajectories: 3,061,560 windows.
They were already observed by the parent. Every interval is exactly 0.70 ps,
span 3.5 ps, without interpolation. The nominal parent label remains 0.75 ps;
actual source cadence is never rewritten as exact 0.75 ps.

Ta evaluation uses the four completed million-atom parent00 shooting branches.
Shot00 is selection, shots01/02/03 are test, declared before scoring. There are
10,240 fixed, outcome-blind centers per branch and anchors at 9/12/15/18/21/24 ps:
61,440 selection and 184,320 test windows. Entire branches are withheld from
fine-tuning gradients. The parent saw another trajectory from the same initial
configuration. All branches therefore share ONE preparation ancestry; this is
conditional generalization to new velocities, not independent-preparation or
unseen-material generalization. No confidence interval over independent sources
is claimed. Both parent and adapted model use exactly these evaluation rows.

The Ta reference reuses the existing PTM and component-lineage definitions over
the full periodic cell: RMSD cutoff .1, crystalline PTM classes 1/2/3, component
size >=64 with 1.5-ps persistence measured every .5 ps. Only components confirmed
by the current anchor enter the distance target. Future confirmation never enters
inputs or retroactively changes a current label. Recorded normalized Ta distances
are Al-equivalent Angstrom, not physical Ta Angstrom.

## Tables

`validation.csv` contains the inherited proper objective, distance NLL, early
CDF log loss, capped-mean RMSE and Brier scores. Al front tables retain the
[dense-history definitions](distance_encoder_dense_history.md), including
strict probability thresholds .5/.75/.95 and two-consecutive-position alarms.
The comparison copies the parent's frozen numbers rather than recomputing them.

Ta `distance.csv` contains parent/adapted rows per role, equal-trajectory NLL,
capped-mean RMSE, capped-median MAE, censored fraction and Brier at 4/8/12/20/32.
Equal-trajectory weighting does not imply independent ancestry. Its
`confidence-reliability.csv` reports the weighted empirical precision, mean
probability and coverage for P(distance <= radius) > .5/.75/.95. Empty selections
are undefined, not zero. These are point-distance diagnostics; no Ta spatial scan
or warning-distance result is claimed. Raw predictions retain atom IDs, frames,
source lookup and checksums for paired comparisons.

There is one seed per material. Metrics do not establish statistical significance.
Training is online in W&B. Evaluations create no runs; final scores update the
associated material training run through its stable existing ID.

The explicit interim-checkpoint evaluation command uses a separate output and metric family after a user-requested stop; final twelve-epoch exports remain unchanged.
