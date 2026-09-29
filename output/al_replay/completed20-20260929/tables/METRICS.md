# Paired Al replay diagnostics

Scope: the 20 completed original/rerun pairs at 520 K. Identical native prepared
liquid and melt-restart hashes, velocity seed and atom IDs are verified. The
rerun changes integration from 3 to 2 fs and observation cadence from 0.75 to
0.1 ps. Measurement zero follows 15 ps equilibration in each run; these are
already different evolved states. No models are fitted or selected, including
when the audit includes validation/test ancestry. Completion-limited coverage
is not an unbiased sample of all 150 lineages or of other temperatures.

Common observations are identified using integer femtoseconds: 401 exact times
0, 1.5, ..., 600 ps. No interpolation. Both inputs are existing float16 binary
positions and velocities; thermodynamics come from original full-precision
LAMMPS logs, matched at the same observation times. Large arrays are checked
against their declared shapes/types; this audit does not rehash every array.

## Atomic and thermodynamic tables

`same_atom_rms_A`: RMS over all 70,304 corresponding atom IDs of periodic
fractional-coordinate differences, multiplied by the mean of the two box side
lengths. Fractional differences use minimum images in [-0.5,0.5]. This removes
affine box dilation but not rotations or rigid translation. It is a cross-run
configuration distance, not a diffusion MSD or an unwrapped displacement.
`same_atom_median_A` and `same_atom_within_1A` summarize the same distances.
`random_position_rms_A` is sqrt(sum(mean_box_lengths**2)/12), the independent
uniform-position reference, not a fitted baseline.

`velocity_correlation`: summed Cartesian dot product of corresponding velocities,
divided by their vector norms after subtracting each frame's mean velocity.
`temperature`, `pressure`, `volume`, `energy` are the logged Temp, Press, Volume,
PotEng fields; volume and energy are divided by 70,304. Per-source mean deltas
are new minus old, with equal weights on the exact common 401-time grid.
Temperature is a physical audit observable, never a predictor input.

## Structural table

Recompute both sides with OVITO PTM, RMSD cutoff 0.1, using all atoms and periodic
boxes. This avoids comparing old raw-text PTM labels with new quantized-input
labels. Crystal means FCC + HCP + BCC (codes 1,2,3). Largest crystal cluster uses
OVITO selected-particle connectivity with 3.5 Å cutoff. Fractions for Other,
FCC, HCP, BCC and ICO divide counts by all atoms. Structural times are every
15 ps including endpoints, plus 1.5, 3, 6 and 12 ps, declared in the frozen config.

`neighbor_retention`: mean fraction of the original 12 nearest neighbors also
present among the rerun's 12 nearest neighbors, for 2,048 uniformly sampled
centers without replacement (seed + source ID). Self is excluded; periodic
Euclidean distances and atom identity are used. Quantization can change ties.
`ptm_agreement`: corresponding atom labels equal. `ptm_kappa` removes the agreement
expected from the two label-frequency marginals: (agreement-chance)/(1-chance).
Kappa is undefined if chance=1. `crystal_atom_jaccard` is the crystalline atom-ID
intersection divided by union; undefined if both sets are empty.

`t10` and `t50`: first sampled time at which crystal fraction is at least 0.1 or
0.5 and remains above threshold at the next structural sample. Missing means
no confirmed crossing by the last sample, not time zero. These are coarse bulk
transformation landmarks, not critical-nucleus or committor estimates. Their
resolution is normally 15 ps, and an endpoint-only crossing is unconfirmed.

## Paired aggregation and controls

Each source gets equal weight; atoms/time frames are not independent replicates.
`mean_absolute_crystal_fraction_difference` and crystal-curve MAE average the
absolute difference over the declared structural sample grid (equal sample
weights, not a continuous-time integral). Final fractions use exactly 600 ps.
Reported source-bootstrap intervals resample 20 paired sources with replacement
10,000 times (declared seed), taking 2.5/97.5 percentiles of the source mean.
These intervals are descriptive, conditional on completed 520 K coverage.

For the parent-association control, form all original/new crystal-curve MAEs,
then compare the observed diagonal mean against 10,000 random permutations of
new-run parent identities. Permutations can contain fixed points. The one-sided
lower-tail p is (1 + count(permuted <= observed))/(10000+1). This asks whether
same-parent bulk curves are more similar than arbitrary pairs at the same
nominal temperature; it neither establishes path identity nor proves a timestep
bias or equivalence. There is one old and one new run per parent, with no matched
same-timestep replicas to isolate integration effects from trajectory divergence.


Table export: 2026-09-29T12:19:27.200925+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/al_replay.json` relative to the analysis root.
