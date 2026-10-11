# Dense Al GPU versus CPU comparison

Read-only physical audit of the five completed GPU descendants, all completed
main dense CPU descendants at capture, and all 150 historical CPU ancestors.
Exclude the two extra 0.01 ps descendants from independent counts. Pair by the
original source ID, verify native preparation hashes and velocity seeds, and
retain independent melt lineage. Historical CPU uses 3 fs integration and
0.75 ps observations; current dense CPU/GPU uses 2 fs and 0.1 ps. GPU uses
float32 forces and float64 native NPT integration. These are not a controlled
long-duration comparison at identical integration settings.

Historical PTM and thermodynamic arrays are verified against their original
outcome hashes. Historical PTM derives from native snapshots; the half-stop
monitor uses a float32 consumer of native snapshots. Cached earlier dense PTM
and newly evaluated missing dense curves use verified float16 observations.
Crystal means FCC/HCP/BCC, PTM RMSD cutoff 0.1; largest cluster uses 3.5 angstrom
connectivity. These small coordinate-precision differences remain explicit.
This audit does not identify or track nucleus births or estimate nucleation rates.

Primary curves use exact 15 ps observations with no interpolation. Append an
actual final off-grid observation when present, without using it as a 15 ps
confirmation. `t50_ps` is the first of two consecutive observations exactly
15 ps apart with crystal fraction at least 0.5. Missing means no confirmed
half crossing by the observed endpoint. It is not time zero or permanently
liquid. Conditional medians/minima/maxima include confirmed events only;
report unresolved counts separately. Historical follow-up is 600 ps; stopped
dense runs have their own endpoints. Do not compare terminal fractions as
equal-time or equilibrium measurements.

Thermodynamic values are native logs/arrays: temperature K, pressure bar,
potential energy eV/atom and volume angstrom cubed/atom. Historical GPa is
multiplied by 10,000. Atom count is 70,304. Plot energy/volume at exactly the
observed structural time, never a trailing average or interpolated state.
Repeated run-boundary steps use the last logged force reevaluation; their
maximum four-column discrepancy is saved in native log units (K, bar, total
angstrom cubed, total eV). Resume suffixes replace duplicated boundary rows.
Missing thermodynamic prefixes are retained as missing, not reconstructed.

`early_mean_*` averages each trajectory's actual native samples in (0,15] ps:
20 historical samples or 150 dense samples. Only a complete early window
receives a mean; sample count and completeness remain explicit. Different
cadences, temporal correlation and finite-window pressure fluctuations mean
these differences are descriptive, not significance tests. `early_max_crystal_fraction`
uses structural observations at 0 and 15 ps and audits early phase content;
it does not establish that no small nucleus exists. Paired differences are
GPU minus the historical same-parent CPU mean, energy multiplied by 1,000
to meV/atom, and volume 100*(GPU/CPU-1). Full logged mean temperature/pressure
also report the available time span and are not matched-duration quantities.

Each cohort row counts unique source IDs within one engine protocol.
CPU parents and descendants share ancestry and must not be summed as
independent samples. One GPU trajectory per temperature cannot establish
equal crystallization-time distributions or thermodynamic equivalence.
This workflow changes no simulations, queues, training inputs or labels.


Table export: 2026-10-04T16:26:15.289206+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/meam_dense_comparison.json` relative to the analysis root.
