# Relaxed-input/target pilot and relaxation fidelity

Frozen readout predictions use the original MD first sustained local crystalline
onset and horizons 0.75, 3, 6, 9 and 12 ps. They use exactly the hazard likelihood,
source weighting, calibration thresholds, classification and timing definitions in
`crystallization_information.md`; raw readout CSVs include that definition separately.
Positive `nll_gain` is instantaneous→instantaneous test NLL minus the named arm's
NLL, averaged over test sources. `ci95` resamples whole sources within temperature
1,000 times. This interval excludes training-seed uncertainty.

The quench benchmark pairs full periodic cells, fixed box and generating Lee2003
MEAM potential on the same host and MPI rank count. Each tolerance starts from the
same original coordinates, not the other minimizer's endpoint. Reference infinity
force tolerance is 0.01 eV/Å; approximations use 0.03 and 0.1. `seconds` is LAMMPS
subprocess wall time including startup and output; excludes Python extraction,
TDA and encoder inference. `speedup` is reference seconds / candidate seconds.
`seconds_per_64_centers` and `seconds_per_256_centers` are amortized estimates
(seconds / number of centers), not separately measured minimizations.

`coordinate_rms_A` is sqrt(mean squared Euclidean coordinate difference), using
center-relative periodic coordinates and identical atom identities. Target fidelity
is mean squared error after dividing each component by its original training-only
standard deviation: Physical85, TDA144 and order8 respectively. Physical85 and TDA
are evaluated in physical Å after the same normalized radius8 crop as encoder inputs. Order targets use the same nearest-neighbor definitions
as `neighborhood_jepa_regularization.md`. Lower is better; these measures do not
certify the same energy basin. `energy_eV_per_atom` is minimized potential energy /
atom count. Force convergence at a loose tolerance is not reference convergence.

The encoder stage retains the existing v3 reconstruction/JEPA/regularization
calculations, exported with `neighborhood_jepa_regularization.md`. Present/future
reconstruction errors across different target domains must not rank encoders.
Use the common physical crystallization readouts for that comparison.


Table export: 2026-09-21T08:45:34.246526+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
