# GPU MEAM relaxation benchmark

Matched original full periodic Al cells, generating Lee2003 MEAM, fixed box,
FIRE timestep 0.001 ps, infinity norm tolerance 0.01 eV/Angstrom, 10,000 iterations
and 50,000 force evaluations; 1,800-second per-attempt timeout. No convergence
retries in this benchmark. Each repeat starts at the original MD coordinates.
Two repeats per case/backend. CPU-current and both GPU builds share pinned source
release d71abe6102c44577442ba7f03b7378a83166b9fd. CPU-legacy is the production binary.

`median_seconds` is median successful LAMMPS subprocess wall time including device
initialization and final dump I/O, excluding input writing, compilation, Python
startup, target computation, untimed force probe and archive publication.
`cells_per_hour = 3600 / median_seconds` is extrapolated serial successful-cell
throughput, not aggregate cluster throughput. Speedup divides the matched CPU
median by the backend median. CPU uses 32 MPI ranks; each GPU uses one process
and one device, double precision. Report completion/failure counts alongside
speedups. All-success throughput across the case suite cannot be claimed while
any case fails or remains unfinished. Two repeats do not estimate broad uncertainty.

CPU-current repeat zero is the fidelity reference. Initial forces are compared
at precisely the same original coordinates/atom IDs, via a separate run-zero
probe. Force RMS averages squared Cartesian component differences; relative RMS
divides by reference component RMS (floor 1e-30). Maximum error is absolute maximum
component difference. Energies and differences are eV per atom.

Final local clouds preserve 64 observed centers and the same nearest-80 observed
atom IDs. Coordinate RMS is sqrt(mean over neighborhoods/atoms of squared Euclidean
relative-offset differences), in Angstrom. Physical, TDA and bond-order standardized
MSE use the existing pilot descriptor producers and fixed training-only standard
deviations. Target cropping is matched, as in relaxed_encoder.md. Normalization is
fixed for comparison, not refitted per hardware. Final energy/target differences
can reflect different minimization basins, not merely force implementation error.
CPU-current repeat one provides a within-backend repeat comparison.

Case names are selected from existing training-source local onset records; they
do not certify the thermodynamic phase of the complete 70,304-atom cell. The
benchmark does not modify production dataset receipts or switch queue backends.
