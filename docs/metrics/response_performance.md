# Fixed-MACE response-oracle performance

This local numerical benchmark preserves the frozen MACE-MPA-0 teacher, Al256
periodic box,1fs BAOAB update,450K noise,20/100fs observables, initial-anchor
derivative, two directions and independent branch seeds of the completed response
training experiment. No encoders are fitted and no online runs are created.
The historical source hashes and archived query checksums are verified before use.

Variants isolate conditional derivative graphs, GPU minimum-image neighbor
construction, an optional0.6A Verlet skin, cuEquivariance, replica batches1/4/8/16,
and separately declared float32 arithmetic. The actual teacher cutoff is read
from its model (6A); cutoff+skin must be below half the shortest orthorhombic box.
Candidate skin edges are rebuilt at maximum displacement>=skin/2 and filtered to
the true cutoff each call. Geometry remains live; edge/image selection is detached.

Each branch retains its own original float64 CUDA RNG stream. Float32 variants
cast the original float64 noise increments, rather than changing the random draw
sequence. Features retain the same Fourier frequencies/phases, cast only for the
explicit float32 experiment. For finite differences, the central and four perturbed
trajectories share a seed; every perturbed path uses its own initial anchor. All
five trajectories are timed together, yielding one central value and two responses.

`throughput.csv` reports one full100fs measurement per variant/mode after a
force/HVP warm-up, excluding potential loading, backend conversion, numerical
gates, disk I/O and CUDA initialization. Seconds per trajectory is elapsed wall
time divided by independent seed replicas; trajectories per GPU-hour is3600 times
that count divided by elapsed time. For finite-difference responses, one replica
means the entire five-trajectory query; trajectory_executions records the five.
Thus response throughput always counts complete queries, not individual tangents.
Model calls count scalar-energy forward invocations, including disconnected graph
batches. Peak allocated GiB is torch.cuda.max_memory_allocated/2^30 during that
measurement, including resident model memory but excluding external allocations.
These short measurements are throughput estimates, not repeated timing intervals.

Relative error is the Frobenius norm of candidate minus archived float64 reference,
divided by reference norm floored at1e-12. Maximum absolute error is the largest
coordinate error. All returned values are checked; response comparisons use up to
eight archived independent response seeds (batch16 has16 value references and8
response references). Additional full-horizon gates span all four displacement
strata for the declared batch4 variants. Static checks compare energy, force and
two HVPs to the original eager float64 teacher, including moving one atom by a
whole box vector. This is equivalence to the original teacher, not merely a check
against the same accelerated backend.

Recipe tolerances define numerical acceptance separately for float64 and float32.
A variant is eligible only if its receipt state is complete and all its gates pass;
a throughput row marked passed is insufficient if a later gate fails. Failures,
including out-of-memory and unsupported higher derivatives, are retained with
tracebacks. No fallback or dtype substitution is performed. Float32 accuracy is
reported against an explicit budget, never described as exact float64 equivalence.
The benchmark changes no completed label bank, metric definition or frozen run.
