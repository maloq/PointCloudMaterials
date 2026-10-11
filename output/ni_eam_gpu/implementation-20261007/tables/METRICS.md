# Ni EAM GPU numerical validation and timings

These are local engineering diagnostics, not predictive metrics or a Ni production dataset.

Energy error is absolute total potential-energy difference from LAMMPS divided
by atom count, eV/atom. Force RMS is sqrt(mean((F_GPU-F_LAMMPS)^2)) over all
Cartesian components, eV/Angstrom. Virial error is the absolute configurational
virial-trace difference divided by atom count, eV/atom; native reference pressure
has zero velocities and is converted using the actual LAMMPS `nktv2p` value (1602176.5 for this executable).

HVP relative error is the Frobenius norm of the difference divided by the
reference norm, across all atoms/components/directions. A float64 central
finite difference with epsilon 1e-5 Angstrom validates the analytic derivative;
it is not the production derivative algorithm. Separate checks compare float32
force arithmetic against float64 analytic HVPs and propagate a 20-step NVE tangent.

NVE excursion is max(abs(E(t)-E(0)))/N across every ten steps plus the final
observation. States and velocities remain float64; forces use the specified
dtype. Energy reduction is float64; this does not make per-atom arithmetic float64.

Timings exclude build, graph capture and initial warm-up. Force API timings
include coordinate copies, skin-displacement checks, result copies and error
synchronization; static input does not trigger neighbor rebuilds. Kernel graph
replay times cover force/energy/virial kernels and reductions with fixed neighbors,
not MD. Verlet step time includes moving states, rebuilds and Python integration.
Captured Verlet time includes 100-step CUDA graph blocks, float64 state updates,
GPU displacement checks and conditional neighbor rebuilding every step, plus
host error checks between blocks. It excludes graph capture and a 100-step warmup.
The 500-step measured trajectory begins after warmup, while the Python-loop
measurement begins from the initial state; their rebuild counts and states are
therefore not identical. The captured input is also shifted uniformly by 0.05
Angstrom to exercise safe reuse of an earlier list-build origin.
ns/day = timestep_fs * 1e-6 * 86400 / seconds_per_step, an extrapolation without
trajectory I/O, thermostat/barostat or structure analysis. All timings use medians
of five repeated batches except the explicitly recorded complete Verlet run.
FastAl comparisons use different physical potentials on similar-size distorted
FCC cells and compare implementation cost only, not interchangeable dynamics.
Ni uses lattice parameter 3.60 Angstrom; Al uses 4.05 Angstrom; both add Gaussian
0.10 Angstrom displacements and use float64 states with float32 force arithmetic.
Two random, RMS-normalized direction columns exercise analytic force tangents.
FastAl force medians use five batches of 20 calls; tangents use five batches of
10. Ni force/tangent medians use five batches of 100/50 calls, respectively.
The kernel-only timing uses five batches of 500 graph replays. NVE velocities
have a Maxwell scale of 1370 K with removed center-of-mass velocity. Timings on
distorted FCC cells do not establish equal performance in a dense liquid.


Table export: 2026-10-07T11:28:40.665674+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/ni_eam_gpu.json` relative to the analysis root.
