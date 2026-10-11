# Differentiable Lee2003 Al MEAM: numerical fidelity and response cost

This is a hardware/numerical benchmark, with no trained encoder, predictor or
phase labels. It reuses four fixed-Al64 training ancestors (indices0,1,2,45),
their unrelaxed nearest80 displacement bases and original independent noise
streams from the frozen force-engine comparison. No new independent ancestors
are created. Release e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d.

The specialized PyTorch energy reproduces the SHA256-pinned Lee2003 pure-Al
MEAM library/parameter files, including 2NN reference density and pair-series
correction, the original1000-point cubic pair interpolation, angular densities
through order3, ibar3, ialloy2, geometric three-body screening and4.5A cutoff.
All atoms in the open24A sphere move. Dynamic neighbor/screening support is
selected outside AD; the active smooth functions, forces and two force-direction
derivatives are differentiated in PyTorch. No fitted potential replaces MEAM.

Energy error is absolute total-energy difference divided by atom count (eV).
Force RMS is over all3N Cartesian components (eV/A). Force-tangent relative error
is the Frobenius norm of AD minus central finite differences of LAMMPS forces,
divided by the finite-difference norm, across both unit-norm local80 directions.
Displacement epsilon1e-4A is collective, not per-atom RMS. The force gates are
1e-8eV/atom,1e-8eV/A RMS, and0.002 relative tangent error, respectively.

The complete-path gate uses the existing CPU LAMMPS force adapter, float64 BAOAB,
450K,1fs steps,100fs friction and observations20fs/100fs. Saved original full-cell
noise is gathered by exact atom identities. A central trajectory plus positive
and negative perturbations along each direction estimates future-response
derivatives at epsilon0.003A. Each perturbed path uses its own initial anchor.
One original seed per ancestor checks all768 future features: legacy256,
rich-delta256 and rich-joint256. Rich train-only observable normalizers and actual
float32 Fourier coefficients come from the frozen response-geometry campaign;
double precision promotes the same coefficients, rather than redrawing maps.
Value RMS averages768 coordinates. Response relative error uses all768x2
coordinates. Gates: value RMS1e-4, response relative0.02. Float32 PyTorch versus
float64 PyTorch full paths must pass the same gates. Maximum errors in the final
report are maxima across the four predeclared ancestors, not confidence bounds.

Timing compares MEAM float64, MEAM float32 and current cuEquivariance MACE-MPA-0
float32, sequentially on the same GPU, same neighborhoods, noise streams,
integrator,768 targets and two directions. It includes online original-full-cell
noise generation, neighbor updates,101 force/tangent evaluations, integration,
and target values/JVPs. It excludes model setup, warmup, disk IO and host/device
input transfers. GPU synchronization brackets each wall-clock timing. A complete
untimed query warms each engine/shape. Three repeats rotate engine order. Saved
outputs are from repeat0; repeated timings are not new scientific samples.

batch1 is the current local24A collection setting; batch2 is an explicit
throughput experiment exceeding the old4096 batch-atom budget. seconds_per_shot
is synchronized batch wall time divided by batch size. peak_allocated_MiB is
PyTorch's total peak allocated memory during the timed query, including resident
comparison objects; it is not an isolated model-memory measurement.

Aggregate each engine/precision/batch using mean seconds_per_shot with equal
weight per ancestor, original seed and hardware repeat (the schedule is balanced).
Speedup is matched MACE mean divided by MEAM mean at the same batch size. Report
both means and ratio; potential physics differ even when hardware/protocol match.
This does not measure training speed, crystal clustering or nuclei prediction.
