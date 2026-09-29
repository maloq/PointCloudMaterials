# Full liquid-information relaxation production

Requested 2026-09-28. The original crystal-free cohort requires 3,106 source/frame
cells: 976 have existing accepted quenches and 2,130 need minimization. This is
coverage of the existing 325,970 observations, not a new independent MD ensemble.
Original train/selection/calibration/test melt ancestry is preserved.

Recipe: `configs/simulation/liquid_full_relaxation_20260928.json`.
Producer: `src.research.liquid_predictability.control_relaxation` using the
established `src.simulation.relaxation.relax_frame` and trajectory converter.
Full periodic fixed box, Lee2003 Al MEAM, FIRE, 0.001 ps minimizer step, infinity
force tolerance 0.01 eV/Å; execution limits 50,000 iterations/250,000 evaluations,
7,200 seconds per cell. No MD integration or temperature/time model input.

Sixteen detached CPU workers each use sixteen MPI ranks, grouped into four Slurm
jobs of 64 CPUs/64 GiB each to stay within the per-user submission limit. Each
worker inherits its own disjoint 16-CPU affinity mask. Cells are deterministically
mixed across sources; receipts are resumable per cell. Failures archive all stopped
outputs on STORE and prevent full-study submission; they are never silently
removed from the full-coverage cohort. Other cells continue after a cell failure.
Workers have 48-hour Slurm limits; interrupted workers resume their frozen command.

Fresh work is in SCRATCH, completion receipts in IDS, verified completed archives
and failures in STORE. The converter verifies float16 positions, float32 boxes,
exact identities/timestep, checksums and measured quantization before deleting
text coordinates. No pre-quantization local clouds are claimed. Downstream matched
observations are reconstructed from archived coordinates just like the older cells.

Once every missing cell completes, a CPU dependency submits the full paired study:
raw/relaxed inputs × original/recomputed relaxed labels, matched raw/relaxed rich
feature prediction, and final comparisons. The existing-archive study runs in
parallel and has its own frozen cohort and output. Label-based common-population
exclusions remain explicit; relaxation availability itself must be complete.

Launch receipts and per-cell failures live under
`${storage:analysis}/liquid_predictability/full-relaxation-20260928/technical`.
