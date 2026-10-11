# Ni relaxation speedup audit, October 11

The attached literature summary motivates missing optimizer comparisons. The
current implementation already uses modern native LAMMPS FIRE, CUDA graphs,
one synchronized callback transfer, persistent workers and independent cells.
The separate sorted-neighbor backend fixes identical-input repeatability on the
previous five-source assay. Existing production recipes keep their provenance.

New isolated benchmarks compare ABC-FIRE, two tuned FIRE profiles, stopping
tolerances, unpreconditioned L-BFGS, Exp-preconditioned L-BFGS and PreconFIRE.
The last three run as bounded three-case pilots because preconditioning cost can
dominate this inexpensive GPU potential. Twelve full-cell inputs are reused from
the existing pinned Ni response bank. No original MD or current training job is
modified. No new training run or W&B entry is created for these diagnostics.

Recipe: `configs/simulation/ni_relaxation_speed_20261011.json`.
Implementation: `src.research.ni_relaxation_speed`.
Results: `output/ni_relaxation/speed-20261011-v1`.
Run `python -m src.research.ni_relaxation_speed.experiment submit --config
configs/simulation/ni_relaxation_speed_20261011.json` once. The launch receipt
owns the separate three-hour, one-GPU alternative-partition job. Per-case
timeouts/failures do not block subsequent independent cases. Frozen source,
config, inputs, dependency installation, progress and logs are under technical.
Final tables and plots are produced automatically after the queue completes.

[Metric definitions](../metrics/ni_relaxation_speed.md).
[LAMMPS optimizer options](https://docs.lammps.org/min_modify.html).
[ASE preconditioning](https://ase.gitlab.io/ase/ase/optimize.html#preconditioned-optimizers).
