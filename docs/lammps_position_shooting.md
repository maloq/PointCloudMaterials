# Position-conditioned 2NN-MEAM shooting data

This campaign produces dense, independent future ensembles around observed
homogeneous-crystallization events.  It is intended for temporal representation
learning and future-neighbor evaluation, where one needs multiple possible futures
from comparable present structures rather than one long, highly correlated path.

The archived source trajectories contain positions and cells, but not velocities or
the internal Nose-Hoover NPT state.  The generated shots are therefore **not exact
restarts**.  Each shot starts from an archived position/cell, draws independent
Maxwell-Boltzmann momenta, and runs the same Lee-Shim-Baskes Al 2NN-MEAM Hamiltonian
under fixed-cell Langevin NVT.  There is no post-branch equilibration: time zero is
the actual parent state.

The production configuration selects 20 independent 400/450/500 K sources, at 12 ps
and 3 ps before their detected nucleation time, and launches eight 48 ps futures per
parent.  Coordinates and velocities are saved every 0.3 ps.  Descendants inherit the
train/validation assignment of their source velocity seed, preventing sibling or
same-source leakage.

Prepare the immutable manifest and inputs with:

```bash
conda activate pointnet
python scripts/run_lammps_meam_shooting_campaign.py prepare \
  --config configs/simulation/atomistic/al/meam_position_shooting_70304.yaml
```

The preparation step writes `manifest.json`, checksum-bound parent metadata, all
LAMMPS inputs, and `slurm/run_branch.sbatch`.  Submit one smoke branch first:

```bash
ROOT=output/synthetic_data/al_meam_position_shooting_70304_400-500K_48ps_8shots_20260830
sbatch --array=0 "$ROOT/slurm/run_branch.sbatch"
```

After it succeeds, submit the rest using the concurrency cap in
`slurm/array_spec.txt`; `slurm/summarize.sbatch` provides the strict dependent
aggregation job.  A completed branch contains the full phase-space text dump,
final LAMMPS restart, stdout/logs, immutable branch metadata, and a validated
`outcome.json`.  `run-task` is idempotent for complete branches and refuses to
overwrite partial output.  Summarize only after every branch completes:

```bash
python scripts/run_lammps_meam_shooting_campaign.py summarize --campaign-root "$ROOT"
```

Production branches use Slurm's PMI-2 launcher (`srun --mpi=pmi2`) rather than
starting a separate MPICH Hydra process manager inside every allocation.  This is
important when several array tasks share a large CPU node.  Wave controllers use
an `afterany` dependency so a transient branch failure does not prevent all later
branches from being submitted; the final summary remains strict and reports every
missing branch.

If an unstarted branch directory is absent, `run-task` reconstructs its immutable
`metadata.json` and `in.lammps` directly from the campaign manifest.  It never
repairs or overwrites an existing incomplete directory; partial trajectories still
require explicit inspection before retrying.

The fixed-cell volume is inherited from each NPT parent.  Results estimate a
position-conditioned Langevin-NVT operator; they must not be mixed silently with the
original Nose-Hoover NPT transition operator.
