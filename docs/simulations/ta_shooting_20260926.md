# Ta position shooting, 2026-09-26

Submitted 2026-09-26 at 11:55 UTC: Slurm array **1009563**, dependent preservation
job **1009564**. Completion is not yet established; consult the durable receipts
and per-parent outcomes below.

Requested extension of the Al shooting work to Ta. The
[potential review](ta_potential_review_20260926.md) supports keeping Zhong/Sheng
2014 EAM as the primary baseline without claiming it is the best available model.

## Frozen protocol

[Recipe](../../configs/simulation/ta_shooting_20260926.json): six original archived
parents, four independently seeded velocities per parent, 24 ps per shot at
1900 K and zero pressure. The existing elemental producer retains its 2 fs
integration step, NPT thermostat/barostat settings and 0.1 ps observations.
No encoder or predictor is run by this campaign.

The `model_1m` parent contains 1,024,000 atoms. The 2.7, 2.8, 2.9, 3.0 and 3.60 ns
parents each contain 10,000,422 atoms. All use their original full periodic cells
and unquantized archived position files. Input SHA256 checks, atom counts and
dump headers were verified before preparation. The original generating potential
and common ancestry are unknown; all descendants are conservatively grouped as
one preparation lineage. Selection includes all six available parents without
new outcome-dependent screening.

This is an exploratory finite-horizon position-conditioned ensemble. It neither
adds independent melt preparations nor defines a true committor experiment.
Four shots are insufficient for a precise per-configuration probability.

## Execution and preservation

Use conda `pointnet-torch214`. The machine's established external LAMMPS executable
and MPI environment are recorded separately; Python's conda environment is not
the LAMMPS installation. Code, machine settings, executable checksum and prepared
recipes are retained at submission. The frozen code has its own artifact Git
commit for the existing provenance recorder; `git_commit.txt` and
`working_tree.patch` identify the original checkout separately.

The Slurm array contains six tasks, each running four shots sequentially. At most
two tasks run concurrently, each with 48 CPU MPI ranks, 96 GB and a 36-hour limit.
This is 24 trajectories, 576 ps of aggregate simulated time. Historical large-cell
timings were about 81 minutes of dynamics per shot before conversion and archiving;
they are not a completion-time guarantee on the new nodes.

Commands (prepare and submit refuse an existing preparation/submission):

```bash
python -m src.simulation.campaigns.position_shooting prepare --config configs/simulation/ta_shooting_20260926.json
python -m src.simulation.campaigns.position_shooting submit --launch /store/PERSO/vmorozov/simulation-launches/ta-shooting-20260926
```

Durable campaign receipts, source/config hashes and Slurm logs:
`/store/PERSO/vmorozov/simulation-launches/ta-shooting-20260926/`.
The live `submission.json` contains actual job IDs; `status.json` distinguishes
prepared/submitted/completed/failed states. A submitted state is not proof that
all tasks have started.

During execution, each parent lives at
`/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/ta-shooting-20260926-parentXX/`.
Completed parents publish with verified copies to
`/store/PERSO/vmorozov/simulations/ta-shooting-20260926-parentXX/`.
Per-shot data are in `branches/shot00` through `branches/shot03`.
Precise LAMMPS restarts are retained every 8 ps and at completion. Existing
`scripts/convert_trajectory.py` verifies float16 observations, float32 boxes and
exact identity/timeline arrays before disposable text deletion.

An `afterany` preservation task archives stopped failures and their restart state
under the corresponding `-stopped` STORE IDs, records partial completion and
updates dataset ancestry. It checks that no array task remains active before
archiving. Its own errors must be checked before any SCRATCH purge; submission
alone is not proof of successful preservation.

The aggregate dataset ID is `ta-shooting-20260926`. Per-parent publications get
their own registered IDs and dependencies. This collection is not part of the
fixed Al comparison or a new train/test split.
