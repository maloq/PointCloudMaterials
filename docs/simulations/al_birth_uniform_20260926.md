# Al birth campaign: uniform temperature grid

Authorized 26 September 2026 following the
[birth-production proposal](nucleus_birth_campaign_proposal_20260926.md).
The user requested more uniformly spaced temperatures and fewer replicas.

The first stage contains **22 independently melted sources: two each at
400, 410, 420, 430, 440, 450, 460, 470, 480, 490 and 500 K**. All are development
sources, assigned `train` before dynamics. Two replicas give limited uncertainty
resolution; this screen is not an independently held-out model evaluation.
The later size comparison and production expansion await this screen's results.

Submitted at **11:37 CEST on 26 September**, Slurm array **1009492**, workers
`1009492_0`–`1009492_5`. Initial nodes: nodecpu04, nodecpu03 (two workers),
nodecpu09, nodecpu08 and nodecpu07. This queue is detached from the interactive
GPU allocation. The 36-hour worker limit is a ceiling, not a finish estimate.

Before submission, local 256-atom fixtures completed the 300 ps melt/QC,
a duration-limited hold, exact paired conversion of 21 frames, and a separate
positive stopping case that ended at step 4,000, exactly 12 ps after crossing.
These checks validate execution paths, not physical event yield or finite-size
convergence. No automated test suite or online tracking run was added.

## Scientific contract

- 70,304 Al atoms, cubic periodic cell, existing pinned Lee2003 2NN-MEAM files.
- Fresh independent melt and velocity seeds, checked against both old Al source
  manifests and all planned September 17 precision-campaign seeds. No continuation
  of that explicitly stopped campaign.
- 300 ps at 1325 K, then one target-temperature velocity initialization.
  Zero-pressure isotropic NPT, 3 fs steps, thermostat 0.3 ps, barostat 3 ps,
  COM removal every 0.3 ps, matching the retained source dynamics.
- Save from target-temperature initialization at step zero, every 50 steps
  (0.15 ps). The first 5,000 steps (15 ps) remain a preparation segment.
  Main full-history examples must lie entirely after this segment; earlier
  observations and births remain explicit rather than being hidden.
- Check full-cell PTM every 500 steps (1.5 ps), including step zero. Once the
  crystalline fraction first reaches 10%, continue for 4,000 steps (12 ps),
  capped at 205,000 total steps (615 ps). No velocity or thermostat resets
  between checks. Checkpointing and observation do not change forces.
- The operational stopping criterion is FCC/HCP/BCC PTM fraction, RMSD cutoff
  0.1. It does not determine birth labels or claim a physical critical nucleus.
  Later lineage analysis distinguishes emergence, arrival and ambiguity.
- Melt QC examines 11 frames in its final 15 ps: each must have less than 1%
  crystalline atoms and no 64-atom cluster (3.6 Å connectivity). Record the
  last-150-ps MSD, requiring at least 10 Å², plus final RDF. Reject and preserve
  failed preparation; do not silently reroll seeds.
- Temperature, elapsed simulation time and absolute time are audit metadata,
  not model inputs. No model fitting, AP selection or W&B training run occurs.

Uniform observations and zero-event runs are retained. The early-stopped
population is restricted to early transformation: do not treat its raw frequency
as a fixed-duration equilibrium nucleation rate. Incomplete future windows are
censored, not negative. For later 6 ps labels, reserve 6 ps follow-up plus the
declared persistence-confirmation interval.

## Execution and preservation

[Recipe](../../configs/simulation/al_birth_uniform_20260926.json), implemented by
[`birth_sources.py`](../../src/simulation/campaigns/birth_sources.py) through the
maintained simulation dispatcher. Six CPU workers, 48 MPI ranks each, no GPUs;
36-hour allocation ceiling per worker. Sources are assigned round-robin across
workers, so each worker receives several temperatures.

```bash
conda run -n pointnet-torch214 python scripts/run_lammps_campaign.py birth-sources prepare \
  --config configs/simulation/al_birth_uniform_20260926.json \
  --run-name al-birth-uniform-20260926
conda run -n pointnet-torch214 python scripts/run_lammps_campaign.py birth-sources submit \
  --campaign-root '${storage:simulation_runs}/al-birth-uniform-20260926'
```

Preparation and submission refuse existing attempts. A source failure stops
its worker with an explicit error; other workers remain independent. Slurm
sends an advance signal before the time limit to terminate dynamics and archive
partial trajectories and full-precision restart states. Uncatchable failures
still require inspection/publication before SCRATCH purge; there is no silent
restart from quantized observations.

Simulation data stage on SCRATCH. Launch records, code snapshot, source hashes,
manifest and Slurm logs live durably at
`${storage:archive}/simulation-launches/al-birth-uniform-20260926/`.
Workers verify the frozen code and prepared inputs before dynamics. Each source
is converted through `scripts/convert_trajectory.py birth-pair`, retaining paired
float32/float16 observations, float32 boxes, exact identities/timeline, precision
errors and checksums. Temporary ASCII observations are deleted only after
verification. Native melt/final and rolling restart states remain intact.
Completed sources publish to STORE with a verified copy and registration.

Source progress lives in `runs/<source>/status.json`, `source_progress.json` and
native LAMMPS logs. Worker summaries live in `workers/worker-<index>.json`;
the campaign `status.json` aggregates source completion. Scheduling/progress
is not a completed birth count; full lineage analysis follows collection.

Local runtime-validation fixtures are outside the repository and outside the
scientific campaign. They do not count as sources or create online training logs.
