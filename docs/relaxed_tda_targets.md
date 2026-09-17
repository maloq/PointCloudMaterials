# Expanded Al relaxed-TDA targets

Recipe: [`configs/simulation/relaxed_tda_al.json`](../configs/simulation/relaxed_tda_al.json).
Implementation: `src/data/relaxed_targets/`. This generates labels from existing
trajectories; it does not run new molecular dynamics or train an encoder.

## Scope and target definition

- The current local-predictability cohort's **90 training sources**, using all
  **64 preselected pool centers** at 3 ps intervals from 48 to 498 ps inclusive:
  **869,760 planned center/window targets**, versus 23,040 native training windows.
  Existing validation/test sources are not added to the training set.
- All completed branches in the registered position-shooting, nested first-passage,
  fixed-24 ps continuation and completed smoke collections. Sample their existing
  frames on a **3 ps physical-time grid**, up to the available duration (maximum
  72 ps). This covers all valid trajectories, **not every stored frame**. An
  event-stopped final frame off that grid is not appended.
- Each shooting parent gets 64 deterministic, uniformly selected center IDs,
  shared across siblings and derived continuations. Selection does not inspect
  future outcomes. Smoke collections are diagnostic-only.

The full 70,304-atom periodic cell is minimized at fixed box with its generating
Lee–Shim–Baskes 2003 Al MEAM potential, FIRE, initial timestep 0.001 ps and force
tolerance 0.01 eV/Å. Unconverged cells have explicit failure records and no targets.
The target is [relaxed topology](research_glossary.md#relaxed-topology), **not a
future-time observation**.

Select the nearest 80 atoms **in the observed frame**, including the center, then
retain those exact IDs after minimization. Compute the existing raw 144D
[persistence image](research_glossary.md#persistence-image-and-tda-vector) from
centered float32 offsets, before quantizing the full relaxed cell. Also retain the
matched instantaneous target and both local clouds. There is no PCA, target
normalization or fitted transform in this release. The existing 3.5 Å death cutoff
remains part of the descriptor; this does not introduce a smooth-boundary TDA variant.

## Ancestry, duplication and selection

`technical/plan.json` freezes source paths, manifest hashes, original split labels,
lineages, parent IDs, center IDs and frame indices before minimization. Exact
whole-trajectory position/timeline duplicates are listed as exclusions. Prepared
copies without additional trajectories are explicitly excluded, as are branches
without completed outcomes. Float16 continuations are distinct inputs from their
float32 ancestors, with ancestry retained; they are not independent evidence.

A lineage with **any held-out relative is ineligible for training**, including
conflicts between older position-shooting and nested-shooting split assignments.
The original declared labels remain visible. This release does not redefine an
evaluation split: its `heldout_or_diagnostic` records require the original
protocol-specific selection/calibration/test roles when used later. Shooting
parents were selected by earlier protocols and first-passage trajectories have
state-dependent follow-up; they must not be treated as an unbiased ordinary-MD
population or as independent windows.

## Execution and resume

```bash
python -m src.data.relaxed_targets prepare --config configs/simulation/relaxed_tda_al.json
python -m src.data.relaxed_targets run --config configs/simulation/relaxed_tda_al.json \
  --worker cpu0 --ranks 32 --hours 16
python -m src.data.relaxed_targets status --config configs/simulation/relaxed_tda_al.json
```

Use `pointnet` and a 32-rank CPU allocation. The machine's configured LAMMPS/MPI
launcher supplies the backend; workers refuse changes to the bound executable,
implementation or minimization settings within a release. Each worker shares the
frozen queue using process locks, prioritizes coverage across families and then
densifies time. Exact duplicate cells with the same center IDs reuse one target.
Use a unique worker name for each process. `--max-tasks` is for an end-to-end pilot.
`--retry-failed` explicitly revisits failures while preserving failed relaxation
attempts. A worker stops after three failures for inspection; no failed cell is
silently accepted. A bounded allocation reports `partial_resumable` until every
planned task has succeeded. Inspect process/Slurm status as well as progress files;
a killed worker cannot update its own status.

The initial production wave uses four detached 32-rank CPU workers for up to
16 hours each. This is a first wave of the complete queue, not a promise that all
planned targets finish in 16 hours. No GPU allocation is needed by these workers.
Execution specifications, source snapshots and Slurm logs are under
`output/relaxed_tda/al-expanded-20260917/technical/`.

The September 17 frozen plan contains 1,090 trajectories (90 independent training
and 1,000 shooting/derived/diagnostic trajectories), 25,991 sampled frames and
1,663,424 center/window references. Of these, 1,280,960 are training-eligible.
These are planned counts, not completed or independent observations. There are
120 excluded unfinished branches and six cross-protocol lineage split conflicts,
all kept out of training. The production array is **Slurm 997270, tasks 0–3**.
`technical/launch.json` records submission and validation; use `status` for progress.

## Artifacts and precision

- IDS: `${storage:cache}/relaxed_tda/al-expanded-20260917/frames/KEY/targets.npz`.
  Arrays: `center_atom_ids [C]`, `neighbor_atom_ids [C,80]`,
  `observed_clouds [C,80,3]`, `relaxed_clouds [C,80,3]`, and paired
  `instantaneous_tda` / `relaxed_tda [C,144]`. Coordinates/targets are float32;
  atom IDs are exact integers. `complete.json` records hashes and convergence.
- `technical/tasks/ID.json` maps a frozen source/frame task to its target file.
  Many task records can refer to one identical target; retain those relationships
  when constructing training weights or performing source-level uncertainty analysis.
- SCRATCH: per-cell minimization attempts. Completed full cells are converted
  through `scripts/convert_trajectory.py relaxation --local-cloud-dtype float32
  --delete-source`, which verifies float16 rounding/checksums before deleting the
  text dump and LAMMPS input data. Centered float32 target clouds are already saved.
- STORE: `${storage:archive}/relaxed_tda/al-expanded-20260917/frames/KEY/`, containing
  full-cell float16 positions, exact IDs/timestep, float32 boxes, convergence,
  potential hashes, quantization error and verified archive checksums. Failed
  relaxation attempts are preserved separately under `failures/`.

The first selected observed frame is reloaded from its original producer; its
positions are box-relative and explicitly shifted to absolute coordinates for
LAMMPS. This avoids applying the box origin twice. Existing global float16 input
quantization cannot be undone by relaxation and remains recorded per source.

The expanded targets are **not automatically wired into the current MACE/GATr
training runs**. A later training recipe must join their source/frame/center keys
to causal observations and respect `training_eligible` and the original ancestry.
