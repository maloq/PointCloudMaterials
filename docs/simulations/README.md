# Simulations and datasets

[Local80 MLIP response collection](local_response_20261002.md) samples fixed-Al64 parents with inherited source roles and gates surrounding-environment convergence before training.


[Halfway stopping for unstarted Al sources](al_main_half_stop_20261001/README.md):
user-authorized version for 128 pending trajectories, confirmed 50% crystal
plus a 6 ps tail and declared peer caps; current dynamics and complete full-length
trajectories remain intact.

**Response-atlas Al256 pilot:** [execution](../response_atlas.md), fixed MACE-MPA-0, controlled FCC perturbations; development-only query bundles, not independent liquid trajectories.


[Full liquid-information relaxation](liquid_full_relaxation_20260928.md): 2,130 missing quenches, fixed ancestry and automatic downstream experiment queue.

[Main Al 0.1 ps reruns, 2026-09-27](al_main_010ps_20260927/README.md): all 150
retained melt ancestors, originally 600 ps with velocities, exact 0.1 ps output and frozen
ancestry roles; CPU waves publish verified canonical trajectories to STORE.
The [October 1 continuation](al_main_010ps_20260927/README.md#october-1-continuation)
recovers three interrupted native restarts and uses two sources per worker wave.
[Two additional Al sources at 0.01 ps](al_main_001ps_20261001/README.md) retain
the 400/520 K train ancestors and save velocities.
The [150-source duration audit](../../output/al_duration/all150-20261001/README.md)
finds substantial late growth and misleading temporary plateaus under the earlier
full-plateau objective.
The user's subsequent **50%-crystal target** supports much earlier individual
stops in the [halfway/90%-peer analysis](../../output/al_duration/half-crystal-20261001/README.md).
The user-authorized [queue version](al_main_half_stop_20261001/README.md) now
applies that target to unstarted sources. Already-running sources retain their
full 600 ps protocol; earlier full-length data remain available.

[Detailed holdings after cleanup, 2026-09-26](inventory_20260926_after_cleanup/README.md): materials, atom
counts, observed durations/cadences, sampled velocities, formats and measured
disk allocation; includes collection and trajectory CSVs, duplicates and active runs.
Refresh into a new dated directory with `python scripts/project.py simulations --details --output DIRECTORY`.
The [earlier inventory](inventory_20260926/README.md) is a historical snapshot.
[Cleanup and Al cadence audit](cleanup_20260926/README.md) records the authorized
coarse/failed-payload deletion and how the main Al sources can be rerun more densely.

Simulation production and inventories belong here. `experiments/` is reserved for
scientific comparisons using the data.

**[Dataset registry: all materials, potentials and current schemas](../../DATASETS.md)**
is the main discovery entry point. Refresh with
`python scripts/project.py datasets --refresh`. The CSVs below retain their
distinct role as producer-outcome indexes.

Deferred launcher retirement and compatibility exclusions are listed in the
[post-queue cleanup checklist](../src_refactor.md#post-queue-cleanup-checklist).

The location database is [configs/datasets.json](../../configs/datasets.json).
It records stable IDs, physical storage roles, aliases and dependencies. The readable
[collection catalog](collections.csv) and [run-record index](run_records.csv) are
exported from it with:

```bash
python scripts/project.py simulations --output docs/simulations
```

The run index lists producer `outcome.json` records, including superseded attempts,
duplicate copies and fixtures. Its `recorded_state` is the producer's statement,
not a fresh validation or a count of independent simulations. Missing outcomes do
not imply success. `collections.csv` also includes prepared roots without outcomes.
The exporter does not hash large arrays, deduplicate trajectories or query Slurm.
See [the September 13 inventory](inventory_20260913.md) for checked canonical counts,
protocols, precision, lineage and duplicate exclusions. Its storage measurements
predate the [storage relocation](../data_storage.md).

| Collection | Canonical evidence at the September 13 inventory |
| --- | --- |
| Independently melted Al MEAM sources | 126 of 150 complete: 30 each at 400/450/500/510 K, six at 520 K |
| Al position shooting | 480 original futures plus the accepted 160-future 15 ps top-up; mixed horizons |
| Al nested shooting | 144 completed branches; the fixed-24-ps set is a derived collection |
| Ti crystallization | One source and six unique branches; six additional copies are duplicates |
| Ta branches | Six complete; one 1,024,000-atom baseline and five 10,000,422-atom runs |
| New 100,000-atom Al source | Incomplete near 748 ps; not in the active simulation queue at review |
| Portability Ti smoke | 128 atoms, 20 steps; infrastructure fixture, excluded from research counts |

Maintained workflows: [scripts](../../scripts/README.md),
[simulation recipes](../../configs/simulation/README.md),
[conversion](../trajectory_conversion.md) and [storage/publication](../portability.md).
New elemental runs use `scripts/run_lammps_campaign.py elemental run --config
configs/simulation/ti_crystallization.json --run-name NAME`; choose Al/Ta explicitly.

- [More spontaneous crystal births: proposed campaign, 2026-09-26](nucleus_birth_campaign_proposal_20260926.md):
  literature, training-only yield audit, independent Al temperature/size sweep,
  early training collection and fixed-duration evaluation; revised launch below.
- [Uniform-temperature Al birth screen, 2026-09-26](al_birth_uniform_20260926.md):
  authorized first stage, 22 fresh sources at 11 temperatures, six detached CPU
  workers; continuous observations and an early-transformation stopping rule.
- [Ta position shooting, 2026-09-26](ta_shooting_20260926.md): six archived full-cell
  parents, four velocity replicas each; [potential literature review](ta_potential_review_20260926.md)
  and explicit force-field/ancestry limitations.
- [Predictive-memory precision sources, 2026-09-17](predictive_memory_precision_20260917.md):
  12 fresh Al lineages, 192 ps at 0.075 ps cadence, retained float32/float16 pairs
  and four preassigned sealed test sources; separate CPU production.
- [Expanded Al relaxed-TDA targets](../relaxed_tda_targets.md): fixed-cell
  minimization of denser existing training frames and completed shooting data,
  with verified relaxed cells archived on STORE and target caches on IDS.

Historical campaign documentation and exact configurations:

- [Al source and branches](al_crystallization/README.md)
- [Ti/Ta crystallization](ti_ta_crystallization/README.md)
- [Independent Al source queues](independent_al_sources/README.md)
- [Ta baseline](ta_baseline/README.md)
- [Restart audit](restart_audit/README.md)

On September 13 the live queue contained only GPU allocations 990987, 991149 and
991141. Their processes were inspected: current forecast queues use retained
experiment/output paths; no simulation controller remained active. Historical
external batch scripts were left unchanged. For a new launch use the maintained
commands and current recipes; old exact launcher paths remain in the STORE repo copy.

- [Million-atom Al launch, 2026-09-13](al_crystallization/MILLION_ATOMS_20260913.md):
  same crystallization source protocol as the recent 100k Al run, without branches, capped at 400 ps after melting; full melt saved; Slurm job 991395.

- [Remaining 520 K sources, 2026-09-13](independent_al_sources/RECOVERY_20260913.md):
  six of thirty initially complete; remaining 24 submitted as Slurm array 991371.

Response-atlas follow-up (2026-10-01): stronger toy controls and fixed-parent shot precision; see `experiments/response_atlas_20261001/FOLLOWUP.md` and `docs/response_atlas.md`.
