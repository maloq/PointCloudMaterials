# Simulations and datasets

- [Dense deterministic Ni response observations](ni_phase_nve_20261013.md):630 full-cell100fs continuations for atomic Q,V and latent tangent prediction.

[Thirty additional Ni crystallization runs](ni_births_lowT_010_20261012.md):
30 fresh 32,000-atom melts over the existing six low-temperature settings,
six A40 workers, a 400-ps hold cap, and dependent GPU birth extraction.

New Ni relaxations default to a maximum force-component tolerance of
**0.05 eV/angstrom**; use the [current settings preset](../../configs/simulation/ni_relaxation_default.json).
Historical campaigns keep their recorded tolerances.

[Ni birth-window GPU relaxation](ni_birth_relaxation_20261008.md): full periodic
fixed-box FIRE for native 0.10 ps windows from -6 to +6 ps around all 70 currently
catalogued established liquid-origin Ni births; shared native-state cache,
deduplicated source frames and audited persistent CUDA callback optimization.

[Ni low-temperature nucleus-birth campaign](ni_births_lowT_010_20261008.md):
36 new independent melts, six each at 950/1000/1050/1100/1125/1150 K,
four GPU workers, native 0.10 ps positions/velocities and automatic birth search.

[Million-atom Al velocity recovery](al_million_velocity_replay_20261007.md):
checkpoint-anchored native CPU replay of 400 ps production, with 0.10 ps Q/V,
per-section fidelity gates and no added independent ancestry.

[Independent Ni nucleus-birth campaign](ni_births_010_20261007.md): the original twelve
plus twelve additional independent 32,000-atom melts, 1100–1400K NPT, 0.10ps
positions/velocities, three added GPU workers and automatic birth search.

[Fast Ni EAM/FS on GPU](ni_eam_gpu_20261007.md): pinned Mendelev 2012 tables, analytic force tangents, graph replay and LAMMPS numerical validation.

[Additional dense Al V100 reshoots](al_dense_shooting_v100_20261007.md): 40 planned
velocity reshoots from 20 existing liquid parents, exact 0.10 ps Q/V, inherited
roles and native NPT; one V100 with a combined 24-hour validation/production budget.

[Continuous-motion tangent teachers](../continuous_motion.md): outcome-blind native Q,V, full-cell100fs BAOAB with float64 integration, and a frozen1600-update training campaign with an extended execution window.

[Native-velocity response training to 0.5 ps](response_native500_20261006.md): float64 integration, source temperature, weaker damping, four model arms and three fitting seeds.

[Box and thermostat continuation controls](nucleus_dynamics_ablation_20261006.md): 384 new 12 ps NVT/damping controls on the same 64 calibration states, nine-hour cap.

**October 4 Ti duplicate cleanup:** six `Ti_early_slurm` branch exports were removed after checksum comparison of all 36 arrays against retained `Ti/branches` files. [Deletion receipt](../storage/ti_duplicates_float32_20261004/deletion.json). Retained Ta/Ti text snapshots receive verified float32 sidecars; existing float16 trajectories and native restart/input files keep their precision.

[Dense Al on GPU MEAM](al_dense_gpu_20261004.md): five unstarted 70,304-atom sources transferred to two GPUs, with native isotropic NPT and original dense sampling/stopping preserved.

[Differentiable CUDA MEAM on Blackwell](meam_gpu_blackwell_20261004.md): compiled analytic response derivatives, precision gates, shared-liquid200ps and separate seeded-growth controls.

[Periodic16k MEAM CPU/GPU comparison](meam_md_16k_20261004.md): new shared liquid,200ps at450K, native-precision divergence and compiled force benchmarks.

[Initial-velocity MEAM response](response_velocity_20261004.md): new float32 compiled CUDA MEAM full-cohort comparison at0.1/0.5ps; separate three-hour0.1/0.5/1ps subset, stopped original large campaign and recorded prior validation.

[Differentiable Lee2003 Al MEAM](response_meam_torch_20261004.md) validates the exact PyTorch port and measures complete20/100fs responses against MACE on the same GPU.


[Expanded unrelaxed local80 responses](response_geometry_20261003.md):720 training geometries, generic future-target ablations and two-GPU detached collection/training.


**Ta holdings after October 3 cleanup:** the earlier million-atom baseline and
original Ta snapshots remain. The five earlier large-cell runs and 24 shooting
futures were explicitly deleted; historical inventories below describe their
former availability. Reports/provenance remain in
[the cleanup archive](../storage/ta_simulation_removal_20261003/preserved.json).

[October 3 Al rerun recovery](al_main_half_stop_20261001/README.md#october-3-quota-recovery)
resumes native checkpoints from the stopped exact 0.1/0.01 ps queue with verified
binary prefixes; temporary execution is on STORE after SCRATCH quota failure.

[Force-only UNEP/MEAM response comparison](response_force_engines_20261002.md)
measures complete common-noise finite-difference bundles and checks potential-specific
halos against the existing local80 MACE reference.

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

- [Local80 finite-displacement forwards](local_response_followup_20261003.md): common-noise20/100fs simulation at0.0034–0.338Å per-atom RMS and matched training controls.

- [Saved-state nucleus recovery, October 4](nucleus_recovery_20261004.md): known-site recovery vs restart offset and paired precision controls.

- [Dissolved-embryo and random-liquid controls, October 5](nucleus_controls_20261005.md): matched 12 ps continuations, shared 16-hour deadline.
- [Additional negative controls, October 5](nucleus_negative_extra_20261005.md): new liquid ancestors and repeated embryo shots, shared three-hour deadline.
- [Continuation bias and birth-rate calibration](nucleus_continuation_bias_20261005.md): paired historical audit and present-selected, matched-temperature embryo continuations.

- [Ni relaxation optimizer speed audit, October 11](ni_relaxation_speed_20261011.md): matched GPU FIRE/ABC-FIRE and ASE preconditioning benchmarks.
