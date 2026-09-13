# IDS dataset inventory — 13 September 2026

This is a read-only inventory of `/home/ids/vmorozov`, observed around
16:54–17:04 CEST. The six research directories total **507.81 GiB allocated**;
**415.95 GiB** is under `simulations`. The entire IDS home, including small
hidden directories, is **507.85 GiB**. Sizes include trajectories, restarts,
source snapshots, provenance, and retained interrupted attempts. GiB means
2^30 bytes. These are filesystem allocated bytes, not logical array sizes or a
measurement of remaining personal quota. The scan is not an atomic snapshot.

The inventory reads campaign manifests, canonical outcome records, binary
manifests, selected NPY headers, current Slurm state, and `du` block counts.
It does **not** rehash every large trajectory. “Complete” below means a canonical
producer completion record and the corresponding binary manifest are present.
Declared campaign counts are not substituted for completed trajectories.

## Storage map

| IDS directory | Allocated GiB | Contents |
|---|---:|---|
| `simulations/` | 415.95 | Molecular-dynamics sources, shooting trajectories, restarts, partial runs and provenance |
| `training-cache/` | 54.51 | Derived local atomic neighborhoods, legacy temporal caches and frozen embeddings |
| `experiments/` | 14.29 | Older training inputs, targets, cached representations and experiment artifacts |
| `data/` | 18.94 | Other research: geospatial/fire-prediction features, MODIS, land features, night lights, Parquet tables |
| `analysis/` | 2.71 | MACE analysis products, plots, representative structures and metric records |
| `models/` | 1.41 | Older predictive-model checkpoints, including CatBoost/LSTM/MLP models; not MD interaction potentials |

The existing repository storage command also produced a **54.39 GiB local-output**
report at `output/maintenance/storage/STORAGE.md`. That figure is outside these
IDS totals: the command intentionally does not follow external dataset symlinks.

## Interaction potentials

| Material/data family | Potential actually used | Files and implementation |
|---|---|---|
| Al MEAM sources, all Al MEAM shooting families, new 100k Al campaign | Lee–Shim–Baskes 2003 Al **2NN-MEAM** | `Lee2003_Al.library.meam` + `Lee2003_Al.meam`; LAMMPS `pair_style meam`, pure-Al mapping |
| Ti crystallization and branches | Kavousi et al. 2019 **Ti/Ni 2NN-MEAM**, Ti component only | `Kavousi2019_NiTi.library.meam` + `Kavousi2019_NiTi.meam`; `pair_coeff * * LIB Ni Ti PARAM Ti`; no Ni atoms |
| Ta baseline and additional branches | Zhong et al. 2014 Ta **EAM** | `Ta_Zhong2014.lammps.eam`; `pair_style eam/alloy` |
| Al archived-snapshot dataset, outside IDS | Mendelev et al. 2008 Al EAM | `Al1.eam.fs`; NIST version recorded in the dataset |
| Mg archived-snapshot dataset, outside IDS | Wilson–Mendelev 2016 Mg EAM | `Mg1.eam.fs`; NIST version recorded in the dataset |

Exact recorded SHA256 identifiers:

- Al MEAM library: `f72f19b5185e6da9c4e4c26029346b9210296b289ba791178dee1e923281835e`.
- Al MEAM parameters: `b1ba33a29d8884692aeb4a1f0c78df51146f6f68d281121135dfca3207506e6a`.
- Ti/Ni MEAM library: `9ed750c73c224dc3acd168410e70cad4a79bc201bab308b265b3c5dfab11972d`.
- Ti/Ni MEAM parameters: `6f7ff46d837819f4b1b42ace00ab7f53987111d9d5586628f080714178c3d4fe`.
- Ta EAM: `8908993117f2502ed48bd31b737556719c3f7f11d1e7b7213eb257cd1ca42386`.
- Al EAM: `768a9ad9b0cda57f36523b5d247942130101b26b0cbbd9d30c7bd7e1decc7ae3`.
- Mg EAM: `0ceed5387f16d0cb7f4a2088fc4665e27708dc28e3e5b3a6c2e01ac0528faba2`.

These identify the repository files, including their provenance comments. Old
manifest dtype fields can predate conversion; current binary manifests and
conversion reports determine the stored precision.

## Al: independently melted source histories

These are the largest collection of genuinely separate melt preparations.
Every source contains **70,304 atoms**, its own 300 ps melt at **1325 K**, a
separate velocity initialization, 15 ps equilibration, and a **600 ps NPT
measurement history**. Measurement timestep is **3 fs**, output cadence
**0.75 ps**, giving **801 frames** including time zero. Positions and velocities
are both stored as float16, box bounds as float32, atom IDs/timesteps as int64,
and atom types as int32. Sources include PTM/crystalline-cluster progress,
thermodynamic arrays, nucleation/candidate records and a final restart.

| Temperature | Complete | Planned | Physical measurement time available |
|---|---:|---:|---:|
| 400 K | 30 | 30 | 18 ns |
| 450 K | 30 | 30 | 18 ns |
| 500 K | 30 | 30 | 18 ns |
| 510 K | 30 | 30 | 18 ns |
| 520 K | 6 | 30 | 3.6 ns |
| **Total** | **126** | **150** | **75.6 ns** |

That is **100,926 stored full-system frames** from completed sources. The
400/450/500 K root occupies **59.50 GiB**; the 510/520 K root occupies
**29.71 GiB**. The second figure includes unfinished histories and retained
recovery artifacts. At inspection there were no Al source jobs in the live
Slurm queue; **24 intended 520 K sources still have no complete outcome**.
A historical “submitted” status alone is not evidence that the queue is active.

Exact roots under `/home/ids/vmorozov/simulations/`:

- `al_meam_independent_sources_70304_400-500K_30perT_float16_20260902`
- `al_meam_independent_sources_70304_510-520K_30perT_float16_20260903`

The design allocates 18 optimization, six model-selection and six final-validation
sources per temperature. Descendants inherit their source split. These are
unseeded supercooled liquids: crystals form spontaneously after full melting.

There is also an older **nine-run, 11.85 GiB** collection:
`al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901`.
All nine runs are complete: three per 400/450/500 K, 70,304 atoms, 600 ps, 3 fs
steps and **201 frames at 3 ps cadence**. Its temporal binaries store float32
positions, without a velocity array. Despite “independent” in its directory
name, its manifest explicitly says the runs share validated liquid positions
and cell, and differ by velocity initialization. It is a different independence
contract from the 126 separately melted sources.

The older 12-temperature/72-replica preparation at
`al_homogeneous_unseeded_2nn_meam_70304_12temps_390-500K_600ps_1ps_positions_velocities_6seeds_20260830`
is only **0.083 GiB** here. Its temperature statuses are prepared with no
completed replicas; do not count its nominal 72 trajectories as existing data.

## Al: shooting trajectories

All following families use **70,304 atoms** and the same Lee 2003 Al MEAM
potential, but they differ in thermostat, horizon, branching design and storage.
Multiple futures from a parent are conditional realizations, not independent
source structures.

| Collection | Completed / declared branches | Horizon and frames | Stored position/velocity dtype | Allocated GiB |
|---|---:|---|---|---:|
| Original 40 parents × 8 shots | 320/320 | 48 ps; 161 frames | float32 | 82.53 |
| One-shot local supplement | 40/40 | 48 ps; 161 frames | float32 | 10.43 |
| Two-shot local supplement | 80/80 | 48 ps; 161 frames | float32 | 20.73 |
| Later 40-branch supplement | 40/40 | 48 ps; 161 frames | float32 | 10.43 |
| Superseded 48 ps top-up | 40/160 | 48 ps; 161 frames | float32 | 10.44 |
| Accepted 15 ps top-up | 160/160 | 15 ps; 51 frames | float32 | 13.67 |
| 48 ps smoke test | 16/16 | 48 ps; 161 frames | float32 | 4.19 |
| 15 ps smoke test | 16/16 | 15 ps; 51 frames | float32 | 2.64 |

The first four rows are the original **480 futures**, at **12 futures per each
of 40 parents**, spanning 400/450/500 K. The accepted 15 ps top-up adds four
futures per parent, producing the existing **640-future merged collection**.
It mixes 48 ps and 15 ps horizons. The partially completed superseded 48 ps
campaign and the smoke tests are separate products, not another complete top-up.

The 48 ps families use **fixed-cell Langevin NVT**, 3 fs steps, 0.3 ps frames,
and independently initialized momenta/noise. The 15 ps family uses fixed-cell
**`temp/csld` NVT**, also 3 fs and 0.3 ps. Its first 12 ps are the main prediction
window and the final 3 ps support shifted starts.

The exact-continuation test failed: a restarted 15+9 ps trajectory did not
match uninterrupted 24 ps dynamics. The rejected extension remains on disk
under the smoke directory but is **not accepted exact-continuation data**.
The requested 160-new-parent/2,560-future production campaign is not present as
completed data.

Exact shooting roots appear in the complete directory table below.

## Al: nested first-passage and fixed-24-ps data

`al_meam_nested_shooting_pilot_70304_400-500K_20260902` contains **144 canonical
completed branches from 36 parents**: 30 transition candidates and six controls,
with 48 branches at each of 400, 450 and 500 K. It occupies **19.84 GiB**.
The 2×2 nested design shares a momentum seed between two thermostat-noise
children. It uses 3 fs, fixed-cell Langevin NVT and **float32 positions and
velocities**, plus conservative initial-force information and basin/PTM records.

Runs stop on a persistent liquid/crystal basin arrival, with a **72 ps cap**.
Output is deliberately nonuniform: 0.03 ps initially, approximately 0.1 ps through
3 ps, then 0.3 ps, plus exact confirmation frames. Two retained invalid canary
outcomes are excluded from the canonical 144 count.

`al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible`
contains **144 complete, fixed-24-ps, 81-frame float16 artifacts**, occupying
**9.63 GiB**. It reuses original path segments and, when necessary, continues
from first-passage restarts with a new Langevin noise seed. Original first-passage
labels are retained. Thus it is a standardized descendant of the nested set,
not 144 new independent source realizations and not an exact reproduction of
uninterrupted original noise.

## Ti: crystallization and branches

Under `ti_ta_crystallization_20260907/`:

- `Ti/`: **13.93 GiB**. One completed **100,000-atom** source at **1250 K**,
  ending at **724 ps and 94.008% crystalline**. It started with a validated
  3000 K melt; no crystal seed was inserted. NPT at zero pressure, 1 fs timestep.
- The source has **7,241 frames**. Six completed branches have **240 ps and
  2,401 frames each**, sampled every 0.1 ps, with float16 positions and full
  restarts. Parents are at source times **0, 8, 32, 40, 56 and 192 ps**.
- The six parents belong to **one source lineage**. Branches reinitialize
  velocities; they are not exact continuations of the source velocities.
- `Ti_early_slurm/`: **8.45 GiB**, six complete branches. **Every array checksum
  matches the corresponding local Ti branch**, checked directly between all
  six pairs of current binary manifests. These are duplicate trajectories;
  count **six unique branches**, not 12 independent futures. The `mace-full`
  cache already excludes these duplicates.
- `Ti_shooting_round2_20260909/`: preparation/submission records only at this
  inspection; **no completed branch outputs**. The submitted six new seeds must
  not be counted as produced data.

Unique completed Ti sampling is **724 + 6×240 = 2,164 ps**, represented by
**21,647 frames** when each source/branch time-zero frame is included. Reused
starting frames and shared lineage mean these are not statistically independent
observations.

## Ta: six completed 24 ps branches

Potential: Zhong et al. 2014 EAM. NPT at **1900 K and zero pressure**, 2 fs
steps, 0.1 ps cadence, **241 frames per branch**, float16 positions. Each starts
from archived positions with new velocities; these are not independently melted
source histories or exact continuations of the old trajectories.

- `ta_initial_model_1m_24ps_npt_20260905`: one **1,024,000-atom** `model_1m`
  branch, **1.83 GiB** including its provenance/preflight.
- `ti_ta_crystallization_20260907/Ta`: five **10,000,422-atom** branches,
  **78.55 GiB**, from the **2.7, 2.8, 2.9, 3.0 and 3.60 ns** snapshots.
  They are approximately ten million atoms each, unlike the smaller baseline.

All six have complete outcomes and float16 binary manifests. Old paths named
`trajectory_binary_float32` can be compatibility symlinks to the converted
float16 directories. The Ta/Ti parent directory also contains **2.64 GiB of
interrupted-attempt archives** and **0.61 GiB of preflight data**; these are not
additional production trajectories.

## New Al crystallization source following the Ti workflow

`al_meam_crystallization_100k_450K_20260911`: **23.82 GiB**, **100,000 Al atoms**,
Lee 2003 MEAM, 300 ps melt at 1325 K, then **450 K NPT**, 1 fs steps and 0.1 ps
frames. Planned completion is >=94% crystalline in two checks 4 ps apart,
followed by six 240 ps branches.

**This is incomplete.** The last assessment is **748 ps, 83.571% crystalline**;
the last thermo entry is **748.5 ps**. The source log was last modified at
**02:04 CEST on September 12**, no matching MD job appears in the September 13
queue, and no complete source outcome or branch outputs exist. Its “running”
JSON is stale. The directory includes raw partial trajectory, snapshots and
checkpoints; it is not yet a verified completed float16 training trajectory.
The inventory did not resume or alter this run.

## Related Al/Mg EAM data are outside IDS

`/home/infres/vmorozov/PointCloudMaterials/datasets/zr_al_mg_initial_6x24ps`
resolves to the repository filesystem, **not `/home/ids`**. It contains:

| Material | Potential | Complete branches | Atoms per branch | Temperature | Horizon / cadence | Allocated GiB |
|---|---|---:|---:|---:|---|---:|
| Al | Mendelev 2008 EAM, `Al1.eam.fs` | 6 | 1,048,576 | 650 K | 24 ps / 0.1 ps | 7.52 |
| Mg | Wilson–Mendelev 2016 EAM, `Mg1.eam.fs` | 6 | 1,048,576 | 600 K | 24 ps / 0.1 ps | 7.35 |

Both use 1 fs NPT, 241 frames and float16 positions. Al parents: 166, 170,
174, 175, 177, 240 ps. Mg parents: 940, 960, 980, 990, 1000, 1500 ps.
The Zr trajectories were removed; the historical directory name remains.
The large raw-artifact archive is recorded at `/home/tehbek/lammps_archive/`,
also outside IDS and not measured here. Original static structures under
`datasets/Al`, `Mg`, `Ta`, `Zr` and `Al50Ni50` are likewise not IDS source runs.

Consequently, **Al EAM at 650 K and Al MEAM at 400–520 K/450 K are distinct
potentials and protocols**, not interchangeable samples of one simulator.

## Derived training datasets

| Cache under `training-cache/` | GiB | What it contains |
|---|---:|---|
| `embedding-forecast-full-20260911` | 48.91 | 125 source shards; frozen 256-component MACE embeddings, 1,024 center atoms × 801 times per source, float16; 0.75 ps cadence |
| `temporal` | 5.17 | Legacy temporal input caches; float32 coordinates, exact IDs/timesteps and float32 boxes; example shape `(201, 70304, 3)` |
| `mace-full` | 0.26 | 24 Al/Mg/Ta/Ti trajectories, 48 train/validation shards; 80-atom neighborhoods, anchor/spatial-neighbor/same-atom-future views, 0.1 ps lag, float16 |
| `mace-meam` | 0.17 | 90 view shards × 256 samples = 23,040 anchors; example views `(256, 3, 5, 80, 3)` float16; relaxed targets `(23040, 144)` float32 |

The embedding cache contains 30 sources per 400/450/500/510 K and five at
520 K: **125 cached sources**, although 126 physical sources are now complete.
Its frozen split is **74 train / 24 validation / 27 test sources**. Its input
history is 6 ps and forecast horizon 9 ps. Center IDs persist through time;
the encoder sees instantaneous periodic 80-neighbor environments. These arrays
are derived features, not another MD potential or new simulation trajectories.

`mace-full` includes six Al EAM, six Mg EAM, six Ta and six unique Ti branches.
Its own manifest warns that added Ta/Ti validation uses disjoint time blocks
and center pools **within shared trajectories/lineages**; it is not an
independent-source generalization test. `mace-meam` records source-radius
normalization and targets only for the anchor view. Preserve their manifests,
scalers, source identity and encoder/checkpoint hashes when interpreting results.

The remaining 14.29 GiB under `experiments/` includes approximately 6.72 GiB in
`pretrained_mace_spatiotemporal_20260906`, 2.97 GiB in
`temporal_hypotheses_12h_20260906`, 2.01 GiB in
`pretrained_mace_80_dt01_cosine_20260906`, and 1.09 GiB in
`geoframe_v2_spatiotemporal_Al_Mg_Ta_20260905`. These are older derived stores and
experiment artifacts, not four new physical datasets.

Other IDS research data includes roughly 8.85 GiB of boosted-model feature
Parquet datasets, 3.10 GiB of MODIS data, 2.84 GiB of land features, 2.21 GiB of
older saved features and 1.46 GiB of night-light data. This review records their
presence and sizes; it does not infer their row-level scientific contracts from
filenames.

## Loading and interpreting the atomistic data

The main arrays are `positions.npy`, optional `velocities.npy`, `box_low.npy`,
`box_high.npy`, `timesteps.npy`, `atom_ids.npy` and `atom_types.npy`, accompanied
by a checksum manifest. Units are angstrom, picoseconds, kelvin and (for raw
LAMMPS metal-unit logs) bar. Some processed thermodynamics explicitly convert
pressure to GPa; follow the array's producer metadata.

Use `TemporalLAMMPSBinaryTrajectory` for position-only trajectories and
`ShootingBinaryTrajectory` for position/velocity trajectories. Arrays can be
memory-mapped. Decode float16 positions to float32 and enforce the periodic
coordinate convention before neighbor calculations. Float16 changes exported
precision, not LAMMPS integration or restart precision.

Recorded float32-to-float16 minimum-image rounding error per coordinate is
max 0.0625 angstrom / RMS 0.01610 angstrom for the Ti source, max 0.125 /
RMS 0.03120 angstrom for the Ta baseline, and max 0.25 / RMS 0.07055 angstrom
for the large Ta 3.0 ns branch. These are conversion errors relative to the
float32 export, not measurements of integration or potential-model accuracy.

A million-atom frame is not a million independent statistical observations.
Split by the independent melt/source lineage where that is the scientific
contract. Do not combine different potentials, temperatures, horizons,
thermostats, sampling intervals or first-passage stopping rules without an
explicit analysis design. In particular, do not add the fixed-24-ps descendants
to their original nested parents as independent paths, or count Ti duplicates.

No datasets were deleted, recompressed, restarted or resubmitted during this
inventory. Detailed manifest snapshots and measured directory sizes are under
`technical/`.

## Complete simulation-directory size table

All paths below are under `/home/ids/vmorozov/simulations/`. Sizes include
retained history/diagnostics and do not imply completed simulation data.

| Directory | Allocated GiB |
|---|---:|
| [al_homogeneous_unseeded_2nn_meam_70304_12temps_390-500K_600ps_1ps_positions_velocities_6seeds_20260830](/home/ids/vmorozov/simulations/al_homogeneous_unseeded_2nn_meam_70304_12temps_390-500K_600ps_1ps_positions_velocities_6seeds_20260830) | 0.083 |
| [al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901](/home/ids/vmorozov/simulations/al_homogeneous_unseeded_2nn_meam_70304_400-500K_600ps_9independent_runs_20260901) | 11.849 |
| [al_independent_sources_recovery_20260909T203221Z](/home/ids/vmorozov/simulations/al_independent_sources_recovery_20260909T203221Z) | 0.000 |
| [al_meam_crystallization_100k_450K_20260911](/home/ids/vmorozov/simulations/al_meam_crystallization_100k_450K_20260911) | 23.823 |
| [al_meam_independent_sources_70304_400-500K_30perT_float16_20260902](/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_400-500K_30perT_float16_20260902) | 59.501 |
| [al_meam_independent_sources_70304_510-520K_30perT_float16_20260903](/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903) | 29.705 |
| [al_meam_independent_sources_70304_510-520K_30perT_float16_20260903_prepared_before_manifest_checksum](/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903_prepared_before_manifest_checksum) | 0.001 |
| [al_meam_nested_shooting_pilot_70304_400-500K_20260902](/home/ids/vmorozov/simulations/al_meam_nested_shooting_pilot_70304_400-500K_20260902) | 19.839 |
| [al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible](/home/ids/vmorozov/simulations/al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible) | 9.634 |
| [al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904](/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904) | 13.666 |
| [al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831](/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831) | 10.434 |
| [al_meam_position_shooting_70304_400-500K_48ps_2shots_local_followup_20260831](/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_2shots_local_followup_20260831) | 20.734 |
| [al_meam_position_shooting_70304_400-500K_48ps_40branches_local_20260901](/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_40branches_local_20260901) | 10.434 |
| [al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903](/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903) | 10.435 |
| [al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_protocol_flag_correction](/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_protocol_flag_correction) | 0.001 |
| [al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_without_legacy_window_spec](/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903_prepared_without_legacy_window_spec) | 0.001 |
| [al_meam_position_shooting_70304_400-500K_48ps_8shots_20260830](/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_8shots_20260830) | 82.533 |
| [al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904](/home/ids/vmorozov/simulations/al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904) | 2.636 |
| [al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903](/home/ids/vmorozov/simulations/al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903) | 4.188 |
| [al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903_invalid_endpoint_descriptor_preparation](/home/ids/vmorozov/simulations/al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903_invalid_endpoint_descriptor_preparation) | 0.068 |
| [interrupted_attempts](/home/ids/vmorozov/simulations/interrupted_attempts) | 0.067 |
| [local_overnight_lamedell11_20260905](/home/ids/vmorozov/simulations/local_overnight_lamedell11_20260905) | 0.000 |
| [local_overnight_lamedell11_20260905.startup_attempt_010736](/home/ids/vmorozov/simulations/local_overnight_lamedell11_20260905.startup_attempt_010736) | 0.000 |
| [nested_shooting_prepare_logs](/home/ids/vmorozov/simulations/nested_shooting_prepare_logs) | 0.000 |
| [nonshooting_float32_migration_20260901](/home/ids/vmorozov/simulations/nonshooting_float32_migration_20260901) | 0.000 |
| [restart_boundary_audit_20260905](/home/ids/vmorozov/simulations/restart_boundary_audit_20260905) | 0.290 |
| [shooting_float32_migration_20260901](/home/ids/vmorozov/simulations/shooting_float32_migration_20260901) | 0.014 |
| [superseded_preparation_al_meam_position_shooting_15ps_topup_manifest_metadata_20260904T0955Z](/home/ids/vmorozov/simulations/superseded_preparation_al_meam_position_shooting_15ps_topup_manifest_metadata_20260904T0955Z) | 0.001 |
| [superseded_preparation_al_meam_predictive_dynamics_fixed15_smoke_wave979929_20260904T0955Z](/home/ids/vmorozov/simulations/superseded_preparation_al_meam_predictive_dynamics_fixed15_smoke_wave979929_20260904T0955Z) | 0.000 |
| [ta_initial_model_1m_24ps_npt_20260905](/home/ids/vmorozov/simulations/ta_initial_model_1m_24ps_npt_20260905) | 1.826 |
| [ti_ta_crystallization_20260907](/home/ids/vmorozov/simulations/ti_ta_crystallization_20260907) | 104.182 |
