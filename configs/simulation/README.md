# Simulation recipes

[`ni_births_lowT_010_20261012.json`](ni_births_lowT_010_20261012.json):
30 additional independent Ni melts, five each at 950/1000/1050/1100/1125/1150 K,
six GPU workers, native 0.10 ps Q/V and a 400-ps target-temperature hold cap;
[protocol, launch and dependent GPU birth search](../../docs/simulations/ni_births_lowT_010_20261012.md).

[`ni_relaxation_default.json`](ni_relaxation_default.json): current settings
preset for new Ni relaxations (user decision, October 11): fixed-cell FIRE,
maximum absolute force component 0.05 eV/angstrom, float64 minimizer states
and float32 GPU forces. Copy its explicit `relaxation` settings into a new
campaign's `relaxation` or response experiment's `quench_settings` before
freezing the plan. This is a settings preset, not a campaign launch config.
Historical dated recipes retain their original tolerances. The shared index
includes the tolerance in recipe identity, so cached minima retain their
actual settings. [Decision and comparison](../../docs/simulations/ni_relaxation_speed_20261011.md).

[`ni_birth_relaxation_20261008.json`](ni_birth_relaxation_20261008.json):
full-cell FastNi GPU/FIRE relaxation of -6 to +6 ps established-birth windows,
native 0.10 ps observations, two GPU workers and indexed native restarts;
[protocol and launch](../../docs/simulations/ni_birth_relaxation_20261008.md).
The evening speedup adds a separate four-GPU shared dispatch receipt, reusing
the original frozen numerical producer and completed cells.

[`ni_births_lowT_010_20261008.json`](ni_births_lowT_010_20261008.json):
36 independent Ni melts across 950/1000/1050/1100/1125/1150 K, four GPU workers,
the existing native 0.10 ps Q/V protocol and an automatic 12-CPU birth search;
[protocol and launch](../../docs/simulations/ni_births_lowT_010_20261008.md).

[`ni_births_lowT_hard_stop_400ps_20261008.json`](ni_births_lowT_hard_stop_400ps_20261008.json):
the user's live duration revision: maximum 400 ps at target temperature, 420 ps
with cooling. A receipt-owned controller publishes verified exact Q/V prefixes
and native checkpoints before ending source children; native overshoot is audited.
Earlier crystallinity stops and already-completed raw histories are preserved.

[`ni_lowT_rtx_handoff_20261008.json`](ni_lowT_rtx_handoff_20261008.json):
move the current Ni hold from a native 210-ps checkpoint to another GPU and
dispatch the final independent source separately. Preserve the session allocation,
original source identities, float64 NPT state and the 400-ps native hold limit.

[`al_birth_focused_v100_20261007.json`](al_birth_focused_v100_20261007.json):
33 unstarted V100 shoots transferred to continuous 0.10-ps Q/V from initialization,
an early cluster-growth trigger and 12-ps follow-up; original lane deadlines retained.
[Handoff and scientific limitations](../../docs/simulations/al_dense_shooting_v100_20261007.md#birth-focused-handoff-october-7-evening).

[`al_million_velocity_replay_20261007.json`](al_million_velocity_replay_20261007.json):
96-rank native replay of the million-atom Al production trajectory, preserving
the original 1 fs NPT settings and requiring historical agreement before export;
[workflow](../../docs/simulations/al_million_velocity_replay_20261007.md).

[`ni_births_010_20261007.json`](ni_births_010_20261007.json): twelve independent
Ni NPT melt/cooling/hold trajectories, exact0.10ps Q/V and a dependent birth search;
[workflow](../../docs/simulations/ni_births_010_20261007.md).

[`ni_births_010_20261007_extension01.json`](ni_births_010_20261007_extension01.json):
12 additional independent Ni melts under the same protocol, three additional
GPU workers, replicas 3–5, and a verified combined 24-source birth-table export;
[extension](../../docs/simulations/ni_births_010_20261007.md#twelve-additional-melts-and-three-more-gpus).

`ni_eam_gpu_20261007.json`: local numerical gates and hardware timings for the pinned pure-Ni Mendelev EAM/FS GPU implementation; no production precursor campaign.

[`al_dense_shooting_v100_20261007.json`](al_dense_shooting_v100_20261007.json):
40 prepared velocity reshoots, one V100, 24-hour combined validation/production
budget, exact 0.10 ps Q/V and inherited dense-campaign stopping. The subsequent
[`al_dense_shooting_v100_expand_20261007.json`](al_dense_shooting_v100_expand_20261007.json)
adds three V100 workers for the same prepared 40-branch queue;
[workflow](../../docs/simulations/al_dense_shooting_v100_20261007.md).

`response_native500_20261006.json`: actual saved MD velocities, source temperature, 1000 fs BAOAB damping, float64 integration, 100/500 fs response targets; [workflow](../../docs/simulations/response_native500_20261006.md).

`nucleus_dynamics_ablation_20261006.json` adds 384 paired fixed-box NVT and weaker-damping BAOAB continuations, with a nine-hour shared deadline; [workflow](../../docs/simulations/nucleus_dynamics_ablation_20261006.md).

[`response_velocity_cuda500_float32_20261004.json`](response_velocity_cuda500_float32_20261004.json): float32 states, tangents and response features with CUDA MEAM at 100/500 fs; full original cohort and 12 encoder fits. [Workflow](../../docs/simulations/response_velocity_20261004.md).

[`al_dense_gpu_20261004.json`](al_dense_gpu_20261004.json): five unstarted dense Al sources on CUDA MEAM with native isotropic NPT, original 0.1 ps cadence and halfway stopping. [`al_dense_gpu_transfer_20261004.json`](al_dense_gpu_transfer_20261004.json) transfers the last two unstarted sources to A100/L40S/A40. [Workflow](../../docs/simulations/al_dense_gpu_20261004.md).

[`al_dense_gpu_comparison_20261004.json`](al_dense_gpu_comparison_20261004.json): read-only physical audit of the completed five GPU trajectories against historical CPU parents and completed dense CPU descendants; preserves timestep, cadence and ancestry distinctions. [Results](../../output/meam_gpu/dense-cpu-comparison-20261004/index.html).

[Differentiable CUDA MEAM](meam_gpu_blackwell_20261004.json): response derivatives plus gated mixed-precision crystallization MD; [execution](../../docs/simulations/meam_gpu_blackwell_20261004.md).

[Periodic16k MEAM CPU/GPU comparison](meam_md_16k_20261004.json): shared liquid,450K,200ps,1fs, gated NVE fidelity and separate native LAMMPS throughput; [workflow](../../docs/simulations/meam_md_16k_20261004.md).

[Local80 response pilot](local_response_20261002.json) uses real fixed-Al64 parent neighborhoods, gated moving MLIP environments and a two-GPU training comparison; [workflow](../../docs/simulations/local_response_20261002.md).


`response_atlas_feasibility_20261001.json` is the bound Al256 response pilot recipe, exposed to the research workflow through `../response_atlas/feasibility_20261001.json`.


[Full liquid-information relaxation](liquid_full_relaxation_20260928.json): existing fixed-cohort cell coverage and a dependent full paired study.

All maintained simulation recipes belong here, never in the `configs/` root.
Training and analysis loaders for existing data belong in `../data/loaders/`.

| Recipe | Protocol |
| --- | --- |
| `al_main_half_stop_20261001.json` | Halfway/90%-peer stopping for 128 unstarted Al trajectories; preserves running sources, both cadences and velocities. [Protocol](../../docs/simulations/al_main_half_stop_20261001/README.md). |
| `al_main_010ps_20260927.json` | 150 main Al prepared-liquid descendants: 600 ps, exact 0.1 ps samples with velocities, 2 fs integration; [campaign](../../docs/simulations/al_main_010ps_20260927/README.md). |
| `al_main_010ps_continue_20261001.json` | Native restart recovery and two-source waves for the same Al campaign; [continuation](../../docs/simulations/al_main_010ps_20260927/README.md#october-1-continuation). |
| `al_main_001ps_20261001.json` | Two existing Al train ancestors, 400/520 K, with positions and velocities every 0.01 ps for 600 ps; [campaign](../../docs/simulations/al_main_001ps_20261001/README.md). |
| `al_birth_uniform_20260926.json` | 22 independently melted Al sources: 400–500 K in 10 K steps, two per temperature, early birth collection and continuous 0.15 ps observations; [campaign](../../docs/simulations/al_birth_uniform_20260926.md). |
| `al_crystallization.json` | Al MEAM source generation and position-conditioned branches. |
| `ti_crystallization.json` | Ti MEAM source generation and position-conditioned branches. |
| `ta_crystallization.json` | Ta EAM branches from recorded initial configurations. |
| `ta_shooting_20260926.json` | 24 replicated Ta position shots through the established elemental protocol; [campaign and execution](../../docs/simulations/ta_shooting_20260926.md). |
| `predictive_memory_precision.json` | Twelve fresh Al melt lineages, fixed 192 ps histories at 0.075 ps cadence, paired float32/float16 observations and preassigned sealed test sources. |
| `relaxed_tda_al.json` | Full-cell fixed-box FIRE targets for denser existing training windows and all completed Al shooting collections; no new MD. Use `python -m src.data.relaxed_targets prepare|run|status`; [details](../../docs/relaxed_tda_targets.md). |

Launch with `python scripts/run_lammps_campaign.py elemental run --config CONFIG
--run-name NAME`. New results use the machine's simulation storage root; completed
runs publish to STORE with verification. See [portable execution](../../docs/portability.md).

`potentials/ti_kavousi2019/` contains the Ti potential referenced through the dataset
catalog. `atomistic/al/producer_compatibility.json` remains at its original path:
it is a checkpoint-compatibility registry consumed by the atomistic producer, not
a launch recipe. Its bytes and producer implementation are unchanged.

Older Al MLIP, shooting, runtime-benchmark and campaign variants are in the
[verified config archive](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/configs/simulation/).
Their records and dataset inventory are indexed under
[docs/simulations](../../docs/simulations/README.md).

`al_crystallization_1m.json` is the requested million-atom source-only variant: 300 ps melt, up to 400 ps crystallization, no branches, with the melt restart retained. `source_limit_policy: save_state` records duration-limited completion separately from attaining the crystal-fraction threshold.

The million-atom recipe also enables `save_melt_trajectory: true`: the full melt is sampled at 0.1 ps, converted to verified float16, and kept alongside its native liquid restart.

The distinct memory-source recipe uses `run_lammps_campaign.py memory-sources
prepare --config CONFIG --run-name NAME`, then `memory-sources run-worker` inside
48-rank CPU allocations. It reuses the retained source family's NPT dynamics,
does no outcome-based stopping or measurement PTM screening, and publishes each
completed source to STORE. See the [production record](../../docs/simulations/predictive_memory_precision_20260917.md).

Response-atlas follow-up (2026-10-01): stronger toy controls and fixed-parent shot precision; see `experiments/response_atlas_20261001/FOLLOWUP.md` and `docs/response_atlas.md`.

The Al256 20/100-fs response oracle defaults to [float32 cuEquivariance and batch4](response_atlas_fast_default.json). The [new collection/training recipe](response_atlas_training_fast_20261002.json) uses separate paths, one training execution audit and disjoint test response seeds; [workflow](../../docs/response_atlas.md#default-for-new-al256-response-runs).

- `nucleus_recovery_20261004.json`: known isolated nuclei, original-NPT/source-T/response-T continuations and paired float32/float16 shooting states.

- `nucleus_controls_20261005.json`: matched dissolved/establishing/random-liquid controls, 12 ps per branch and a 16-hour campaign deadline.
- `nucleus_negative_extra_20261005.json`: 64 additional liquid ancestors and fresh shots of 12 failed embryos; 240 twelve-ps branches under a separate shared three-hour deadline.
- `nucleus_rate_calibration_20261005.json`: 64 present-selected embryos, matched source temperatures and common observation precision; 192 twelve-ps continuations with a new three-hour cap.
