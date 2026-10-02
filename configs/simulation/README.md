# Simulation recipes

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
