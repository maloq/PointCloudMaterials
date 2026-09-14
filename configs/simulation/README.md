# Simulation recipes

All maintained simulation recipes belong here, never in the `configs/` root.
Training and analysis loaders for existing data belong in `../data/loaders/`.

| Recipe | Protocol |
| --- | --- |
| `al_crystallization.json` | Al MEAM source generation and position-conditioned branches. |
| `ti_crystallization.json` | Ti MEAM source generation and position-conditioned branches. |
| `ta_crystallization.json` | Ta EAM branches from recorded initial configurations. |

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
[docs/simulations](../../docs/simulations/README.md). Regression-only configurations
live in `tests/fixtures/simulation/`; do not use them to launch production runs.

`al_crystallization_1m.json` is the requested million-atom source-only variant: 300 ps melt, up to 400 ps crystallization, no branches, with the melt restart retained. `source_limit_policy: save_state` records duration-limited completion separately from attaining the crystal-fraction threshold.

The million-atom recipe also enables `save_melt_trajectory: true`: the full melt is sampled at 0.1 ps, converted to verified float16, and kept alongside its native liquid restart.
