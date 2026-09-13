# Simulations and datasets

Simulation production and inventories belong here. `experiments/` is reserved for
scientific comparisons using the data.

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
