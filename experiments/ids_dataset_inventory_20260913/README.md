# IDS dataset inventory — 2026-09-13

Question: what datasets and derived products occupy `/home/ids/vmorozov`, how
much storage do they use, and which physical potentials/protocols and completed
sample counts do they represent?

Detailed findings: [readable inventory](../../output/ids_datasets/inventory-20260913/README.md).
Configuration and method: `technical/inventory_scope.json`. The report's
`technical/` directory stores allocated sizes, manifest/outcome snapshots,
cache schemas, and recorded comparisons between duplicate Ti trajectories.
No data were modified or simulations submitted.

The independent Al source pool has 126 completed 70304-atom histories out of
150 planned. Ti has a completed source and six unique completed branches,
plus six checksum-identical Slurm copies. Ta has six complete branches, one
with 1024000 atoms and five with 10000422. The new 100000-atom Al source remains
incomplete at about 748 ps; it is absent from the active MD queue. The related
Al/Mg EAM trajectories are on the repository filesystem, outside IDS totals.

Reproduce the storage measurements from the repository root in `pointnet`:

```bash
python scripts/experiment_registry.py storage --min-mib 1024
du -x -B1 --max-depth=1 /home/ids/vmorozov/simulations /home/ids/vmorozov/training-cache /home/ids/vmorozov/experiments /home/ids/vmorozov/data /home/ids/vmorozov/models /home/ids/vmorozov/analysis
du -sx -B1 /home/ids/vmorozov
squeue -r -u "$USER"
```

The maintained storage command covers local output roots, explicitly excluding
external symlinks. External `du` measurements are therefore recorded separately.
Read each physical campaign's `manifest.json`/`config.json`, canonical
`outcome.json`, and current binary `manifest.json` to regenerate protocol/count
summaries. Match only declared branch directories, excluding interrupted,
invalid-canary and rejected-extension outcomes. Compare all binary-array
descriptions between Ti and Ti_early_slurm to identify the six duplicate pairs.
The report defines all counts and size units; it exports no newly calculated
scientific metric CSVs.

This README and scope JSON are research records. Generated size/manifest
snapshots and the readable report are inventory outputs under
`output/ids_datasets/inventory-20260913/`. No new maintained tools were added.
