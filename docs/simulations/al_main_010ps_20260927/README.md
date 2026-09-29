# Main Al sources at exact 0.1 ps — 27 September 2026

User-authorized rerun of all **150 main independent Al preparations**, with
70,304 atoms each, 15 ps equilibration and 600 ps measurement, saving positions
and velocities every **0.1 ps** (6,001 measurement frames including endpoints).
The existing 0.75 ps trajectories are preserved.

Submitted CPU array **1010643** (four workers, first 24 sources) and dependent
preservation/successor job **1010644**. Successive waves cover all 150 sources.
See [submission receipt](submission.json), [input audit](preparation.json),
[historical timing evidence](runtime_baseline.csv) and [local validation](smoke.json).

[Recipe](../../../configs/simulation/al_main_010ps_20260927.json) ·
[Producer](../../../src/simulation/campaigns/dense_al.py) ·
[Durable live status](/store/PERSO/vmorozov/simulation-launches/al-main-010ps-20260927/status.json)

## Physical protocol and ancestry

Reuse each original native `prepared_liquid.lammps.data`, its velocity seed and
the pinned Lee2003 Al MEAM files. The validated 300 ps melt need not be repeated.
Use 2 fs integration, 7,500 equilibration steps, and 300,000 measurement steps.
Output every 50 steps; remove center-of-mass momentum every 150 steps to retain
the original physical 0.3 ps intervention interval. NPT remains zero pressure,
0.3 ps thermostat and 3 ps barostat. Save rolling native restarts every 15 ps,
plus final and original melt restarts. There is no outcome-based early stopping.

The old 3 fs integration cannot produce exactly 0.1 ps at constant integer-step
output intervals. This is a changed integration protocol and new chaotic dynamics,
not recovery of the missing historical frames. Every daughter keeps its parent
source ID, melt lineage, original manifest hash, preparation hashes and the frozen
`fixed_al64_v1` ancestry role: 90 train / 15 selection / 15 calibration / 30 test.
The 150 daughters and their 150 parents represent **150**, not 300, independent
melt ancestors. Existing benchmark data/releases are not overwritten. Future
comparisons must explicitly declare the new trajectory protocol; old labels cannot
be copied to the new paths. Temperature remains simulation/audit metadata.

## Execution and storage

Prepare with conda `pointnet-torch214`:

```bash
python -m src.simulation.campaigns.dense_al prepare \
  --config configs/simulation/al_main_010ps_20260927.json
python -m src.simulation.campaigns.dense_al submit \
  --launch /store/PERSO/vmorozov/simulation-launches/al-main-010ps-20260927
```

Each CPU worker uses 48 MPI ranks, 64 GiB RAM and at most six sources per wave,
with a 48-hour allocation. Up to four workers run concurrently; initial worker
count respects the current 30-submitted-job limit, reserving a slot for a
preservation/successor controller. After every wave, an `afterany` controller
verifies publication and submits the next wave. A failed wave is preserved to
STORE and stops automatic continuation for inspection, without overwriting or
silently restarting partial dynamics. Prepared source order is fixed by ancestor
source ID, never by observed crystallization outcomes.

SCRATCH: `/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/al-main-010ps-20260927/`.
Each completed source is checksum-published to
`/store/PERSO/vmorozov/simulations/al-main-010ps-20260927-sourceNNNN/`, with a link
from its SCRATCH staging directory. All code, hashes, environment, input manifest,
Slurm scripts and submission receipts remain under the durable launch directory.

Conversion uses `scripts/convert_trajectory.py dense-al RUN_DIR --delete-source`.
The shared shooting converter verifies float32 coordinates and velocities, then
checks every float16 rounding value, exact IDs/timesteps, unchanged float32 boxes
and array checksums. Quantization maximum/RMS and hashes are retained. ASCII and
the transient float32 reference are removed after verification; canonical float16
arrays and full-precision native preparation/restart files remain. This protocol
does not retain a paired float32 trajectory. Estimated canonical positions and
velocities: **759.4 GB** for the whole campaign, plus metadata and native artifacts.

## Runtime estimate and remote compute

All 150 original measurement logs contain timing evidence. Their 200,000-step
runs took 0.02258–0.04781 seconds per step (median 0.03378), about 1.3–2.7 hours.
The new 307,500 total steps imply about 1.9–4.1 hours per source before the added
cost of 7.5-times-denser output, conversion and publication. A practical allowance
is **3–4.5 hours per source**, about **5–8 days** with four lanes with no queue delay. Fewer continuously available lanes, filesystem contention or
queue delays extend this. This is an estimate, not a completion deadline; actual
first-source timings can refine it.

The direct SSH attempt to `vmorozov@lamedell11` failed at DNS resolution from
node61. The usual `enst.fr`, `telecom-paris.fr`, `telecom-paristech.fr` and
`lame.enst.fr` qualified variants also did not resolve. No computation has been
started there. The gateway attempt also failed SSH authentication. A reachable hostname/IP and any required jump host are pending.

Validation: the 108-atom infrastructure fixture completed the full 300,000-step
measurement and verified all 6,001 sampled frames through the maintained converter.
It is excluded from the scientific sources. All four production workers entered
dynamics successfully; [startup snapshot](startup.json) and [Slurm snapshot](slurm_startup.txt)
record this observation, not ongoing monitoring.

[September 29 same-parent comparison](../../../experiments/al_replay_20260929/README.md)
checks completed dense descendants against the original paths and crystallization.
