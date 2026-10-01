# Main Al sources at exact 0.1 ps — 27 September 2026

User-authorized rerun of all **150 main independent Al preparations**, with
70,304 atoms each, 15 ps equilibration and 600 ps measurement, saving positions
and velocities every **0.1 ps** (6,001 measurement frames including endpoints).
The existing 0.75 ps trajectories are preserved.

Submitted CPU array **1010643** (four workers, first 24 sources) and dependent
preservation/successor job **1010644**. Successive waves cover all 150 sources.
The first wave stopped at the 48-hour limit on September 29. The user-authorized
[October 1 continuation](#october-1-continuation) recovers the interrupted sources
and uses smaller waves; the original launch and its frozen manifest are preserved.
See [submission receipt](submission.json), [input audit](preparation.json),
[historical timing evidence](runtime_baseline.csv) and [local validation](smoke.json).

[Recipe](../../../configs/simulation/al_main_010ps_20260927.json) ·
[Producer](../../../src/simulation/campaigns/dense_al.py) ·
[Original launch status](/store/PERSO/vmorozov/simulation-launches/al-main-010ps-20260927/status.json) ·
[Latest queue status](/store/PERSO/vmorozov/simulation-launches/al-main-010ps-plus001ps-20261001/status.json) ·
[Initial continuation status](/store/PERSO/vmorozov/simulation-launches/al-main-010ps-continue-20261001/status.json)

The user-authorized [two-source 0.01-ps addition](../al_main_001ps_20261001/README.md)
versions the execution queue to include 152 daughter trajectories from the same
150 ancestors. Controller **1017489** replaces the pending controller **1017390**;
the current workers retain their original code and inputs. The two additional
sources get priority after this wave, with one source per lane. The 150-source
0.1-ps contract and earlier frozen manifests are preserved.

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

## October 1 continuation

Twenty sources were complete and checksum-published; three were interrupted in
dynamics and 127 had not started. The continuation retains the same 150 source
IDs, melt ancestors, frozen roles and physical protocol. Its execution manifest
references the original manifest SHA256 rather than replacing the historical
launch. Operational recipe:
[al_main_010ps_continue_20261001.json](../../../configs/simulation/al_main_010ps_continue_20261001.json).

| Interrupted source | Native restart step | Measurement time | Retained frames | Remaining dynamics |
| --- | ---: | ---: | ---: | ---: |
| 876 | 285,000 | 570 ps | 5,701 | 30 ps |
| 881 | 97,500 | 195 ps | 1,951 | 405 ps |
| 882 | 217,500 | 435 ps | 4,351 | 165 ps |

The same LAMMPS executable reads the binary restarts. Inspection restored the
`ensemble` NPT fix state and checked all 70,304 identities, positions, velocities
and box coordinates against each saved checkpoint frame: all maximum differences
were **zero**. Production retains 48 MPI ranks and the original fix IDs, without
recreating velocities, resetting timesteps or repeating equilibration. LAMMPS
documents the requirement to restore fixes with their original IDs in
[read_restart](https://docs.lammps.org/read_restart.html).

Completed STORE sources are untouched. Interrupted working directories were
moved into `attempts/al-main-010ps-continue-20261001/` on SCRATCH, and their
previous verified STORE failure archives remain. New staging keeps the original
prepared inputs and selected native restart. After resumed dynamics, the producer
checks the checkpoint overlap, verifies the full partial-dump SHA256 against its
failure-publication receipt, and joins the retained prefix with the new tail,
discarding the duplicate checkpoint frame and superseded old tail. The maintained
converter must verify all 6,001 frames and float16 rounding before deleting the
new ASCII exports. Attempt-specific failure IDs preserve subsequent failures
without colliding with the September 29 archives.

```bash
python -m src.simulation.campaigns.dense_al prepare-continuation \
  --config configs/simulation/al_main_010ps_continue_20261001.json
python -m src.simulation.campaigns.dense_al submit \
  --launch /store/PERSO/vmorozov/simulation-launches/al-main-010ps-continue-20261001
```

These preparation/submission commands create a new immutable launch once; do not
repeat them against the already prepared continuation. Execution code and receipts
are in the continuation launch. The original SCRATCH source paths and final STORE
dataset IDs remain the same.

Each worker now handles **two** sources per wave, with a 48-hour limit, 48 ranks
and 64 GiB memory. The first wave uses three workers and a dependent controller;
CPU array **1017389** and controller **1017390** were submitted on October 1.
The [submission receipt](continuation_20261001/submission.json),
[recovery audit](continuation_20261001/preparation.json) and
[startup snapshot](continuation_20261001/startup.json) record the launch. All three
workers entered dynamics on nodecpu06/nodecpu07; the two first-lane recoveries
also matched their checkpoint frames exactly with the production 48 MPI ranks.
Later waves allow up to four. Quota checks use Slurm's actual `MaxSubmitJobsPU`
counter for the declared `normal` QoS. Expanded `squeue` array rows overcounted
the current submission usage: Slurm reported 26/30 while `squeue -r` displayed
30 rows. No unrelated jobs were canceled or repurposed.

The completed production runs took 3.26–9.86 hours for dynamics, with a **7.48-hour
median**, before conversion/publication. This supersedes the initial 5–8-day
estimate: roughly **12–15 more days with four continuously available workers**,
or **16–20 days with three**, plus scheduler delays and filesystem contention.
This is an estimate; the automatic controller stops for inspection if a new
source fails rather than silently discarding its dynamics.
