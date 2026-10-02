# Halfway stopping for unstarted Al sources

User authorized this queue update after the
[half-crystal analysis](../../../output/al_duration/half-crystal-20261001/README.md).
Apply it to **128 unstarted trajectories**: 126 main sources at 0.1 ps and the
two additions at 0.01 ps. Preserve all 21 completed trajectories and the three
sources already running at capture (880,881,882). Their records, native inputs,
restarts and dynamics remain unchanged. Both new cadences keep positions and
velocities; native integration remains 2 fs.

[Recipe](../../../configs/simulation/al_main_half_stop_20261001.json) ·
[Stopping producer](../../../src/simulation/campaigns/dense_al_half_stop.py) ·
[Queue versioning](../../../src/simulation/campaigns/dense_al_half_queue.py) ·
[Converter](../../../src/data/conversion/dense_al_half.py) ·
[Submission receipt](submission.json) · [Validation](validation.json).

Submitted successor controller **1017711** after active array **1017389**;
pending old controller **1017489** was canceled. All current array tasks were
still running at the [submission observation](startup.json).
[Preparation verification](preparation.json) checks identical protected records
and every native input checksum, all 128 handoffs and the complete frozen code
inventory. [Receipt hashes](provenance.json) identify the local copies.

For this exact pending source mix, replaying the declared rule on old parent
curves would save **48.8% of measurement duration** (47.6% including equilibration)
and approximately **370 GB** of positions/velocities, from 739 GB to 369 GB.
This [reference calculation](reference_savings.json) is an estimate, not a forecast:
new 2-fs dynamics have different transformation times, and PTM monitoring adds
wall-time overhead. Completed/running full-length payloads are excluded.

## Scientific stopping contract

Monitor whole-cell PTM at measurement zero and after each 15 ps of native
dynamics. Use RMSD cutoff 0.1 and FCC/HCP/BCC fraction over all 70,304 atoms.
The monitor consumes the established float32 representation of a native
full-precision snapshot, before float16 observation export. Two observations
separated by **exactly 15 ps**, both ≥50%, confirm the halfway event. Then retain
**6 ps** for the existing 3/6 ps prediction horizons, stopping at the declared
maximum if less time remains. No energy/volume plateau or complete solidification
is required. Record the final PTM fraction as well.

The reviewed historical 90%-peer calendars give these **maximum measurement
durations, including the prediction tail**:

| Temperature | Maximum |
| --- | ---: |
| 400 K | 600 ps — population 90% time not observed |
| 450 K | 291 ps |
| 500 K | 411 ps |
| 510 K | 411 ps |
| 520 K | 561 ps |

Add 15 ps equilibration, reusing each original 300 ps melted preparation and
velocity seed. These are fixed, declared reference-derived caps for this
version. They are not retuned from wall-clock completion and do not inherit an
individual parent's transition time. New chaotic paths can differ from the
reference. Temperature is a simulation condition and audit grouping, not an
encoder/predictor input.

Ending at a cap without confirmation is **right-censored**, not permanently
liquid. Record actual duration, first/confirming observations, final fraction,
reason, retained tail and tail truncation. Mask future targets without a complete
horizon. Keep coordinates, velocities, native final/rolling/melt restarts and
ancestor roles. The old fixed `al64_v1` evaluation release stays unchanged;
variable-length descendants must not silently replace its original observations.

## Execution and conversion

Use one native MPI LAMMPS process per source. Chunked `run` commands preserve
the NPT fix, velocities and integration state. Monitoring reads coordinates
without relaxing or perturbing them. The input uses [LAMMPS shell](https://docs.lammps.org/shell.html)
to call the frozen observer and then reads its decision. Remove the previous
decision include first, so callback failure produces a missing-include error
rather than stale decisions. Loop control occurs after the include; it does not
try to exit `run every`, which [LAMMPS documents as unsupported](https://docs.lammps.org/run.html).

`dense-al-half` checks the endpoint certificate and converts exactly 0 through
the actual last step at the recorded 0.1/0.01 ps cadence. Positions/velocities are
verified float16, boxes float32, identity/timeline exact. Compare every encoded
value with the established float32 consumer in bounded chunks; retain quantization,
source/array hashes and termination metadata. Reopen the complete binary before
deleting verified text. No complete float32 reference tree is allocated. Native
restart precision is preserved. Variable-length runs no longer require exactly
6,001 or 60,001 frames.

Four 108-atom infrastructure fixtures (halfway/cap × both cadences) check real
native two-rank execution and conversion, excluded from scientific counts.
Cold FCC confirms at 15 ps and ends at 21 ps after its tail. Hot disordered
fixtures reach a 21 ps cap without confirmation. A partial final interval does
not count as 15 ps confirmation. Verify exact timelines, float16 positions and
velocities, float32 boxes, IDs, endpoint provenance, source deletion and restarts.

## Queue transition

Keep all earlier launches, code/manifests and old prepared inputs. New inputs
live under the existing staging root's `al-main-half-stop-20261001/runs/`.
The new execution manifest also tracks protected records, unchanged. Its
controller waits for the existing worker array, then submits the remaining queue.
The two 0.01 ps additions retain priority and their own full worker lanes.

Unstarted second sources 884/885 were assigned to old workers. Their old
directories receive `queue_handoff` markers. After finishing current dynamics,
an old worker exits before starting that source. This intentional transfer is
not an MD failure. The successor collects only already-started wave records,
then schedules new directories. It uses `afterany`, so an exit at the handoff
boundary does not lose successful current sources. No running process is canceled.
The pending old controller is held during preparation and replaced only after
code, inputs and validation are frozen.

Two preparation attempts failed before activation: the first assumed the active
worker script was named `workers.sbatch` in the add-on launch, whereas its
producer retains it as `previous-workers.sbatch`; the retry encountered the
collection registration left by that attempt. Both prepared input copies and
receipts were preserved, the registration was reconciled with the first attempt's
actual location, and both conditions now have explicit preflight checks. No
scientific dynamics ran in these attempts and no current worker was interrupted.
The validated native stopping/conversion implementation was unchanged.

Completed variable-length trajectories publish to the existing STORE source
IDs with verified copying and SCRATCH aliases. Their own metadata identify the
new protocol and actual endpoint. Old full-length trajectories/failure archives
remain intact. The manifest preserves 150 ancestor IDs plus two additional
observations: 150 independent melts, not 152.

From conda `pointnet-torch214`:

```bash
python -m src.simulation.campaigns.dense_al prepare-half-stop \
  --config configs/simulation/al_main_half_stop_20261001.json
```

The command refuses to overwrite an existing launch or change started sources.
A changed availability capture needs review; preserve immutable preparations.
