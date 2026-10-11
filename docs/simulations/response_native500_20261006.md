# Native-velocity response training through 0.5 ps

The completed continuation controls identified stronger growth suppression at
100 fs BAOAB damping than at 1000 fs. This follow-up retrains the existing
Q/QV response encoders with the corrected simulation recipe, preserving the
original 135 geometries and 90/15/30 source roles. Its main horizon is 500 fs;
100 fs is retained as a diagnostic from the same trajectory.

Full 70,304-atom periodic cells keep their original box fixed. Initial velocities
come from the exact same saved MD frame, without redraw/rescale/COM removal.
The thermostat uses each source's original temperature. BAOAB uses a 1 fs step
and 1000 fs damping time. FastAl force arithmetic remains float32; the user's
October 6 instruction changes integration, tangents, noise and observables to
float64. Original saved Q,V quantization is still recorded. Neural training
remains float32. No temperature/time metadata enters encoder or predictor.

One measured phase state per geometry gets four training noise shots, sixteen
validation values and sixteen held-out values plus eight disjoint held-out
response shots. Four Q/QV × value/response arms, three fitting seeds, 400 updates,
batch256 with replacement from 90 states, microbatch8. The new conditional
distance target and unchanged likelihood/response metrics are detailed in
[definitions](../metrics/response_velocity.md#native-velocity-source-temperature-float64-integration-revision-6-october-2026).
All64 clustering also uses actual same-atom velocities, with exact released
position correspondence and no omitted model-specific rows. Selection is
validation feature NLL; no crystal loss or clustering/AP checkpoint choice.

Use conda `pointnet-torch214`:

```bash
python -m src.research.response_velocity.queue prepare --config configs/simulation/response_native500_20261006.json
python -m src.research.response_velocity.queue prepare-clustering --config configs/simulation/response_native500_20261006.json
python -m src.research.response_velocity.queue launch --config configs/simulation/response_native500_20261006.json
```

The launch authenticates W&B online, freezes source/config/metric definitions,
requests four one-GPU helpers, and detaches one worker on each of the two
explicitly selected free local GPUs. Hardware-specific compiler caches retain
architecture provenance. A NEW nine-hour deadline includes scheduler wait;
Slurm helpers request eight hours within it. Own workers stop/checkpoint before
deadline; no automatic requeue or changes to other jobs/containing allocations.
Source temperatures and actual tensor precision are archived per query batch.
No new MEAM force validation is required. The existing student derivative and
symmetry startup check remains local, without a W&B run.

Output is `response_velocity/native500-f64-20261006-v1` under training storage.
IDS holds the sealed training bank and actual-velocity clustering cache. New
response batches use SCRATCH, then checksum publication to STORE under
`simulations/response-native500-f64-20261006-v1`; no full trajectory arrays are
exported by this response producer. Inputs, full-cell initial velocities and
their hashes remain in `technical/parents.pt` and `native-inputs.json`.
Registration: `response-native500-f64-20261006`. Existing response banks and
trained checkpoints are preserved separately.

`technical/data-complete.json` seals collection; each fit and cluster evaluation
has an individual completion receipt. `technical/complete.json` requires all
twelve fits AND their all64 evaluations. Online scientific training uses the
existing teshbek/PointCloudMaterials project with stable run IDs. This is the
original-cadence cohort only; dense/shooting, million-atom and relaxed populations
are not covered. Historical QV clustering with sampled velocities is a different
input protocol, so its score cannot be presented as a matched native-V comparison.

## Launch receipt (6 October 2026)

Launched at 10:53 Paris time, with an absolute deadline of **19:53 Paris**.
Output: `/store/PERSO/vmorozov/experiments/response_velocity/native500-f64-20261006-v1`.
The frozen executable code/config and ownership receipts are under its
`technical/` directory. The launch created Slurm jobs **1023779, 1023780,
1023781, 1023782**. At startup the first three ran on node61 (Blackwell),
node02 (A100), and node05 (A100); 1023782 remained queued.
Two detached workers use node59 allocation 1023686, GPUs 0/1, with respective
PID/process groups **20298/20299**. These receipts authorize management of
these workers, not cancellation of the containing allocation.

Preparation completed for all 135 native phase states and all 99,840 selected
all64 clustering rows (74,880 train / 24,960 held out). The existing student
derivative/symmetry check passed. The first archived training queries completed
with finite float64 values and responses, float64 states and tangent bases,
float32 force arithmetic, source-specific thermostat temperatures and 1000 fs
damping. W&B authentication passed; online model runs begin after the bank seals.
This is a startup record, not a report of completed training or improved scores.
