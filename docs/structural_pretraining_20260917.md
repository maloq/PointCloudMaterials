# Structural pretraining: execution and reproduction

Three requested fits launched detached on September 17, 2026 at 21:36 UTC.
Scientific inputs, objectives and interpretation limits are in the
[protocol](../experiments/structural_pretraining_20260917/README.md).
Only these three fits are submitted; the earlier eight-fit proposal is superseded.

## Running jobs

| Run | GPU / Slurm allocation | Detached srun PID | Deadline, UTC |
| --- | --- | ---: | --- |
| MACE–VICReg | H100 NVL, node53 / 995957 | 1052758 | September 18, 03:44 |
| GATr–JEPA, three causal frames | H100 NVL, node53 / 995957 | 1052759 | September 18, 03:44 |
| GATr–VICReg | RTX PRO 6000, node58 / 997207 | 1052760 | September 18, 00:23 |

Each requests 4,096 updates with seed 20260919. The H100 fits share its GPU.
Training stops two minutes before its configured deadline and writes resumable
state, leaving additional time before the allocation expires. Closing this
conversation does not stop the detached processes. No new H200 job is submitted;
no callable H200 host was supplied in this session.

Run directories:

- [MACE–VICReg](../output/structural_pretraining/mace-vicreg-20260917/technical/)
- [GATr–VICReg](../output/structural_pretraining/gatr-vicreg-20260917/technical/)
- [GATr–JEPA](../output/structural_pretraining/gatr-lejepa-history3-20260917/technical/)

Each contains `launch.json` with the exact command, allocation, PID, config hash
and data identity; `identity.json` binds implementation/dependency/data versions.
`status.json` updates at checkpoints, while `updates.jsonl` records every update.
`train.log` contains startup and any failures. Inspect Slurm and these files to
establish current state; this document records the launch, not a live monitor.
The [launch health receipt](../output/structural_pretraining/broad-250k-20260917/technical/launch-health.json)
records verified progress. Its sibling `launch-source.tar.gz` preserves the
Python sources, recipes and metric contract as launched; file hashes were
checked against all three trainers' startup identities.

## Prepared release and registry

Registry ID: `structural-neighbors-250k-20260917`. Open the
[dataset registry](../DATASETS.md) for locations and potential provenance.
Its portable cache path is
`${storage:cache}/structural_pretraining/broad-250k-v2-20260917`.

The completed release has 1,010 shards, 25.735 GiB of binary arrays, 250,000
training records and 480 selection records. There are 249,966 unique training
source/frame/center identities and 34 repeated records; these small repetitions
remain declared rather than changing the frozen schedule. 62,350 training
anchors have instantaneous TDA. The schedule uses 261 source records from the
ancestry-filtered pool. Actual successor intervals range from 0.03 to 0.75 ps.

The [release audit](../output/structural_pretraining/broad-250k-20260917/technical/release-audit.json)
checks target finiteness, mappings, masks, timeline ordering, shard identity,
normalization and group sizes. `plan.json` preserves source manifests, ancestry,
calibration centers, target producer hashes and every extraction task.
`manifest.json` contains finalized normalization and per-array SHA256 receipts.

The first preparation attempt rejected legitimate nonuniform saved intervals.
It was stopped and preserved at `broad-250k-20260917`; v2 accepts strictly
increasing recorded times and retains each actual offset. The same calibration
is reused only after exact source/seed comparison. No simulation or source
trajectory was modified. The independent relaxed-TDA campaign is separate.

## Commands and dependencies

Run from the repository root in conda `pointnet`. Existing CUDA dependencies
include MACE/cuEquivariance and GATr. Install the pinned author SIGReg package
from [requirements](../environments/requirements-structural-pretraining.txt)
in an environment that already contains those dependencies:

```bash
python -m pip install --no-deps -r environments/requirements-structural-pretraining.txt
```

Preparation is resumable at shard boundaries and refuses mixed producer hashes:

```bash
python -m src.data.structural_pretraining.prepare --config configs/structural_pretraining/data.json --workers 6
```

The frozen preparation recipe references the first attempt's calibration plan.
For a fresh machine, transfer the complete prepared release to the cache root
rather than reconstructing a partially copied calibration chain. Source arrays
are not needed to train from the finalized release.

The three training commands are:

```bash
python -m src.training_methods.structural_pretraining.train --config configs/structural_pretraining/mace_vicreg.json
python -m src.training_methods.structural_pretraining.train --config configs/structural_pretraining/gatr_vicreg.json
python -m src.training_methods.structural_pretraining.train --config configs/structural_pretraining/gatr_lejepa.json
```

They are launched inside the allocations using the exact `srun --overlap`
commands recorded in `launch.json`, with stdin detached and stdout/stderr
redirected to each run's log. Their stored deadlines are specific to this launch;
set a new deadline in a copied config before a later continuation.

Add `--resume` to the matching training command to continue `technical/last.pt`.
Resume checks model/data/code/dependency identity; only output location and
deadline may change without changing scientific identity. Copy the entire run
folder, not just the encoder, for exact resume. Checkpoints include optimizer,
CPU/CUDA RNG and SIGReg step state. Do not start a duplicate worker in a run:
the trainer holds an exclusive filesystem lock.

## Outputs and validation

- `last.pt`: exact resume state, step 1 and every 64 updates plus shutdown.
- `best.pt`: best fifteen-source selection checkpoint, evaluated every 256 updates.
- `encoder.pt`: the selected encoder only, including species vocabulary, input
  frame count, fitted scales and scientific identity.
- `selection_predictions.npz`: selection states and per-block target errors.
- `tables/selection.csv` and `tables/METRICS.md`: defined source-balanced
  physical/TDA selection errors with metric implementation hashes.

The requested budget is 4,096 updates. Selection runs also at that final update;
if stopped between scheduled validations, the export is the last best validated
checkpoint and `last.pt` is the exact final training state. The single seed and
absence of physical-only controls limit causal claims about the objectives.

Thirteen targeted tests passed, covering real geometry primitives, periodic
charts, symmetry, packed batches, causal isolation, species, full-batch gradient
cache equivalence and collapse detection. All three trainers completed real-data
integration runs with batch 128 and microbatch 32, selection, metric exports and
checkpoints; the JEPA checkpoint also passed CLI resume. Diagnostic artifacts
are under `output/structural_pretraining/preflight-20260917/` and are not the
scientific training release or initialization.

CPU prefetching, a bounded graph cache and gradient recomputation keep the
statistical batch at 128 while limiting activation memory. No hardware benchmark
is embedded in training. Frozen probes, held-out analysis and fine-tuning are
left for the user's next analysis request.
