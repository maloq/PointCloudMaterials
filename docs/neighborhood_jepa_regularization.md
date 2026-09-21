# Neighborhood JEPA regularization queue

Use `pointnet-torch214`. Configuration:
`configs/neighborhood_jepa/regularization_20260920/study.json`.
The derived order/reservoir collection is registered in `configs/datasets.json`.

```bash
python -m src.training_methods.neighborhood_jepa.regularization.data orders --config configs/neighborhood_jepa/regularization_20260920/study.json
python -m src.training_methods.neighborhood_jepa.regularization.data reservoir --config configs/neighborhood_jepa/regularization_20260920/study.json
python -m pytest tests/test_neighborhood_regularization.py -q
python -m src.training_methods.neighborhood_jepa.regularization.queue submit --config configs/neighborhood_jepa/regularization_20260920/study.json
```

The current submit recipe uses the two GPUs in each existing allocation 1000616
(node53) and 1000818(node59). It also submits four independent one-GPU, four-hour
jobs, eligible for RTX6000PRO, H100 or L40S, eight CPUs and 40 GB host memory each.
One worker claims one fit; new jobs can start wherever scheduling allows rather
than waiting for a particular two-GPU node. No other user's allocation is used.

Submission freezes source/configs/metric descriptions under technical/code.
There are 16 core fits, three development-selected continuations, 19 paired frozen
linear/MLP assays, and three paired baselines. Per-task flock claims prevent
simultaneous fits of the same model. Long runs wait for all comparisons within
their regularizer family; failed dependencies become explicitly blocked.

Runs checkpoint optimizer, objective state, sampled-step counter and RNG, and
stop ahead of allocation expiry. Another available worker resumes under the same
immutable identity. A failure is recorded and is not silently retried with altered
hyperparameters. To add a worker on another allocation, run the frozen module:

```bash
python -m src.training_methods.neighborhood_jepa.regularization.queue worker --config ABSOLUTE_FROZEN_CONFIG --lane UNIQUE_INTEGER
```

Set CUDA_VISIBLE_DEVICES to that worker's allocated GPU and execute from the
frozen code directory, preserving PCM_PROJECT_ROOT. The worker reads its Slurm
allocation deadline. Existing two-GPU workers use `coordinate` to spawn two lanes.
All launch IDs, task claims and logs live in technical/launches.json,
technical/lane-*.json and technical/runs/NAME/. W&B group:
`neighborhood-jepa-regularization-order-20260920`.

Order cache creation uses CPU only; random-reservoir preparation uses one GPU
once. No hardware benchmark runs inside training. BF16 applies to the encoder's
mixed-precision path, geometric operations retain their established FP32 rules,
objectives/readouts use FP32, and Epi ridge/logdet uses FP64. Gradient replay
computes one B512 regularizer and retains gradients through all 14 views while
processing 128 graphs at a time. Prefetched pinned data uses two CPU workers.

Metric definitions and hashes are exported; the user-disabled metric contract
checker remains disabled. Historical frozen code and tables are not rewritten.

## Performance releases for unstarted tasks

Training fetches group disk reads by shard and restore the original sampled order.
Selection resolves storage paths once, caches immutable phase labels, and encodes
only its required current/future centers. Microbatch size and scientific budgets
remain unchanged.

To move an active queue to an updated immutable snapshot, freeze a new `code/`
under a separate release directory. Run the CPU-only gate from the updated source:

```bash
python -m src.training_methods.neighborhood_jepa.regularization.release --config CONFIG --code NEW_FROZEN_CODE --receipt RELEASE_JSON
```

Keep that gate process alive until its heartbeat says `complete`. It locks only
unstarted legacy tasks, records their names and the original lanes, and leaves
running fits and historical snapshots untouched. Start one replacement worker per
original lane in that lane's existing GPU allocation, from `NEW_FROZEN_CODE`, with
`PCM_PROJECT_ROOT=NEW_FROZEN_CODE` and the same GPU binding:

```bash
python -m src.training_methods.neighborhood_jepa.regularization.queue worker --config CONFIG --lane ORIGINAL_LANE --release RELEASE_JSON
```

The replacement waits for its original fit to finish and release its lock before
using the GPU. It processes only the gate's unstarted tasks under separate locks.
If the gate heartbeat expires, it refuses further claims. Started fits retain
their original producer and resume identity. The original queue exits naturally
as task statuses become terminal. Pending, entirely unstarted Slurm campaigns can
instead be resubmitted with identical resource requests against the new snapshot;
retain the superseded job IDs and launch receipts.

## Multi-horizon JEPA addition

Configuration `configs/neighborhood_jepa/multihorizon_20260920/study.json` adds
three separate fits and their paired frozen probes. It submits three additional
single-GPU jobs; it does not edit the already frozen regularization campaign.
Prepare the registered future-view cache, then use the shared queue:

```bash
python -m src.training_methods.neighborhood_jepa.multihorizon.data --config configs/neighborhood_jepa/multihorizon_20260920/study.json
python -m pytest tests/test_neighborhood_multihorizon.py -q
python -m src.training_methods.neighborhood_jepa.regularization.queue submit --config configs/neighborhood_jepa/multihorizon_20260920/study.json
```

The prepared future graphs provide online embedding targets for3,6,9ps, with
explicit right-boundary masks. The additional horizons require three more encoder
views (17 total); conditional prediction heads and inference inputs are unchanged.
Both target and prediction branches receive gradients. Definitions live in
`docs/metrics/neighborhood_jepa_multihorizon.md`.

## GPU input caching and retained activations

New recipes include explicit `execution_profiles` for the compiled MACE path.
See [memory tiers, correctness checks and separate profiling](neighborhood_jepa_vram.md).
These settings preserve the 512-anchor loss and teacher gradients. They require
a new frozen release for an existing queue; changing the workspace recipe does
not alter the already-running or previously frozen fits.
