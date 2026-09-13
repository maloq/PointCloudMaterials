# Enlarged full-trajectory forecasts — September 11, 2026

**September 13 optimization:** both recovered fits continue from their saved
checkpoints in the same H100 allocations: autoregressive `990769.0` on
`nodesumo01`, direct `990770.4` on `node53`. See the
[GPU-resident restart record](GPU_RESIDENT_RESTART_20260913.md) for the current
execution records and replacement collection job `991021`. Older submissions
below are retained as historical provenance.

**September 12 recovery:** preparation exhausted the `/home/ids` quota after
35/125 shards; the dependent fits never started. See the
[failure diagnosis and storage recovery](RECOVERY_20260912.md) for the explicit
float16 cache protocol and replacement submission.

Question: does a larger autoregressive or direct forecaster learn useful future
local-structure dynamics with all currently completed canonical independent Al
MEAM sources, complete timelines, and augmented observed histories? Both methods
predict twelve 256-channel embeddings at +0.75, +1.5, …, +9 ps from nine observed
embeddings spanning −6 to 0 ps. Outputs also yield the separate means over
(0,3], (3,6], and (6,9] ps.

## Submitted execution

The original 240-epoch chunked fits were replaced on 2026-09-11 at the user's
request. Both methods now train for **32 epochs in one 24-hour H100 job each**.
The twelve old training jobs 989351–989362 and collection job 989363 were cancelled;
preparation job 989350 and all completed embedding shards were retained.

| Work | Slurm jobs | Dependencies |
| --- | --- | --- |
| Active-allocation preparation | 990061.1 on node27/A40 | Resumes the eight shards completed on node53 |
| Queued preparation continuation/verification | 989350 | Released on active preparation success, or after allocation 990061 ends |
| Autoregressive training | 990063 | `afterok:989350`; all 32 epochs and final evaluation |
| Direct full-trajectory training | 990064 | `afterok:989350`; all 32 epochs and final evaluation |
| Paired final comparison | 990065 | `afterok:990063:990064` |

The two training jobs can run concurrently after preparation, subject to GPU
availability. Each GPU job requests one H100, 16 CPUs, 192 GiB host RAM, and
24 hours. Collection requests 8 CPUs, 64 GiB and two hours. These are budgets,
not measured completion times. The measured warmed H100 updates imply roughly
5.8 hours of training GPU computation for autoregressive and 3.6 hours for direct
prediction over 32 epochs (1,807 updates/epoch). Validation, data loading, cache
verification and final evaluation are additional; a full-data epoch has not yet
been measured. The 24-hour request leaves substantial room for that overhead.
No scaled training results are claimed.

The initial active H100 run used
[`run_spec_active_prepare.json`](run_spec_active_prepare.json) and completed eight
shards before allocation 988041 expired at 15:40:48 Europe/Paris on September 11.
The current A40 continuation uses
[`run_spec_active_prepare_990061.json`](run_spec_active_prepare_990061.json), with
five CPUs and 16 GiB step memory within allocation 990061. That allocation ends
at 06:46:54 Europe/Paris on September 12. Its detached launch and logs are under
[`active-allocation-990061/`](../../output/embedding_forecast_20260911/scale/active-allocation-990061/).
Preparation job 989350 has `afterany:990061` until the active command succeeds,
at which point the generated launcher clears that dependency. It then verifies
the completed cache and releases both training jobs. An allocation timeout instead
leaves completed shards reusable and rebuilds the interrupted partial shard.
This avoids concurrent writers without waiting for allocation expiry after success.

The authoritative receipt is
[`queue_32epochs/submission.json`](../../output/embedding_forecast_20260911/scale/queue_32epochs/submission.json).
The [original receipt](../../output/embedding_forecast_20260911/scale/queue/submission.json)
records cancellation and the replacement IDs; its preparation specification and
frozen source remain in use for the existing cache protocol.
The queue freezes the complete `src/` tree, scientific configuration and source
selection. Each job uses the maintained experiment registry wrapper to record
its process, environment, configuration and logs. Repository edits do not change
the queued forecasting implementation.

## Data and configuration

[`scale_sources.json`](scale_sources.json) selects all 125 completed canonical
sources in the 400/450/500 K and 510/520 K independent-melt campaigns, using a
completed float16 trajectory manifest without filtering crystallization outcomes.
It records the 25 unfinished sources and excludes the duplicate archival campaign.
Whole-source splits and unique preparation seeds are retained.

| Split | Independent sources | Forecast windows |
| --- | ---: | ---: |
| Training | 74 | 59,181,056 |
| Validation | 24 | 19,193,856 |
| Test | 27 | 21,593,088 |
| Total | 125 | 99,968,000 |

At each of 400, 450, 500 and 510 K there are 18/6/6 train/validation/test sources.
The completed 520 K subset has 2/0/3: no independent 520 K validation source is
available in this frozen selection. Test sources overlap earlier exploratory
studies, so this is not a newly untouched benchmark.

Use 1,024 reproducibly sampled center atoms/source and all 801 frames over
0–600 ps. Existing preparation uses anchor frame 398 with a 292.5 ps margin,
which encodes exactly [0,801) after including 6 ps past and 9 ps future context.
There are 781 forecast anchors/center spanning 6–591 ps. This covers every
completed source and sampled time with a practical center budget, rather than
claiming all 70,304 atoms in each box.

The float32 embedding cache is approximately 105 GB (97.8 GiB), plus metadata,
at `/home/ids/vmorozov/training-cache/embedding-forecast-full-20260911`. Each
embedding is stored once. Four spawned workers gather overlapping windows from
memory maps into pinned batches; scaling uses unique training embeddings only.
The frozen snapshot MACE encoder and instantaneous 80-atom neighborhoods retain
the pilot target definition. Verified float16 simulation positions are unchanged.

[`scale.json`](scale.json) fixes both experiments:

| Setting | Autoregressive | Direct full trajectory |
| --- | --- | --- |
| Variant | `path_ar_large_aug` | `path_direct_large_aug` |
| History encoder | Four-layer GRU, width 512 | Four-layer GRU, width 512 |
| Decoder | Residual GRUCell with prediction feedback | Shared time-conditioned residual MLP |
| Baseline | History mean initializes rollout | History mean plus correction |
| Parameters | 9,989,376 | 8,014,080 |
| Loss | Full-rollout standardized frame MSE | Standardized frame MSE |
| Batch size | 32,768 | 32,768 |
| Epochs | 32 | 32 |

The pilot favored rollout over teacher forcing, with little benefit from auxiliary
bin/increment losses. Both enlarged methods therefore use MSE alone; bin and
increment errors remain monitored. This compares methods with different decoder
capacities. One seed, 20260911, is used for this expensive enlarged comparison.

Training-only augmentation adds Gaussian jitter with standard deviation 0.01
in standardized coordinates to past embeddings. Intermediate past frames are
dropped with probability 0.15 and replaced by the preceding available observation.
The oldest frame cannot be dropped; the anchor is neither jittered nor dropped.
Future targets and evaluation histories remain clean. Physical times are unchanged.

AdamW: peak LR 0.0003, weight decay 0.0001, three warmup epochs, cosine decay to
5% of peak LR, gradient clipping at 5. Model dropout is zero; augmentation and
weight decay supply regularization. Patience equals the 32-epoch budget, so
validation selects the checkpoint without shortening the planned run.

Each job trains from scratch, saves model/optimizer/scheduler/scaler/RNG recovery
state every epoch, and evaluates its best validation checkpoint after training.
There are no scheduled training continuations. Failed dependencies prevent
collection from running; an interruption still permits explicit recovery using
the saved scientific configuration, implementation and last epoch checkpoint.

Validation and history interventions aggregate metrics by source while reading
batches, without retaining every error row. Epoch logs retain losses, gradient
norm, LR, source-mean validation MSE and frame/bin curves. Final analysis saves
paired test-row errors, source/temperature results, baseline skill, change amplitude
and history interventions, as defined in the [main protocol](README.md).

## Reproduction, verification and artifacts

From the repository root in `pointnet`:

```bash
python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/scale.json \
  --stage queue \
  --queue-config experiments/embedding_forecast_20260911/scale_queue.json
```

This submission already exists; the command refuses a duplicate queue directory.
A new campaign requires explicitly changed output locations. The queue plan's
`preparation_job_id: "989350"` reuses existing preparation without scheduling a
duplicate writer. `epochs_per_invocation: 32` equals the entire epoch budget,
so each model produces one training job with no `--resume` continuation.
Native Slurm dependencies provide detachment from this session.

Before replacement submission, **28 forecasting tests passed**, including causal augmentation,
bit-exact augmented CPU training across a checkpoint boundary, streaming metric
agreement, independent Slurm chains and reuse of an existing preparation job for
one-job fits. Both enlarged models completed
three finite-loss/finite-gradient H100 updates at batch 32,768 using repeated real
pilot windows. Peak allocated memory: 38.7 GiB autoregressive, 28.4 GiB direct.
These are resource checks, not scientific enlarged-data results. The snapshot
encoder was also timed on 1,024 real neighborhoods.

Maintained implementation and tests: `src/training_methods/embedding_forecast/`
and `tests/test_embedding_forecast.py`. Versioned experiment records: this report,
`scale.json`, `scale_sources.json`, `scale_queue.json`, and the active-run specs. Generated job scripts,
frozen execution snapshots, logs and results: `output/embedding_forecast_20260911/scale/`.
Disposable resource diagnostics: `output/embedding_forecast_20260911/scale_preflight/`.
Per-fit outputs: `scale/runs/`; eventual paired report: `scale/runs/comparison.json`.
