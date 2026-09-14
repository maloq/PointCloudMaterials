# Forecast execution performance — September 13, 2026

Low GPU utilization in the spatial/mixture study came mainly from validation input
gathering and transfers. Backpropagation reached 98% GPU utilization. Keeping both
the training and spatial validation splits on the H100 throughout training had
previously exhausted device memory, so the first spatial fits used host validation.
Their validation passes took about 75–98 seconds per epoch.

The `staged_device` runtime keeps training embeddings and pooled spatial context on
the GPU, and validation tensors in host RAM between epochs. At validation it copies
the complete split to the GPU, evaluates it, restores the host tensors, and releases
the GPU copy before the next training batch. The 74-source training inputs occupy
about 57.9 GiB; the 24-source validation inputs occupy about 18.8 GiB. They coexist
only during inference, which needs less temporary memory than backpropagation.

This preserves batch size 8192, all 7,350,272 training windows, 2,383,872 validation
windows, source weighting, augmentation, sample order, precision, optimizer updates
and checkpoint selection. Spatial means are copied after pooling, without another
rounding step. Model/loss/data/sampler/metric/augmentation implementations match the
previous source snapshot exactly. Existing completed research artifacts remain intact.

The optional runtime recipe is
[`staged-validation.json`](../configs/embedding_forecast/staged-validation.json).
Use the existing training command with `--runtime-config` pointing to that recipe.
It requires the resident loader and enough GPU memory for both splits during
inference. Transfer failure restores the original host tensors and raises the error.

## Validation and deployment evidence

Diagnostics, benchmark source, immutable execution source and replacement tracking
records are under
[`technical/staged-validation`](../output/embedding_forecast/context-space-mixture-20260913/technical/staged-validation/).
The full-split benchmark loads the selected trained spatial K=4 checkpoint and
compares every source/sample mean score and per-step curve, plus exact raw batches.
Both timings include their real transfer costs; staged timing also includes release.
The benchmark shares the H100 with an active nonspatial training process, so production
epoch logs are the stronger evidence for total runtime improvement.

The full 24-source trained-checkpoint benchmark measured 129.98 seconds for host
validation and 68.56 seconds for staged validation: 1.90 times faster with zero
difference in every checked metric. Whole-split transfer took 7.70 seconds under
concurrent GPU load. Peak memory for the benchmark process was 23.92 GiB, including
the validation inputs and model inference. The earlier three-source preflight used
random weights of the same architecture and is retained as preliminary evidence only.

The first production epoch on node53 measured **109.32 seconds training + 30.17
seconds validation = 139.49 seconds**. The previous spatial K=4 fit on the same node
had median phase times of 109.90 + 97.53 = 207.44 seconds (excluding its first epoch).
Validation is 3.23 times faster and epoch time is 32.8% lower, giving 1.49 times the
epoch throughput. This compares the two fitted seeds of the same model, batch and
data protocol; it excludes initialization, checkpoint writes and final test exports.
The next epoch passed 500 training batches after releasing the validation copy.
GPU samples during validation reached 91–95% utilization; observed peak used device
memory was 89,255 MiB on the 95,830 MiB H100. Full timings and sampled phase metadata
are in `production-timing.json` and `production-device-samples.jsonl` beside the benchmark.

The regression suite passed 51 tests; an additional CUDA training test passed with
identical results for device, host and staged validation. Tests cover restoration
after interrupted validation, spatial batch equality, unchanged training/selection,
allocation deadlines, duplicate-controller rejection and custom progress locations.
Metric documentation contracts were checked; historical exported contracts are frozen.

Replacement allocation recipes:

- [`node53-staged-allocation-20260913.json`](../configs/embedding_forecast/node53-staged-allocation-20260913.json): the last original spatial K=4 fit, seed 20260914.
- [`short-history-staged-allocation-20260913.json`](../configs/embedding_forecast/short-history-staged-allocation-20260913.json): all fourteen queued shorter-history fits and their existing physical analyses.

The active original nonspatial fits continue through completion. Their validation
already resides on the GPU. Replacement controllers retain the original scientific
configs, source snapshots, dependencies and output locations. Predecessor controller
status and interrupted waiting-attempt records are preserved separately; external job
files and submitted Slurm scripts are not edited.

Deployment at 19:47 UTC: the last original spatial K=4 fit started under the new
controller in allocation 991149 on node53. Its predecessor was retired only after
the active nonspatial fit completed and exited. All fourteen shorter-history fits
are queued with the optimized runtime in allocation 990987 on nodesumo01, preserving
their original prerequisite fits and local physical analyses. The old waiting
controller's tracked attempt records an intentional interruption before any fit began.

## Broader spatial context in allocation 990987

The September 13 spatial-context extension queues twelve fits on the existing
nodesumo01 H100 allocation, after its short-history controller finishes. The
[scientific protocol](../experiments/forecast_spatial_mixture_20260913/SPATIAL_CONTEXT_SCALE.md)
compares 32, 128 and 512 cached neighbors with 3/12 ps histories and two seeds.
The active recipes are
[cache preparation](../configs/embedding_forecast/spatial-context/cache-allocation-20260913.json)
and [training plus analysis](../configs/embedding_forecast/spatial-context/training-allocation-20260913.json).
They use the existing allocation runner and experiment registry; the source and
recipes used by the detached processes are frozen under the new run's `technical/`.

CPU preparation runs alongside the current GPU training, with six processes and
one Torch thread each. It persists same-frame pooled embeddings on IDS so broad
neighbor gathers happen once. Each cache reports queued/preparing/complete/failed;
training waits for its own completed context cache and fails on a failed dependency.
This keeps GPU batch computation and resident tensor shapes equal to the existing
eight-neighbor model. Every artifact is hashed and checked on load.

The three caches require approximately 296 GB together, including retained neighbor
indices, radii and float16 pooled embeddings. IDS reported approximately 26 TB free
on its shared filesystem; this is not a personal quota measurement. WORK reported
about 11 GB free at preparation time. Checkpoints and local inference arrays use
the configured training-storage location on STORE, reached through symlinks from
`output/embedding_forecast/spatial-context-scale-20260913`. Readable reports and logs
remain in the regular analysis output. No existing files are removed.

The Slurm allocation ends at **2026-09-14 08:19:56 UTC**. The new controller uses
08:09:56 as its deadline, requires 35 minutes remaining before starting a 3 ps fit
and 45 minutes before a 12 ps fit, and checks smaller analysis budgets separately.
Observed optimized epochs took about 98 seconds for 3 ps history and 143 seconds
for 12 ps. Allowing initialization, final test exports and local assays gives about
6 hours 42 minutes for the twelve-fit sweep and its reports. If the preceding queue
finishes around 00:40, this suggests completion around 07:22 UTC; broader-cache
checksum I/O makes that an estimate. Guards reject starts with insufficient time;
they do not interrupt a command already running or submit another Slurm job.

Queue status, per-step logs, launch receipts and the timing estimate are under
[`spatial-context-scale-20260913/technical`](../output/embedding_forecast/spatial-context-scale-20260913/technical/).
Paired reports are produced after each complete context cohort and once for the
full sweep. CPU regression checks passed for stored/direct pooling, causal frame
selection, 8/32/128/512-neighbor means, failed-cache status, allocation execution
and metric documentation. CUDA tests were not run concurrently with active training.

The first cache attempt failed before writing source arrays because the allocation
bootstrap did not register the executed module as `__main__` for multiprocessing
spawn. The launcher now uses `runpy.run_module(..., alter_sys=True)`; a subprocess
regression test executes a spawned worker through the actual allocation runner.
All five allocation tests passed. Both queues were relaunched at 23:33 UTC with
new `source-v2`/`recipes-v2` snapshots and fresh IDS cache directories. Their active
status files are `technical/cache-allocation-status-v2.json` and
`technical/allocation-status-v2.json`. The first attempt's source, recipes, logs,
failure status and tracking records remain intact.

## Individual-neighbor attention on node51 — September 14

The [attention protocol](../experiments/forecast_spatial_attention_20260914/README.md)
uses allocation 991772 on node51's L40S (46,068 MiB reported device memory), with
12 CPUs and 80 GiB host memory. The detached controller launched at 09:04:37 UTC.
Its [allocation recipe](../configs/embedding_forecast/spatial-attention/allocation-20260914.json)
runs the two 12 ps fits and a paired report first, then the two 3 ps fits and the
full report if sufficient time remains. It uses the existing allocation, with no
new Slurm submission. Job end is 15:34:45 UTC; the queue deadline is 15:24:45.

All 125 center-geometry sidecars were prepared on CPU. Geometry remains on IDS,
alongside the existing neighbor-index and embedding caches. Model checkpoints and
local assay arrays use STORE; analysis output and logs use WORK. The new source,
recipes, launch receipt and execution record are retained under
[`spatial-attention-20260914/technical`](../output/embedding_forecast/spatial-attention-20260914/technical/).
Geometry preparation has its own earlier frozen source and tracked execution.

The full-data preflight loaded **33.240 GiB** of training inputs on the GPU and
**10.781 GiB** of validation inputs in host memory. Individual-neighbor tensors
are gathered only for each 1,024-sample microbatch. A batch of 8,192 still performs
one optimizer update, with sample-weighted accumulation and one clipping operation.
The bfloat16 attention branch and float32 GRU/decoder fit at a measured peak
**37.079 GiB allocated**. Three warmed optimizer updates took 0.3993, 0.3979 and
0.3985 seconds; there are 898 updates per epoch. The first update took 1.776 seconds.

Training inputs are swapped to host for validation, allowing the full validation
split to run on the GPU. The preflight's swap, three validation forward/likelihood
batches and restoration took 41.75 seconds; the individual forward/likelihood
batches took approximately 0.022–0.023 seconds. Those timings exclude the additional
CRPS/coverage/source-aggregation work of complete production validation. They are
not a measured whole-epoch duration. Production logs provide that measurement.

The 12 ps starts require at least 135 minutes remaining before the queue deadline.
The first 3 ps start requires 140 minutes for its paired cohort; the second requires
70 minutes. Physical assays and reports have separate guards. A guard rejects a
new start if insufficient allocation time remains; it does not stop a running fit
or silently submit another job. Completed 12 ps results remain independently usable.

Validation: 63 CPU regression tests passed (4 CUDA cases skipped), then three
CPU/CUDA geometry, swap-restoration and bfloat16-backward checks passed. Seven
focused integration/metric-layout checks passed after releasing the final batch's
resident-store reference before validation/test reconstruction. Checks cover paired
source/frame identities, periodic wrapping, neighbor permutation and joint-rotation
invariance, sensitivity to angular arrangement, gradient accumulation with a partial
tail, local readout parity, interrupted validation restoration and end-to-end exports.
