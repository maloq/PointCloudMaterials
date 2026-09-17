# Predictive-memory pilot workflow

Current optimized training uses one batched encoder implementation. For a fresh
run use `configs/predictive_memory/batched.json`; historical runs belong to their
original commit. Checkpoint format 2 supports exact continuation of new runs and
deliberately rejects earlier checkpoint formats.

```bash
python -m src.training_methods.predictive_memory.train --config configs/predictive_memory/batched.json --history-ps 12 --velocity
```

`training.batch_size` counts independent windows per optimizer step.
`micro_batch_size` controls activation memory; gradients are averaged over the
full batch and clipped once, including uneven final microbatches.
`evaluation_batch_size` controls validation/export packing independently.
`encoder.frame_chunk` limits total spatial frames processed in a packed kernel.
Temporal attention, identities and pooling never cross windows.

The fresh batched recipe enables `encoder.mace_backend: cueq`. MACE's supported
cuEquivariance wrappers replace spatial linear maps, channel-wise tensor products,
fully connected tensor products and many-body symmetric contractions. Custom
motion and temporal layers keep their existing e3nn operations. We use `mul_ir`
layout and `O3_e3nn` conventions so invariant contractions, parity and the
scalar/vector/tensor channel ordering remain consistent. Convolution fusion is
disabled; cuEquivariance's symmetric and channel-wise product kernels are active.
See [MACE's backend documentation](https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html).

The pinned GPU requirements already include MACE 0.3.16 and cuEquivariance 0.10.0
with its CUDA 12 extension. Missing cuEquivariance raises an error; there is no
automatic e3nn fallback. `technical/runtime.json` records the backend, layout and
package versions. Set `encoder.mace_backend: e3nn` only for a deliberate reference
run. Backend changes require fresh fits: resume requires identical configuration
and optimizer state, and CuEq symmetric-product parameters use a different basis.

On another H100/H200 server, use the updated source, pinned
`environments/requirements-predictive-memory-gpu.txt` and the batched recipe with
its existing cache. No new simulation or data transfer is required if that cache
is already present. Choose memory limits before fitting; the backend does not
run a hardware benchmark. Active local-predictability fits retain their e3nn
optimizer state. The queued raw-state fit uses CuEq from initialization; frozen
readouts map copies of completed encoder weights into CuEq and verify outputs
before extraction. Original checkpoints and the cohort's hash-locked native
constructor remain unchanged. AdamW moments are not migrated between the
symmetric-product bases.

CuEq validation on the allocated RTX PRO 6000 used mapped e3nn/CuEq weights.
On actual Al snapshots and 12 ps histories, the maximum absolute output error
was 4.8e-7 and the maximum parameter-gradient error was 9.6e-7, including the
many-body weight-basis projection. CUDA profiling captured forward and backward
`segmented_polynomial` kernels. Alternating complete training updates at effective
batch 4/microbatch 2 measured about 1.95x speedup for 12 ps histories; snapshot
timings varied substantially between trials. These shared-GPU measurements are
provisional. Raw checks and kernel names are in
`output/predictive_memory/cueq-validation-20260917/technical/real_validation.json`.
A separate four-update CuEq smoke fit completed validation, checkpointing and
export of all 450 windows under `output/predictive_memory/cueq-smoke-20260917`.

The example uses effective batch 8, microbatch 2, evaluation batch 4 and up to
8 GiB of immutable observations cached on the GPU. `runtime.cache_scope: train`
prevents validation from evicting training inputs; `all` permits every split.
The limit excludes model parameters, activations and temporary allocations.
Tune these explicit limits for the available GPU before starting a run; there
is no automatic hardware benchmark. Learned atom features are never cached.
Targets and train-normalized targets remain resident on the device.

The old pilot's 3,000 updates at batch 1 are not the same budget as 3,000 updates
at batch 8. Logs and metric tables record both updates and sampled windows, and
the collector checks matching batch/window budgets. Existing experiment configs
and results retain their original settings; this recipe launches a fresh output.

Validation on the allocated RTX PRO 6000 checked independent windows against the
pre-change encoder on actual Al snapshots and 12 ps histories: maximum absolute
output difference below 6e-7 and parameter-gradient difference below 1e-6.
The 45 focused tests cover ragged batches, isolation, 48 ps histories, positions-only
and repeated-frame controls, checkpointing, uneven microbatches, cache bounds,
sample budgets and resume. CUDA reductions permit ordinary last-bit variation
after restoring identical checkpoint tensors; CPU continuation is checked exactly.

A four-update real-data smoke run (batch 3, microbatch 2) completed validation,
checkpoints, metric export and all 450 train/validation/test windows. Initial
interleaved timings at effective batch 4 measured microbatch 2 versus 1: about
1.49x throughput for 12 ps histories and 0.98x for snapshots. The GPU was shared
with active research runs, so these are illustrative timings. Raw numerical and
timing evidence lives in
`output/predictive_memory/batching-validation-20260917/technical/real_validation.json`;
the full smoke output is `output/predictive_memory/batched-smoke-20260917`.

**17 September stop update:** the user stopped the active follow-up. Ten of its
16 planned fits completed; six never started. The last completed fit (original
loss, seed 20260918, H12) had finished training and evaluation before an unrelated
hardware-benchmark metric-contract mismatch blocked table export. The saved
per-window scores, checkpoint budget/configuration and unchanged predictive
metric implementation were verified; table export was completed on CPU without
retraining. Its `technical/export-recovery.json` preserves the original status
and audit. The failed allocation controller was not restarted. See the
[updated results](../output/predictive_memory/research-summary-20260917-stopped/RESULTS.md).

Use conda `pointnet` from the repository root. The module entry points reuse the
existing experiment tracker and allocation queue; no new simulations are needed.

```bash
python -m src.data.predictive_memory.prepare --config configs/predictive_memory/pilot.json
python -m src.training_methods.predictive_memory.train --config configs/predictive_memory/pilot.json --history-ps 48 --velocity
```

The first command audits and derives a separate immutable cache on the configured
cache storage root. It refuses to overwrite a completed release. `source_plan`
references the previous cache's observation inventory and inherited splits;
its outcome-bearing tensor shards are not loaded. Source paths are resolved by
the existing machine configuration. Release artifacts include dataset manifest,
lineages parquet, source splits, precision limitations and a data card. The first
release also includes `DATA_CARD_ERRATA.md` clarifying center selection.

Fit variations use `--history-ps {0,12,48}`, optional `--velocity`, and optional
`--repeat-anchor`. `--resume` requires the exact scientific configuration and
release checksum. Latest checkpoints include optimizer, sampling and Torch/CUDA
random states; best checkpoints are selected only by physical validation NLL.
`--deadline-utc` checkpoints and exits before the allocation reserve. Inspect
`technical/status.json`, `training.jsonl`, and `best.pt`/`latest.pt` in each fit.

The H100 recipe batches all observed frames into disjoint spatial graphs. The
equivariant temporal value sum is algebraically factored to avoid the expanded
time-by-time-by-atom-by-feature tensor. `encoder.frame_chunk` limits simultaneous
spatial frames and `activation_checkpoint` trades recomputation for VRAM. Neither
changes the data, neighbor list, temporal resolution, or gradients through past
frames. The selected H100 setting is 65 frames and no recomputation; a measured
48 ps example used about 16 GiB. Smaller GPUs can change these execution settings
in a new run config; exact continuation still requires identical config.

Scientific rationale and declared limitations: [experiment protocol](../experiments/predictive_memory_20260917/README.md).

The follow-up [velocity-input seed replicate](../configs/predictive_memory/replicate-xv-seed20260918.json)
reuses the release and training implementation with seed 20260918. Run its four
fits using `--velocity` at H=0,12,48 and H=48 with `--repeat-anchor`; compare them
with `python -m src.training_methods.predictive_memory.compare --config
configs/predictive_memory/replicate-xv-seed20260918.json --modalities xv`.
Its detached tracked command waits for the primary pilot to succeed before
using the GPU. Allocation plans and live queue state remain in each output's
`technical/` directory.

The separate [H200 capacity task](predictive_memory_h200.md) assigns eight
width-32 velocity-input fits across two seeds, using a portable bundle of the
same observations. It supersedes the previous crystallization-oriented H200
assignment for new work.

The H100 optimization follow-up uses the four recipes in
`configs/predictive_memory/optimization/`: original versus 1.0 present loss
weight, each at seeds 20260917 and 20260918. Each recipe fits velocity-input
H=0,12,48 and the H=48 repeated-anchor control from scratch for 12,000 updates.
It reuses the completed immutable observation cache; do not rerun preparation.
The earlier 3,000-update checkpoints and configurations remain unchanged.

After a recipe's four fits complete, run both collectors:

```bash
python -m src.training_methods.predictive_memory.compare --config configs/predictive_memory/optimization/original-seed20260917.json --modalities xv
python -m src.training_methods.predictive_memory.diagnose --config configs/predictive_memory/optimization/original-seed20260917.json --modalities xv
```

`diagnose` also accepts the original pilot or replicate configuration. It uses
saved embeddings and heads on CPU, checks release and source/center/anchor
identity, and writes a separate `diagnostics/` result without rewriting old fit
metrics. It refuses to overwrite completed diagnostics. The readouts remain
diagnostic predictors; the native trainable encoder remains the research model.
Each fit now records validation trajectories in `technical/validation.jsonl`.
The detached H100 controller for allocation 995957 records its plan, logs and
progress under `output/predictive_memory/optimization-20260917/technical/`.
