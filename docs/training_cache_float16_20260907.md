# Float16 neighborhood caches — September 7, 2026

The user authorized float16 caches and explicitly requested that caches used by
the active MACE ablation queue wait until the queue finishes.

## Current execution

- **89 unused caches converted and verified.** Payload reduction: 3.4567 GiB.
  Observed allocated-space reduction after writes settled: **2.9543 GiB**.
  Filesystem block counts during a write can lag; the command reports exact
  payload savings, while this allocated-space figure comes from a later scan.
- **407 cache files deferred**, with **31.1686 GiB** expected payload reduction.
  They were checked to remain float32 while the queue was active.
- A detached watcher runs on `node53`. It guards queue PID 1204708 and its current
  training child PID 1204711 using their process start ticks, preventing PID-reuse
  confusion. The full queue must exit before its guarded conversion starts.
  The watcher PID is recorded in `deferred_launch.json`, not assumed permanent.

Plans, logs, statuses and launch record live in `output/cache_float16_20260907/`:

```bash
cat output/cache_float16_20260907/immediate.status.json
cat output/cache_float16_20260907/after_queue.status.json
```

To resume a stopped watcher on node53, first confirm the recorded watcher is no
longer running, then use the same maintained command:

```bash
conda run -n pointnet python scripts/convert_trajectory.py training-cache \
  output/cache_float16_20260907/after_queue.json
```

Completed files are reverified on a repeated run. Partial temporary files cause
an explicit error. The watcher survives the invoking shell but not a host reboot
or forced allocation cleanup; a non-complete status needs attention in that case.

Deferred cache roots include the temporal-hypothesis source cache, MACE uniform
training cache, cosine-run data/source caches, and the active static_context80.
The current queue's on-disk precision was not changed.

## Storage and computation

Only centered neighborhood coordinate caches are converted: `clouds.npy`,
`benchmark_clouds.npy`, `*.views.npy`, and `*.context.npy`. Canonical simulation
coordinates, global sampling-center coordinates, targets, normalization data,
integer IDs, checkpoints, and model weights retain their existing precision.

The shared temporal/MACE/atomic-context producers and the existing spatiotemporal
view recipe now write float16 neighborhood caches. Their readers and evaluation
paths decode to float32 before geometry, neural-network inputs, and float32
target construction. bf16 computation remains a separate training choice.

The converter checks all values for finite float16 range, compares the saved
array to the expected float16 rounding, and records source/output hashes and
rounding errors before atomic replacement. Each file has a `.float16.json`
sidecar; historical producer manifests remain provenance of original creation.
The maximum observed rounding error in the immediate conversion was 0.00390625
in the stored coordinate units (local clouds use Angstrom; normalized views use
their existing normalization). Float16 conversion is lossy.

Validation: 12 targeted cache, reader, temporal-campaign, spatiotemporal-loss, and
static-cache tests passed. Tests include overflow/non-finite rejection without
modifying the original, round-trip layout, repeat verification, and float32 batch
decoding. The new command implementation is a maintained tool indexed in
`scripts/README.md`; JSON plans and logs under `output/` are disposable run records.
