# Causal MACE: H100/H200 execution and data plan

The optimized implementation is integrated into the main checkout after the
original three-seed pilot and comparison finished. The isolated
`codex/causal-runtime` worktree remains the source of the running longer cohort;
keep it intact until those jobs finish. Run new commands from main or the verified
runtime bundle. Do not resume an old pilot checkpoint under these source hashes. The original baseline commit is
`69500e5`; training captures the dirty source and exact metric implementation.

## Implemented execution changes

- Disjoint graph packing combines unequal atomic neighborhoods into one MACE call.
  Spatial edges, physical boxes, same-atom temporal messages, smooth pooling
  denominators and counts remain separate per example. No padding or coordinate
  rounding is introduced.
- `runtime.residency: device` keeps verified histories and normalized training
  targets in VRAM; `train_device` keeps only training histories there; `host`
  transfers packed histories per call. The present cache has 3.785 GiB of history
  tensors before allocation overhead, so device residency is useful on this H100.
- `runtime.batch_size` sets graphs per forward. `training.batch_sources` remains
  the statistical batch. Chunked forwards concatenate states before a single
  physical task loss; this preserves the eligible-example hazard normalization.
  It does not free training activations like gradient accumulation would.
- Fixed inputs are validated once on CPU, avoiding data-dependent GPU checks on
  every forward. Finite gradients and exported outputs are still checked.
- Validation uses batched encoder/heads and one output copy per field. Frozen
  readout evaluation uses large dense batches. The constant-history control
  computes its identical branch once per training batch or evaluation pass.

FP32 remains the precision. Packing changes reduction order, so numerical
agreement is tested rather than promising bitwise equality to sequential execution.
The cache preparer, physical labels, source sampling, loss, normalization and model
parameter shapes are unchanged. See the [physical-state protocol](mace_causal.md),
[history-access diagnostic](research_glossary.md#state-sufficiency-diagnostic), and
[exact metric definitions](metrics/mace_causal.md).

```json
"runtime": {"encoding": "packed", "residency": "device", "batch_size": 8}
```

Run the actual-graph benchmark with:

```bash
python -m src.research.mace_velocity causal-benchmark \
  --config configs/mace_causal/runtime-benchmark.json --device cuda:0
```

It checks outputs/gradients at widths 16/32 and batches 2/8/16 before timing.
Results retain GPU process snapshots, warmup, raw repeated timings and allocated/
reserved VRAM. Contended timings must be identified as such. Inputs are fully
verified before staging; cache verification and staging are outside timed updates.
The [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
describes the synchronization and transfer overhead these changes address.

## More useful experiments per GPU hour

Use C, D and repeated_anchor as the next matched cohort: does observed history
improve physical predictions beyond current geometry and velocities, after a longer
training budget? The H100 recipe in `configs/mace_causal/h100-packed/` uses width 16,
5,000 updates and eight source-sampled examples per update. Run that complete
seed-20260916 cohort first; replicate on other seeds before claiming robust gains.
The H200 recipes run the same cohort at widths 16 and 32 with three seeds each.
A larger width or effective batch is an experiment change, not a free speedup.
Compare widths only with an explicitly paired collector permitting that difference.

Fit linear and nonlinear frozen readouts for every encoder. Predeclare the expensive
matched history-access pair only for D in this cohort. This reduces those branch
fits by two thirds (12 versus 36 across the H200 study), while retaining the main
history-versus-snapshot controls. It narrows the diagnostic question to D; it does
not establish sufficiency of C or repeated_anchor. The comparison recipes enforce
all common readouts and add D's pair separately.

```bash
python -m src.research.mace_velocity causal-train \
  --config configs/mace_causal/h100-packed/width16-seed20260916.json --variant D --device cuda:0
python -m src.research.mace_velocity causal-probe \
  --config configs/mace_causal/h100-packed/width16-seed20260916.json --variant D \
  --probe-modes linear nonlinear --device cuda:0
python -m src.research.mace_velocity causal-probe \
  --config configs/mace_causal/h100-packed/width16-seed20260916.json --variant D \
  --probe-modes state_constant state_history --device cuda:0
```

One resident packed fit can use the current H100 alongside the original pilot.
Choose concurrency from completed-example throughput and peak VRAM, with headroom
for the largest graphs. Filling all VRAM is not the objective. More memory alone
cannot remove kernel-launch or memory-bandwidth limits. For additional hardware,
independent seeds/widths are the first parallelization unit; this implementation
does not introduce distributed training.

## More data: verified transfer inventory, preparation still required

The current cache covers 150 independent sources (90/30/30 split), four tracked
centers per source and three adjacent anchors near 300 ps: 1,800 windows. The test
segment has only one distinct onset center/source, so event skill is not established.
More time coverage should precede a large architecture search on this tiny segment.

A fixed, outcome-independent grid has been checked against all 150 native timelines:
20 blocks centered at 3, 33, ..., 573 ps, with three anchors per block at center
minus 0.75, center, center plus 0.75 ps. Each block retains all native frames from
center minus 3 through center plus 10.5 ps, including event confirmation beyond the
9 ps target. Keep the same four center IDs and whole-source splits. This supplies
36,000 candidate windows (21,600/7,200/7,200). These are more observations of the
same sources, not 36,000 independent trajectories. Source-weighted evaluation and
source bootstrap remain necessary. The source/file/timeline plan is in
`output/maintenance/causal-runtime-transfer/technical/raw-transfer-manifest.json`.

This grid is **planned, not prepared or trained**. The existing causal preparer
extends the retained mid-trajectory segment only; it cannot consume this plan yet.
An explicit grid adapter must regenerate the same fixed physical labels, PTM
assays, causal histories and per-block event/censor labels with fresh provenance.
Do not edit the existing cache plan or disable its identity check. Audit event
counts on train/validation and retain the predetermined test sampling.

The raw inventory contains 1,200 files: 150 trajectory manifests and their seven
arrays, totaling 101,495,260,860 bytes (94.53 GiB, before compression). Both original
WORK and STORE roots occur in the inventory. Array checksums are copied from the
producer manifests; manifest hashes and file sizes were verified here. The copy
must verify array checksums on the receiving server. No new quantization is needed.

For a receiving server with a known SSH alias and destination, copy only this
inventory, preserving its relative path structure under a dedicated data root:

```bash
rsync -a --files-from=output/maintenance/causal-runtime-transfer/technical/raw-files.txt \
  / SERVER:/DATA_ROOT/causal-raw/
# Copy raw-sha256.txt too, then run from /DATA_ROOT/causal-raw/ on that server:
sha256sum -c /PATH/TO/raw-sha256.txt
```

These are placeholders, not an executed transfer. Map original WORK/STORE prefixes
to their copied subdirectories in ignored `machine.local.yaml`. Keep manifests
unchanged. Alternatively prepare here and transfer only the new cache. Scaling the
current per-window size by 20 estimates about 76 GiB of history tensors, or 46 GiB
for training alone; actual graph sizes and archive compression must be measured.
Train-only residency leaves more activation room on the H100. The H200 can hold a
larger cache, but wider graphs still need measured headroom. Geometry/context-radius
and history-duration changes remain separate ablations requiring their own caches.
