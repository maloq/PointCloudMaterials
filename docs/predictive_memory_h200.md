# H200 task: capacity of the partial-observation memory encoder

This handoff replaces the **scientific assignment** in `docs/mace_causal_h200.md`.
That older bundle implements a different target and must not be used for these
runs. The new archive is `predictive-memory-h200-20260917.tar.zst`; copy its
`.sha256` file too. It contains the current uncommitted implementation, tests,
metric definitions, two H200 recipes, and the complete audited 150-source cache.
Raw MD trajectories and pretrained weights are not required. No simulations or
new cache preparation should be run.

## Experiment

Test whether doubling the atom-feature width improves predictive memory while
retaining the same exported 128-dimensional state. H100 runs width 16; H200 runs
width 32 (32 scalar, 32 vector, 32 rank-two channels), two spatial/temporal blocks,
the same four-component future-path mixture and the same 3,000 optimizer updates.

For **each seed 20260917 and 20260918**, run:

| Model | CLI options | Scientific comparison |
|---|---|---|
| Current positions + velocities | `--history-ps 0 --velocity` | Matched snapshot reference |
| Full 12 ps observed history | `--history-ps 12 --velocity` | Shorter memory |
| Full 48 ps observed history | `--history-ps 48 --velocity` | Longer memory, all 65 frames |
| Repeated current frame | `--history-ps 48 --velocity --repeat-anchor` | Separately trained temporal-capacity control |

This is eight fits, in addition to the H100's existing width-16 experiments.
Prioritize a complete four-model comparison for seed 20260917, then replicate
with seed 20260918. Test outcomes must not determine which recipes are run.
Source-balanced batch size stays **1**; changing it would alter the controlled
training budget, and this trainer explicitly rejects larger batches.

The radius is **17 A total**, including every graph atom; target support is
unchanged at 5--7 A. Future offsets remain 0.75,3,12,48,96 ps. Present/future targets
are continuous geometry and motion. No PTM, crystallization, onset, basin or
topology labels may train or select the encoder. No smoothness/bending objective.

## Transfer and setup

Send the new archive and checksum. The cache alone is 2.46 GiB in logical file bytes
(about 1.8 GiB of allocated disk space on the source filesystem);
the archive also includes source and environment recipes. The previous
`causal-mace-runtime-20260916.tar.zst` and its 1,800-window cache are not substitutes
for this **450-window** release. Existing original source hashes and metadata
remain in the new cache; copying does not require their original mount points.

```bash
sha256sum -c predictive-memory-h200-20260917.tar.zst.sha256
tar --zstd -xf predictive-memory-h200-20260917.tar.zst
cd predictive-memory-h200-20260917
PYTHONDONTWRITEBYTECODE=1 python scripts/project.py verify-bundle .
```

Verify before creating a Git directory, bytecode, environment or output inside
the export. Prefer existing conda `pointnet`; otherwise install Python 3.12 and
`environments/requirements-predictive-memory-gpu.txt` in an environment outside
the bundle. The archive has no `.git`: initialize a local import commit for the
experiment tracker after verification, respecting `.gitignore` so the data,
machine profile and outputs are excluded. Preserve the supplied `machine.local.yaml`
for portable dataset resolution; adjust only local execution/storage settings.
The training CLI uses CUDA directly and does not accept a `--device` argument.

## Prompt for the receiving agent

You are continuing the PointCloudMaterials predictive-memory study on this H200.
Read AGENTS.md, scripts/README.md, docs/predictive_memory.md,
docs/metrics/predictive_memory.md, experiments/predictive_memory_20260917/README.md,
and this handoff. Carry out the width-32 experiment above, using existing data only.

1. Verify the supplied archive and bundle before modification. Inspect the actual
   GPU, free VRAM, CPU/RAM allocation and wall-time limit. Reuse conda pointnet if
   present. Do not assume the sending server's node name, Slurm job ID or deadline.
   If this machine has no Slurm allocation, use the existing tracked-command
   workflow with detached processes; do not fabricate Slurm variables.

2. Load a provided H200 config with `src.project_runtime.paths.load_json` and
   instantiate `MemoryDataset(config, 48.)`. Require 150 sources and 450 windows,
   split into 90/30/30 sources and 270/90/90 windows. The loader verifies the
   release and shard checksums. Keep the source manifest, split files, center IDs,
   anchor times, target values and train-only normalizer unchanged. Do not run
   preparation: raw trajectories are intentionally not included. This cache
   supports only the present R=17 A and H<=48 ps study; radius, longer-history and
   additional-center studies require a separately audited observation release.

3. Run the focused tests below and a real-data CUDA forward/backward at width 32
   with all 65 frames. Check finite loss and gradients. Measure several warmed
   optimizer steps on multiple training-source windows; do not benchmark only a
   small synthetic graph. H200 throughput and peak memory must be measured here.

4. Use the extra VRAM first to retain all 65 spatial frames in one batch without
   activation recomputation (`frame_chunk=65`, `activation_checkpoint=false`).
   The factored equivariant temporal sum already avoids materializing the large
   time-by-time-by-atom-by-feature value tensor. Keep every atom, edge and observed
   frame; do not cap neighbors, subsample time or cache learned atom features.
   Reserve about 20% of the actual available VRAM. If needed, lower frame_chunk
   or enable checkpointing in a **new pre-fit config**, checking output/gradient
   agreement. These execution choices preserve gradients through history.

   If profiling shows host-to-device copies or Python gaps dominate, immutable
   training-window GPU residency is a useful optional optimization. Implement
   and test it in this new predictive-memory pathway, preserving exact input
   tensors, sampled example order, outputs and parameter gradients. Do not import
   the old `mace_causal` runtime: it expects different targets and cache objects.
   Benchmark one versus two concurrent independent fits only if both fit with
   memory headroom, and choose by aggregate completed steps/second. Do not enlarge
   the statistical batch, architecture, target packet or update budget to fill
   VRAM. Keep FP32 and record precision/backend settings; AMP/TF32 changes require
   a separate numerical experiment rather than silent changes to this cohort.

5. Use the two supplied `configs/predictive_memory/h200/width32-seedSEED.json`
   recipes. Run the four specified models for each seed and their comparison.
   Reuse `scripts/experiment_registry.py run --spec` to preserve source/config
   snapshots, then detach the queue. On Slurm, use the existing
   `src.training_methods.embedding_forecast.allocation` runner with this server's
   real allocation identity. Budget a complete four-model seed group from the
   measured throughput before starting; leave time for evaluation and checkpoints.
   Pass a timezone-aware `--deadline-utc`. Preserve `best.pt`, `latest.pt`, logs,
   exported embeddings, physical scores and all failure/resume evidence. Exact
   `--resume` requires the same saved configuration and release. Do not edit a
   started recipe to extend its update budget.

6. Select checkpoints by physical validation joint NLL only. The comparison
   reports source-paired improvements over snapshot and repeated-anchor controls,
   physical MSE by target block/lag, present reconstruction, and persistence error.
   Examine 96 ps forecasts separately because they exceed the longest observed
   history. To compare width 32 against H100 width 16, request the completed H100
   per-seed velocity-input outputs. Verify release SHA, exact evaluation keys,
   normalizer tensors, seed, training budget, decoder settings and objective before
   comparing. Width is the intended scientific difference; host/cache/output
   locations differ operationally. Do not compare raw latent MSE across encoders.
   Export any new cross-width metrics with documented formulas and implementation
   hashes through the existing metric-doc workflow. Report training-seed variation
   separately from source-bootstrap uncertainty; do not treat reused sources or
   adjacent anchors as new independent data.

7. Return both `output/predictive_memory/h200-width32-seed*` directories, including
   technical checkpoints/evaluation files, paired tables with frozen METRICS.md
   and implementation hashes, plots, training/runtime logs, source snapshots and
   a short report. Return any tested source/config changes as a patch or commits.
   Report incomplete fits explicitly. This is exploratory evidence on existing
   float16 observations, not exact mutual information, Markov closure or confirmed
   physical memory separated from quantization-noise averaging.

## Commands for one complete seed

```bash
python -m pytest tests/test_predictive_memory.py tests/test_predictive_memory_comparison.py tests/test_mace_causal.py tests/test_mace_causal_comparison.py tests/test_research_layout.py -q

python -m src.training_methods.predictive_memory.train --config configs/predictive_memory/h200/width32-seed20260917.json --history-ps 0 --velocity
python -m src.training_methods.predictive_memory.train --config configs/predictive_memory/h200/width32-seed20260917.json --history-ps 12 --velocity
python -m src.training_methods.predictive_memory.train --config configs/predictive_memory/h200/width32-seed20260917.json --history-ps 48 --velocity
python -m src.training_methods.predictive_memory.train --config configs/predictive_memory/h200/width32-seed20260917.json --history-ps 48 --velocity --repeat-anchor
python -m src.training_methods.predictive_memory.compare --config configs/predictive_memory/h200/width32-seed20260917.json --modalities xv
```

Repeat with `width32-seed20260918.json`. These commands specify the scientific
work; the receiving agent must wrap them in tracked, detached execution and supply
the actual local deadline after measuring the available allocation.
