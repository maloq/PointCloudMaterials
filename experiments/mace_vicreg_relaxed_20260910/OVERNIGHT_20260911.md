# Overnight H100 comparison — September 11, 2026

**Completed:** all twelve fits and their standard analyses finished. The final
aggregation error was corrected without retraining. See the
[September 11 findings](RESULTS_20260911.md) for the full comparison and
checkpoint-selection diagnosis; execution descriptions below record the launch.

Question: does learned use of five noisy observations improve relaxed topology
over the completed single-frame and mean-pooling MACE encoders? In particular,
does temporal information help after controlling for the extra atom-pooling
network? The user requested useful overnight experiments on the existing H100.

## Priority and protocol

The [core plan](overnight_20260911.yaml) contains nine fresh fits, ordered as a
matched trio for seed 20260910, then 20260911 and 20260912:

| Model | What it tests |
| --- | --- |
| `atom_temporal_blocks` | Follow the same atoms across five frames, apply temporal attention, then learned spatial pooling |
| `atom_anchor_blocks` | Same fusion capacity supplied with repeated anchor features; isolates the value of history |
| `transformer_blocks` | Attention over five pooled MACE frame embeddings; compare against the completed mean-pooling models |

The [residual plan](overnight_residual_20260911.yaml) follows with three seeds of
`residual_blocks`, an anchor embedding plus a learned temporal correction. This
tail uses any remaining allocation time after the priority comparison. No claim
is made that every queued fit will finish before the allocation expires.

All fits use `vicreg_mace_relaxed` through the original Lightning
`VICRegModule`: trainable MACE-MP-0b2-small, normalized 80-atom inputs,
9.192189 Å reference length, native 5 Å interactions, five offsets
[-3, -2.25, -1.5, -0.75, 0] ps, original VICReg 25/25/1 plus balanced H0/H1/H2
targets, batch 512, 24 epochs (648 updates), and online W&B. Whole-source
train/validation/test splits, optimizer, schedule and checkpoint selection are
unchanged. Prepared data is reused from `/home/ids/vmorozov/training-cache/mace-meam`.

The earlier transformer attempt stopped after epoch 0 / 27 updates. Its
checkpoint and provenance are retained as an interrupted attempt; this plan
starts all nine priority fits fresh from the same pretrained weights. No
completed single-frame or mean-pooling fit is repeated, and there is no
weights-only warm start or implicit resume. Automatic learning-rate-changing NaN
retries are disabled. The existing runner continues independent experiments
after a recorded failure; collectors require every declared seed and fail
explicitly if any fit/analysis is missing.

## Detached execution

CPU batch job **989068** owns GPU step **988041.37** on **node53**, using the
already allocated H100. The step has 10 CPUs, 80 GiB host RAM and one GPU, with
a 13.5-hour limit. It started at approximately **01:41 CEST** and ends by
approximately **15:11 CEST**, before allocation 988041 expires at 15:40:48 CEST.
It runs independently of the IDE connection. Unfinished fits keep their rolling
recovery checkpoints. The previous duplicate UMAP step 988041.31 was stopped;
the intended UMAP batch 989060 / 988041.34 continues independently.

Execution uses these existing commands through tracked specifications:

```bash
conda run --no-capture-output -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_vicreg_relaxed_20260910/run_overnight_core_20260911.json
conda run --no-capture-output -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_vicreg_relaxed_20260910/run_overnight_collect_core_20260911.json \
  --wait-for-dependencies-until 2026-09-11T15:10:00+02:00
conda run --no-capture-output -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_vicreg_relaxed_20260910/run_overnight_residual_20260911.json
conda run --no-capture-output -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_vicreg_relaxed_20260910/run_overnight_collect_all_20260911.json
```

These exact tracked specs have already been submitted; another execution needs
new execution output paths. The generated Slurm controller and GPU command
sequence are in `output/mace/jobs/overnight/`. They do not implement training.

## Results and assessment

Open the [flat MACE gallery](../../output/mace/index.html). Each completed fit
publishes UMAP, t-SNE, spatial views and topology metrics to
`output/mace/<variant>-seed<seed>/`. Checkpoints and training logs live on IDS
under `/home/ids/vmorozov/experiments/mace/overnight_<timestamp>/`; detailed
analysis artifacts live under `/home/ids/vmorozov/analysis/mace/artifacts/`.
Best checkpoints remain; inference caches and redundant final recovery weights
are removed only after successful analysis, as specified in the current configs.

The standard pipeline evaluates trained heads, training-only ridge readouts,
within-frame R², H0/H1/H2 errors and repeated-anchor/reversed-history interventions.
The [core comparison specification](analysis_overnight_20260911.json) adds the nine
new models to the twelve existing MEAM fits and writes
`output/mace/comparison/RESULTS.md` with paired whole-source bootstrap intervals
after averaging three seeds. The full comparison adds residual fusion if all
three additional fits finish. Primary scientific contrasts are atom history
versus matched atom anchor, and pooled attention versus mean pooling. This is
still the previously examined six-source follow-up cohort.

Launch validation composed all twelve configurations and checked epoch budget,
batch size, cache availability, unique model/seed destinations and fresh-start
semantics. Production execution and W&B status are recorded by the existing
trainer. Per-fit failure logs are retained; submission is not a completed result.

At 01:43 CEST, the first atom-temporal fit was running with online
[W&B run e65btizb](https://wandb.ai/teshbek/PointCloudMaterials/runs/e65btizb).
Logged optimizer steps 4 and 9 had finite training losses 20.1757 and 19.8126.
This verifies that forward/backward/optimizer updates are executing; these early
losses are not a scientific comparison. Slurm confirmed the GPU launch client
belongs to detached CPU controller `nodecpu05:3506647`.

Live controller log: `output/mace/jobs/overnight/slurm-989068.log`.
Core queue log: `output/mace/jobs/overnight-core/command.log`.
Residual queue log: `output/mace/jobs/overnight-residual/command.log`.

File roles: the two plans, comparison configuration, four run specifications and
this document are experiment records. Generated job scripts, logs and config
validation output are disposable run artifacts under `output/mace/jobs/`.
No maintained command, encoder implementation or training loop was added.
