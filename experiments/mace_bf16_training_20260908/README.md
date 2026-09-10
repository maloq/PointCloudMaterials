# Compensated BF16 continuation — 2026-09-08

> Completed experiment record — implementation retired on 2026-09-09.
> Commands, plans, and implementation paths below describe the original run,
> not the current supported trainer. Reproduction of that protocol requires
> the run's `tracking/*/source.tar.gz` and recorded configs under `output/`.
> Existing results/checkpoints are retained. Use the
> [current 80-atom recipe](../mace_plain80_20260909/README.md) for new training.

Question: continue the current 80-atom MACE training with the numerically
qualified compensated BF16 radial matrices, retaining its optimizer and budget.
The user explicitly requested starting training now instead of waiting for the
deferred production-size benchmark. That waiting queue and the FP32 controller
were stopped by verified process identity.

Configuration: [training.json](training.json), [queue](plan.json),
[static analysis](static.yaml), [tracked run spec](run_spec.json).
Output: [`output/mace_bf16_training_20260908`](../../output/mace_bf16_training_20260908/).

```bash
python scripts/experiment_registry.py run --spec experiments/mace_bf16_training_20260908/run_spec.json
```

Resume from the preserved first-epoch checkpoint at **update 500**, retaining
Adam moments and reconstructing the exact next step of the original warmup/cosine
schedule. Complete the original **4,224 updates**, leaving **3,724 updates**.
Rebuild the original validation draw and first-epoch sample permutations from
the same NumPy seed, then skip the 500 completed batches. Coordinate augmentation
is seeded by global step and therefore continues without an RNG-state guess.
The old checkpoint predates any epoch-end selection, so the original initial
validation and encoder are preserved. Epoch-one logged train loss covers only
the remaining segment; `train_loss_updates` records its denominator.

This is an explicit `resume_first_epoch` protocol, separate from a fresh warm
start. The old producer does not save scheduler or sampler state; later-epoch
resumption is deliberately rejected rather than guessing checkpoint selection
or random state. A CPU regression test verifies identical Adam updates and LR
values after restoration, including crossing the warmup boundary. Only execution
configuration is permitted to differ from the checkpoint's scientific settings.

The configuration uses chunk 1536, effective batch 1536, GPU-resident data,
geometry reuse, compiled compensated BF16 radial matrices and FP32 geometry,
pooling, losses and master parameters. Ordinary BF16 was rejected; compensated
BF16 passed fixed-weight numerical checks and measured 1.21× throughput at
chunk 512. No production-size matched speedup is claimed yet. The queue runs
full numerical/peak-LR preflight before training, then frozen probes and the
existing encoder-only full static-Al analysis. W&B is online in a new run linked
to the source checkpoint; the old history remains intact.

Optimizer checkpoints stay under `/tmp/vmorozov_mace_bf16_training_20260908`.
The resume weights, checksums, reports and selected final encoder stay in the
repository. See [resume provenance](../../output/mace_bf16_training_20260908/resume_source.json).

File roles: this directory is an experiment record. First-epoch restoration is
maintained shared implementation in `src/training_methods/mace_resume.py`, with
its numerical test in `tests/test_mace_resume.py`. Logs, status files and generated
reports in the output directory are disposable experiment artifacts. Existing
maintained training and queue entry points are reused.

The full preflight passed and actual BF16 training was verified at update 530/4224. Adam state and scheduler position 500 were restored, with next encoder/head LRs 0.00021741477272727296 / 0.002174147727272731. See [live verification](../../output/mace_bf16_training_20260908/live_verification.json) and [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/e6c2265c).

## Interim metric review — 2026-09-08

The CPU-only experiment diagnostic [analyze_training_metrics.py](analyze_training_metrics.py)
captures the live logs, reconstructs weighted loss contributions, compares fixed
validation measurements, and documents every objective and diagnostic. It does
not change training or allocate GPU memory. Results and plots are saved in
[training_review/RESULTS.md](../../output/mace_bf16_training_20260908/training_review/RESULTS.md).

```bash
conda run -n pointnet python experiments/mace_bf16_training_20260908/analyze_training_metrics.py
# Reproduce the same report from its captured input, without rereading live logs:
conda run -n pointnet python experiments/mace_bf16_training_20260908/analyze_training_metrics.py \
  --snapshot output/mace_bf16_training_20260908/training_review/inputs.json
```

Interim finding: batch losses fluctuate modestly, but improved TDA reconstruction
coexists with worse normalized spatial/temporal coherence. Most of that change
predates the BF16 continuation. The original initial validation is logged at
resume step 500 in W&B; the report restores its correct step-zero position and
includes the actual resume validation. No training settings were changed by
this review. The diagnostic is experiment-specific reproducibility code; its
captured logs, CSV files, figures and report are generated analysis artifacts.
