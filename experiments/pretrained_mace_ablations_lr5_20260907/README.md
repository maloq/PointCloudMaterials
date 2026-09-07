# MACE ablations: quota recovery and 5× peak LR — 2026-09-07

User-requested recovery of the [matched ablation queue](../pretrained_mace_ablations_20260907/README.md).
The main MACE run and standard static analysis completed at 01:50 Paris. Its
successor started normally, but failed at optimizer update 2,000 while writing
`last.pt` to `/home/ids`: **Disk quota exceeded**. The last complete checkpoint
was step 1,900. The controller propagated that failure and stopped. This was a
checkpoint-storage failure, not a failed dependency or a nonfinite loss.

## Recovery and protocol

All five trials now use peak encoder LR **1.5e-4** and head LR **1.5e-3**, exactly
5× their previous peaks. The minimum LR remains 1e-6. Each trial has one epoch
of linear warmup from 5% of peak followed by three epochs of per-step cosine
decay. The full four-epoch budget (5,632 updates), final-epoch selection, seed,
data, 80 points, 0.1 ps temporal pairs, probes and static settings are unchanged.

The five trials are all objectives, no temporal VICReg, no TDA, no forecast,
and no spatial VICReg. **Every trial restarts from the original MLIP weights**
with fresh optimizer state. The failed lower-LR control is preserved and is not
mixed into this matched comparison. The completed main run is also preserved.
New output paths and online W&B IDs distinguish the replacement experiments.

Optimizer checkpoints now use node-local storage:
`/tmp/vmorozov_mace_ablations_lr5_20260907/NAME/checkpoints/`.
An actual 101,996,527-byte model-and-optimizer payload was saved, atomically
replaced, and loaded twice there successfully before launch. Selected weights,
training/probe metrics, plots and comparison reports remain in the repository.
Temporary node-local checkpoints are not persistent across node cleanup; each
completed trial exports its selected checkpoint to its repository run folder.
The source data cache is only read from `/home/ids`.

## Commands and schedule

```bash
conda run -n pointnet python -m src.training_methods.pretrained_mace_queue \
  --plan experiments/pretrained_mace_ablations_lr5_20260907/plan.json
```

Detached controller launched September 7 at **11:01 Paris**, PID 1204708, in
existing allocation 983527. No new Slurm job was submitted. The predecessor
already completed, so the first training starts immediately. Admission estimates
use 85 minutes per training, based on the measured approximately 0.85–0.90
seconds per step, plus probe time. The current safety cutoff is **18:22 Paris**.
Actual completion depends on runtime and remaining allocation time.

The controller trains/probes the five matched trials, then runs standard static
analysis. If insufficient time prevents admitting a later training, it now
still analyzes already trained trials; unfinished trials remain explicitly
pending. It never marks an incomplete sweep complete. Static analyses retain
all six Al frames and their 772,953 centers, with Blender renders disabled as
in the original ablation plan. Completed disposable inference caches are removed
after reporting; selected encoders, plots and metrics remain.

```bash
PYTHONPATH=. conda run -n pointnet python \
  experiments/pretrained_mace_ablations_20260907/verify.py \
  --plan experiments/pretrained_mace_ablations_lr5_20260907/plan.json
```

Verification covers matched configurations/update counts, exact objective
removal, dependency/failure handling, linear probes, and analyzing completed
trials when a later training cannot fit. Storage verification is recorded in
`output/pretrained_mace_ablations_lr5_20260907/storage_verification.json`.

## Outputs and file roles

- Plan/configurations here are experiment records; no runner is copied.
- Maintained orchestration and probes reuse `src/training_methods/pretrained_mace_queue.py`
  and `src/analysis/pretrained_mace_ablation.py`.
- `output/pretrained_mace_ablations_lr5_20260907/status.json`: controller state.
- `output/pretrained_mace_ablations_lr5_20260907/runs/NAME/`: training, probes,
  selected weights and standard static analysis results.
- `output/pretrained_mace_ablations_lr5_20260907/RESULTS.md` and `comparison.csv`:
  incremental comparison. Scientific findings remain pending.
- Launch logs and write-test records are disposable diagnostics in the run output.

The same single-seed, limited-budget and static-ancestor-overlap limitations
documented in the original ablation protocol apply.
