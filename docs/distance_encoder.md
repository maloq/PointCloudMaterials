# Joint distance-encoder training

The new [MD-history distance experiment](distance_encoder_history.md) fine-tunes
the shared encoder on -6/-3/0 ps observations, with a matched repeated-current
control and spatial-front evaluation.

The original model is named **CD-MACE128** (crystal-distance MACE). Its matched
regularized variant is **CD-MACE128-VC**. Names do not replace checkpoint hashes.
See the [comparison protocol and literature](../experiments/distance_encoder_20260926/VCREG.md).

[Scientific protocol](../experiments/distance_encoder_20260926/README.md) ·
[configuration](../configs/distance_encoder/multimaterial_early_20260926.json).

Use conda `pointnet-torch214`. CPU label preparation reuses the sealed dynamic
structural shards and the completed native/external crystal audits. Run `prepare
plan`, all four `prepare lane` partitions, then `prepare seal`. Every source's
raw-frame mapping, graph and PTM receipts are bound; missing/changed artifacts
fail rather than silently dropping rows. The labels live in
`${storage:cache}/distance-encoder/multimaterial-20260926`. Coordinates are reused
from the existing structural release; they are not copied into the checkout.

CPU array **1009557** completed all four lanes. The release contains 12,747,736
training and 192,000 selection rows across 22,098 frames. Label agreement was
checked against 1,914 overlapping original Al observations, with zero difference.
Keep the frozen label producer in the launch's `technical/label-code` directory.

The detached training launch is Slurm step **1009378.16** on **node59**, using
both RTX PRO 6000 Blackwell GPUs. It started September 26 at 13:51 Europe/Paris.
[Online training record](https://wandb.ai/teshbek/PointCloudMaterials/runs/b6fb43f0af0c66701b4a).
The earlier attempt was interrupted before its first validation pass to correct
the Brier diagnostic's Boolean-label dtype; its snapshot, logs and partial
checkpoint remain in `technical/attempts/brier-dtype-correction`. The new fit
starts from the declared initial encoder. A local 448-row real-data check
completed validation and encoder backward passes, including zero-distance and
right-censored observations. This check did not create a W&B run.

Training runs through `torch.distributed.run --standalone --nproc_per_node=2`.
The declared global batch is 4096, with 2048 examples on each GPU. Both processes
hold the immutable, normalized float32 coordinates in VRAM; data are read and
checksum-verified once. No trainable embeddings are cached. Every epoch uses
the same global permutation across ranks, with disjoint slices and zero-weight
padding at the final batch. cuEquivariance, fused AdamW and compiled MACE operate
with bfloat16 autocast; probabilities/losses and optimizer master weights remain
float32. Hardware checks are local, not W&B runs.

Save every 256 updates and at epoch boundaries. `technical/last.pt` contains the
encoder, head, optimizer, full update count and identities. Resume using the
same frozen code/config. The deterministic full-dataset permutation reproduces
the next batch. The state records an incomplete time-limited checkpoint explicitly;
only the completed 12-epoch fit emits `complete.json` and triggers downstream
evaluation. W&B uses a stable resumable online run in
`teshbek/PointCloudMaterials`, group `joint-distance-early-20260926`.

The follow-up evaluates the jointly trained head, then exports the new encoder
over the exact previous fixed Al context dataset, uniform augmentation and scan
paths. The vector-message and harmonic-hierarchy readouts each train 16 epochs
with their existing distance-NLL protocol and batch 256. Frozen feature caches
follow the six-entry policy with active leases. The exported feature precision
is float32, matching the prior context comparison. Checkpoints and predictions
are preserved independently of disposable features.

Run directory: `${storage:analysis}/distance_encoder/multimaterial-early-20260926`.
Progress, online-run receipt, source snapshot and Slurm launch details are under
`technical/`. Learning metrics use `analyses/learning-v1`; downstream comparisons
use `context/<model>/analyses/`. No automated tests or `tests/` directory are added.

## Regularized comparison and independent local evaluation

Submit the new recipe with `python -m src.research.distance_encoder.queue submit
--config configs/distance_encoder/multimaterial_early_vcreg_20260926.json`.
The launcher freezes source, metric definitions and both configurations, requests
two H100 GPUs for eight hours, and saves its Slurm receipt. The worker runs the
12-epoch fit, then evaluates both completed encoders independently of the old
vector/harmonic queue. Check `technical/queue-state.json` and `slurm-JOB.log`.
Resuming uses the recorded frozen `queue worker --config .../code/config.json`.

The `distance_encoder.local` entry point accepts `--config`, `--checkpoint`,
`--output` and `--name`. Use `--provisional` only for a copied, immutable partial
checkpoint and a separate output. It consumes focal geometry only, exports scalar
features into the existing six-entry leased cache, evaluates the joint head,
fits the local distance and onset readouts, and exports quality metrics. It does
not wait for the vector/harmonic feature exports. The declared distance-predictor
fit remains online in W&B. Its final evaluation updates that existing run via the
API; onset diagnostic probes and numerical checks stay local and create no online
runs. Their local predictions, checkpoints, metrics and progress are retained.

Running CD-MACE128 continues using its original frozen source/configuration.
The current trainer implements the explicit regularization recipe; reproduce
the baseline from its saved producer. Its original configuration is preserved.

The regularized training job is **1009584**, two H100 GPUs on node53. Its root is
`${storage:analysis}/distance_encoder/multimaterial-early-vcreg-20260926`.
The independent provisional evaluation runs in step **1009550.3**, one L40S on
node50, under the original run's `local-provisional-20260926/`. It uses a copied
checkpoint at update 20,480 (epoch 6.5789). Final local evaluations are part of the
regularized job's queue and use each model's completed epoch-12 checkpoint.

[CD-MACE128-VC online training](https://wandb.ai/teshbek/PointCloudMaterials/runs/b0e6dd41dfc6c43b99e8). The initial 224 updates
were healthy at about 24,500 neighborhoods/s. Both global-batch regularizer
gradients and real-geometry encoder backward passes were checked locally.
The provisional local evaluation completed successfully; its report is
`local-provisional-20260926/README.md` under the original run.
