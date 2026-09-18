# Shared-pretraining W&B dashboard

New runs use the compact logger in
`src/training_methods/shared_pretraining/tracking.py`. Frozen running jobs and
historical W&B histories keep their original logging. No training objective,
checkpoint selection, health gate or target calculation changes.

The current mixed-triplet VICReg recipe emits **16 custom plotted series**:

| Section | Series | Meaning |
| --- | --- | --- |
| `loss` | total, physical, tda, vicreg, physical_correlation, backtracking | Every component is already weighted; components add up to total. |
| `vicreg` | total, invariance, variance, covariance | Unscaled VICReg: total = 25 I + 25 V + C. |
| `validation` | score, present_physical, present_tda | Source-balanced selection errors; TDA here is unweighted. |
| `health` | projector_std, projector_effective_rank | Within-domain collapse diagnostics; effective rank is the covariance participation ratio. |
| `optimization` | head_learning_rate | Actual scheduled LR at the last update of the logging interval. |

All plots use **epoch equivalents** on the x-axis. `epoch` and `training_step`
are retained as hidden axes, without their own automatic plots. The encoder LR
is the head LR times the configured fixed multiplier; its duplicate curve is
removed.

Loss and health points average every update since the previous emission, rather
than showing one arbitrary update every ten. Emissions occur at the first
update, the configured log interval, validation and clean shutdown. Short final
intervals are included. Raw per-update values remain available for examining
spikes. The means do not change gradients or evaluation calculations.

The W&B summary retains the latest per-material/potential diagnostic values,
selection sample/source counts, the training-group-mean reference score, and
best validation score/step. Full per-group, spatial/temporal, block, timing,
gradient and memory diagnostics remain in `technical/updates.jsonl` and
`technical/validation.jsonl`. Selection CSVs and metric contracts are unchanged.
W&B's built-in system monitoring remains enabled.

There are no `train_by_group/*` copies, spatial/temporal copies of every scalar,
global mixed-material state-spread plots, boolean/count plots, or custom timing
and VRAM plots. An explicit metric allowlist prevents new diagnostic fields
from automatically creating dashboard clutter.

MACE and ordinary VICReg use the same logger, with no backtracking panel when
that objective is absent. JEPA replaces the VICReg panels with its weighted
objective, next-latent MSE and SIGReg. Causal training adds a future-loss
contribution and horizon-averaged future physical/TDA validation errors.
Future supervision contributes zero on structural replay updates when averaging
the loss, so displayed loss components still sum to the displayed total.

The offline W&B verification used real mixed-triplet training rows and checked
loss conservation, VICReg decomposition, hidden axes, final partial intervals
and causal replay averaging. Artifacts:
`output/shared_pretraining/logging-cleanup-20260918/technical/`.
