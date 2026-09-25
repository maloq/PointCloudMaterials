# Running the supervised MACE capacity comparison

[Protocol](../experiments/supervised_information_20260925/README.md)
· [Recipe](../configs/supervised_onset/information_20260925/campaign.json)
· [Output](../output/encoder_supervised/capacity-20260925/README.md)

Use conda `pointnet-torch214` and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the
installed e3nn constants. The live runner now requires
`supervised_onset_information_v4`; AP-specific historical protocols are rejected.
Historical trained protocols retain their frozen source.

For ordinary new training, use the128-channel default in
[`default128.json`](../configs/supervised_onset/default128.json), or run the same
campaign commands below with
`--config configs/supervised_onset/default_campaign.json` for its two observed/
relaxed fits on one GPU. The four-size recipe below is an explicit capacity
ablation. [Measured parameter counts](encoder_research/mace_sizes.md).

**AP optimization is retired.** Jobs 1008330–1008333 and collector 1008336 were
cancelled on 25 September at the user's direction. Their source/checkpoints and
cancellation receipts remain in `capacity-20260925` as historical AP-trained
artifacts. Do not resume them as NLL-only fits.

New recipes train and select by predictive hazard likelihood, with AP reporting
only, under fresh `information-runtime-20260925` outputs. Batch/microbatch default to 256.
They require fresh preflight and submission; this policy change does not submit
replacement training. The existing cache is reused, with no new data generation.

Future runs require online Weights & Biases tracking in
[teshbek/PointCloudMaterials](https://wandb.ai/teshbek/PointCloudMaterials).
Encoder fitting, frozen readouts and descriptor controls open an online run
before optimization. Study identity plus component name determines the stable
run ID; screening, continuation and evaluation reuse that ID. Encoder loss,
gradient norm, learning rates and selection AP use the optimizer-update axis;
each readout has its own update axis. Final AP/Brier/calibration results are
summaries, so evaluating an earlier selected checkpoint cannot rewind history.
Receipts live in `technical/wandb/<component>/run.json`. Offline/disabled modes
are rejected; authentication failures abort instead of silently losing tracking.
Previously submitted frozen jobs retain their original local logging.
Only scientific training runs and their associated evaluation metrics belong
in W&B. Debugging, tests, smoke checks and hardware benchmarks stay local.
Tracking tests mock the SDK; they never create online test runs.

A separate September25 timing check on the RTX PRO6000 Blackwell used the2M
relaxed-input encoder,80-atom cached patches and effective batch256. Microbatch64
versus256 took0.2194 versus0.2290 seconds per ordinary update, and12.603 versus
13.265 seconds per full-population AP replay. Including replay every32 updates,
microbatch256 was about4.9% slower, with14.1 versus38.5 GiB peak allocated memory.
These warmed timings exclude startup, validation, checkpoints and W&B logging;
they do not establish H100 performance or learning quality. Batch256 remains the
requested default. This diagnostic ran separately from scientific training:
[raw local timing record](../output/encoder_supervised/batch256-default-20260925/technical/timing.json).

```bash
python -m src.research.supervised_onset.campaign check \
  --config configs/supervised_onset/information_20260925/campaign.json
python -m src.research.supervised_onset.campaign submit \
  --config configs/supervised_onset/information_20260925/campaign.json
python -m src.research.supervised_onset.campaign collect \
  --config configs/supervised_onset/information_20260925/campaign.json
```

The check runs scientific tests once, then actual CUDA/cuEquivariance forward,
backward, optimizer and likelihood-validation paths for every width and
both input domains. Production batch dimensions use a recorded small cohort
subset; its enriched subset scores are **not experiment results**. It does not
train/select scientific checkpoints. Data hashes are checked against the existing
580 MiB IDS cache. No new data are generated.

Submission verifies preflight identities, freezes executable source and submits
four independent single-GPU jobs to H100/RTX6000PRO. Each requests8 CPUs,48 GiB
host memory and8h45m, covering its8.5-hour worker budget and cleanup. Jobs survive
the interactive session. Duplicate submissions fail; every accepted job is
recorded immediately. The interactive allocation is used only for checks.

Output contains `small/`, `500k/`, `1m/`, `2m/` and shared `technical/`. Each size
retains context, identities, launch records, checkpoints, queue state, Slurm logs
and comparisons. Shared `technical/submissions.json` lists exact job IDs. Resume
using frozen code/config; never edit that tree. `collect` combines existing
per-size CSVs, including repeated controls, and records missing/failed coverage.
Its parameter-budget columns describe the encoder study, not the size of every
auxiliary descriptor predictor or ensemble.

No hardware benchmark or learning-rate search is inserted in training. All
encoders use cuEquivariance, resident graph/radial/angular caches, batch256 and
importance-corrected predictive NLL. No AP replay or ranking objective is used.

The current recipes enable native `ir_mul` layout, fused cuEquivariance convolution
and full-graph dynamic compilation. Repeated geometry-index plans have a bounded
256 MiB cache per domain. See the [implemented runtime and validation](encoder_research/runtime_refactor_20260925.md).

The 2026-09-25 padding fix sets graph-bank atom capacity to `80 * microbatch`,
including partial batches. Dummy nodes are disconnected and zero-weight;
physical graph size and pooling normalization are unchanged. This avoids repeated
cuEquivariance tuning as the total number of real atoms changes across batches.
The run input ledger records capacity. Existing output identities remain frozen:
new implementations require new output paths, while old runs resume using their
original source snapshots.
