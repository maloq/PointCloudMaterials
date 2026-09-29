# Running the encoder mechanism queue

**Completion audit,28 September:** all nine label-free encoders and nine
adaptation fits completed24 epochs. All54 milestone quality/displacement
assays and18 adaptation endpoint evaluations completed across the original,
v2 andv3 output roots. Arrays1012979/1012980 are complete. The submission and
recovery sections below retain their historical status at each cutoff.

To summarize saved results and source-paired predictive contrasts without
training or inference, use conda `pointnet-torch214`:

```bash
python -m src.research.encoder_mechanisms.results \
  --config PATH_TO_RECOVERY/technical/config.json \
  --output '${storage:analysis}/encoder_mechanisms/alignment-readout-recovery-20260928-v3/analyses/completed-review-20260928'
```

Pass a resolved filesystem path for `--config`. The report refuses to overwrite
an existing review; its metrics CSV has frozen definitions and implementation
hashes. Source/bootstrap summaries condition on the three fitted seeds, with
no claim of a population-of-seeds confidence interval.

The [completed scientific report](../experiments/encoder_mechanisms_20260926/COMPLETED-RESULTS.md)
links the reviewed tables and figures. All nine initialization/head-matching
receipts passed; all three frozen encoder states remained exact. The nine
scientific adaptation runs used online W&B and all18 associated evaluation
updates completed through the existing training IDs.

Scientific rationale and contrasts: [experiment](../experiments/encoder_mechanisms_20260926/README.md).
Numerical definitions: [metrics](metrics/encoder_mechanisms.md).
Active recipe: [configuration](../configs/encoder_mechanisms/alignment_readout_20260926.json).

Use conda `pointnet-torch214`:

```bash
python -m src.research.encoder_mechanisms.workflow submit --config configs/encoder_mechanisms/alignment_readout_20260926.json
```

Submission freezes source, configs and metric definitions under
`${storage:analysis}/encoder_mechanisms/alignment-readout-20260926/technical/code`.
It writes a receipt after every sbatch call, so a partial submission is visible.
Do not submit a second copy: inspect `technical/launch.json` and Slurm first.

A one-GPU real-data preflight on node58 must pass before readout controls start
on that node. It checks actual paired256-row tensors and finite gradients for
all three losses; it creates no W&B run. The nine-worker training array depends
on successful controls, allows at most three GPUs, and requests RTX6000PRO,
one GPU/4 CPUs/32 GB host memory/24 hours per worker. Each seed/treatment trains
24 epochs, evaluates0/4/8/12/18/24, and R1 then fits the three adaptation modes.
The same saved seed initialization and normalization are shared under a file
lock. Scientific encoder and predictor fits use online W&B with stable IDs.
Milestone quality evaluations and frozen diagnostic probes stay local; they do
not create the former per-checkpoint `z`, `joint` and `physical` control runs.
Associated final adaptation scores update the original training run via its
recorded ID. Active frozen jobs retain their existing logging; apply this policy
when launching a new frozen queue.

The CPU array separately audits selection/calibration/test roles, one role at a
time, with12 workers. It freezes the completed training harvest definition,
keeps source roles and uniform sampling probabilities, and never fits a model.
Support tables explicitly exclude future-centered candidate examples.

Completed original linear/128-unit readouts and supervised feature exports are
reused by identity and sample order. Retained pretrained features are checked by
native inference replay before reuse. One existing shared six-entry cache pool
serves all GPU lanes, preserving its historical recorded location and active
lease policy. No second pool or new training dataset cache is created; existing
release storage is unchanged. Checkpoints, predictions, metrics and assay input
receipts are permanent research artifacts, outside disposable feature caches.

Streaming training checkpoints every256 updates and at epoch boundaries;
supervised training checkpoints each epoch. Near the allocation deadline a
worker saves state and requests at most two Slurm requeues. Other scientific
failures are recorded and raised; they are never silently skipped. To resume
manually, use the receipt's frozen code/config and the same stage/index in a new
allocation, or requeue its existing Slurm task. Changing code requires a new
output revision, not an in-place rewrite of the frozen producer.

Outputs have a root README, `runs/<seed>/<treatment>` training provenance,
`controls/analyses` readout diagnostics, and `analyses/alignment`,
`analyses/adaptation`, `analyses/birth-availability` scientific bundles. Every new
numerical table carries METRICS.md and implementation hashes. Historical
artifacts retain their frozen definitions. The conditional birth-prediction and
linear-VAMP follow-ups are not automatically launched before their evidence gate.

## Current submission

As audited on 28 September, controls1009654 and all birth-audit1009656 tasks
completed. All nine encoders in1009660 completed24 epochs, but their enclosing
pipelines failed in milestone evaluation at PyTorch's eight-specialization
recompilation limit. The nine adaptation fits had not started. The producer is in
`technical/training-revision-v2/code`; a pending predecessor was replaced to
lock concurrent identity creation. Running controls and audits keep the original
`technical/code` snapshot. `technical/launch.json` records per-stage code paths.

The [results audit](../experiments/encoder_mechanisms_20260926/RESULTS-20260928.md)
separates completed encoder training from incomplete evaluation. Recovery uses:

```bash
python -m src.research.encoder_mechanisms.recovery submit \
  --config configs/encoder_mechanisms/alignment_readout_20260926.json \
  --output '${storage:analysis}/encoder_mechanisms/alignment-readout-recovery-20260928-v2' \
  --exclude-nodes node58
```

This validates all54 saved milestone checksums and completed quality receipts,
preserves21 quality and18 displacement results, and queues33 missing quality
evaluations plus36 displacement evaluations. It evaluates fixed epochs24 and12
first, then18/8/4/0, without selecting on outcomes. Each checkpoint uses a fresh
process; each adaptation mode also gets a fresh process. The nine likelihood fits
remain the original three modes × three seeds, with fixed R1 initialization.
The recovery never retrains the nine completed label-free encoders.

The frozen recovery revision retains identical model/geometry producers and
scientific templates, audits actual source timesteps for the0.75-ps displacement
assay, and reuses its exact saved pairs. New diagnostic fits are local; scientific
adaptation fits remain online. Both arrays use Slurm's RTX6000PRO partition,
at most two GPUs, with no required node and node58 excluded;
adaptation depends on successful evaluation recovery. Inspect the recovery's
`technical/launch.json` before resubmitting. If interrupted, run its frozen
`recovery worker --config technical/config.json --stage alignment|adaptation
--index N` in another allocation; saved fits/evaluations resume by identity.

Submitted28 September: evaluation array **1012894** (`0-8%2`) and dependent
adaptation array **1012895** (`0-2%2`). These replace the pending node58-pinned
arrays1012885/1012886 at the user's request. Slurm chooses an available node
other than node58. The frozen worker/configuration and scientific outputs are
unchanged; scheduler receipts are in `technical/scheduler-revisions/`.
All54 checkpoint checksums and30 source timelines passed the recovery input audit.

### Plotting recovery after the first evaluation array

Array1012894 completed seven workers, bringing the combined quality/displacement
coverage to50/54 and49/54. Two workers failed because dense plotting refitted
K-means and obtained different labels from the metric fit. The native renderer
now uses the actual metric-fit estimator and saves its centers; it retains label
agreement checks. An actual failing4096-anchor Zr frame passed the retained-fit
check. Metric populations, formulas and completed exports are unchanged.

The following submission was made, with array **1012979** for missing assays
and dependent **1012980** for the original adaptation fits:

```bash
python -m src.research.encoder_mechanisms.recovery submit \
  --config configs/encoder_mechanisms/alignment_readout_20260926.json \
  --output '${storage:analysis}/encoder_mechanisms/alignment-readout-recovery-20260928-v3' \
  --previous-recovery '${storage:analysis}/encoder_mechanisms/alignment-readout-recovery-20260928-v2' \
  --exclude-nodes node58
```

`--previous-recovery` requires the same scientific configuration and original
checkpoint source. Completed quality metrics are verified against their receipt
and checkpoint hashes; their original paths are recorded in the new input audit.
Only four quality/five displacement assays remain. All completed encoders and
assays are reused. The blocked adaptation job1012895 was cancelled. Arrays
1012979/1012980 use Slurm RTX6000PRO scheduling with node58 excluded. The
[scientific progress report](../experiments/encoder_mechanisms_20260926/PROGRESS-20260928.md)
records the newly available fixed-endpoint contrasts and their limits.
