# Target-complete compact MACE — 2026-09-08

> Completed experiment record — implementation retired on 2026-09-09.
> Commands, plans, and implementation paths below describe the original run,
> not the current supported trainer. Reproduction of that protocol requires
> the run's `tracking/*/source.tar.gz` and recorded configs under `output/`.
> Existing results/checkpoints are retained. Use the
> [current 80-atom recipe](../mace_plain80_20260909/README.md) for new training.

Research question: does fixing the encoder's hidden exclusion of TDA target atoms
improve topology fidelity, temporal behavior and spatial structure together?
The preceding small gains do not establish a meaningful improvement.

The previous encoder accepted 80 atoms but excluded message sources beyond 6.5 Å.
TDA used the center and 64 neighbors, including atoms beyond that cutoff. The
[audit](../mace_joint_properties_20260908/receptive_field_audit.py) demonstrated
target-changing motions invisible to that encoder. The 65 atoms are a subset of
the 80; the disagreement was introduced by the encoder's mask.

## Configuration and scientific protocol

- [Training configuration](training.json), [queue plan](plan.json),
  [static analysis configuration](static.yaml), [tracked launcher](run_spec.json).
- Exactly 80 atoms. Sort radii; keep full source weight through rank 70. Between
  ranks 70 and 80 use `(1-u)^3 (1+3u+6u^2)`, with normalized radial coordinate `u`.
  Rank 80 has zero source weight. Five full-weight atoms beyond the 65 targets
  protect their membership against small ordering changes after float16 storage.
  Atom counts include the center. Pairwise MACE cutoff stays 5 Å.
- The exhaustive preflight also found target atoms without a two-hop central
  path even after removing the 6.5 Å mask. Therefore compute first-layer atomic
  features throughout the patch and add a weighted context pool to the central
  second-layer readout: `last + W @ (weighted_mean(first_layer) - first_center)`.
  Source messages and pooling both use the smooth support window. Initialize
  `W = 0.1 I`; all 16,384 new weights train. Keep the first 128 central channels.
- 256 scalar channels of MLIP-initialized MACE-MP-0b2 small. Warm-start
  the completed previous all-objective model and fixed train-only scalers;
  restart the optimizer and schedule. This is a continuation, not fresh MLIP-only
  initialization. No teacher. Only the context projection is newly initialized.
- Spatial/temporal VICReg, TDA regression, future-latent prediction, small-noise
  invariance and decoded future-TDA prediction. Same objective as the preceding
  `nuisance_future` configuration; ranking and PCGrad remain disabled.
- Al/Mg/Ta, 0.1 ps temporal VICReg. Al shooting pairs at 0.3 ps remain excluded
  from that term, but supply spatial, TDA and longer-horizon forecast examples.
- Effective batch 1,536 quadruplets; microbatch 512 for graph-memory headroom.
  Whole-batch VICReg gradients are cached exactly across microbatches. Six epochs,
  704 updates/epoch, 4,224 total updates. Per-step warmup/cosine peak rates:
  encoder 3e-4, heads 3e-3. W&B online.
- 286,720 available training quadruplets, 993,768 distinct neighborhood states:
  Al 245,760 / 847,096; Mg 32,768 / 117,338; Ta 8,192 / 29,334.
  Balanced sampling cycles smaller pools. Six epochs expose 6,488,064
  quadruplets; repeated exposure does not create additional distinct data.
- Custom PyTorch loop, not a Lightning Trainer. Trainable encoder: 8,236,176;
  frozen unused MLIP readouts: 2,192; TDA head: 74,016; forecast head: 198,656.
  Exact converted-model counts are checked in the preflight and logged online.

## Reproduction and outputs

Run from the repository root in the `pointnet` environment:

```bash
python scripts/experiment_registry.py run --spec experiments/mace_target_complete_20260908/run_spec.json
```

Use a new output directory for another training invocation. The current detached
supervisor runs on node53 in allocation 984861. Allocation safety cutoff is
2026-09-09 05:25:06 Europe/Paris. Logs, checks, selected weights and reports are in
[`output/mace_target_complete_20260908_context`](../../output/mace_target_complete_20260908_context/).
Optimizer checkpoints are in `/tmp/vmorozov_mace_target_complete_20260908`.
Existing datasets/caches are read-only; no new `/home/ids` analysis outputs.

The queue first runs [support_audit.py](support_audit.py), then the existing
`src.analysis.mace_joint` gradient and peak-LR preflight. Coverage is checked on
every cached training/validation view and all six static-Al frames. Each TDA atom
must have unit source weight and a message path to the pooled readout. The audit
also counts atoms lacking a central path, which motivated the context pool. Model checks cover
previously invisible motions, permutation/rotation invariance, the unchanged
first-layer descriptor and rank-80/81 boundary exchange. The numerical preflight
compares cached gradients with full backpropagation and exercises a complete batch.

Training is followed automatically by frozen-embedding probes, physical forecast
and perturbation metrics, and the existing encoder-only full static analysis on
Al 166/170/174/175/177/240 ps. The exported encoder records the same adaptive window.
Archived spatial-reference scores are reused only after exact center-grid checks.
The preceding control results and interrupted nuisance checkpoints are preserved
in `output/mace_joint_properties_20260908`; its remaining jobs were superseded.

## Interpretation limits and findings

The support correction establishes access to target information, not learned
topology sensitivity or simultaneous improvement. The finite 80-atom graph is
still not the full 10 Å MLIP halo. Its boundary window is continuous, while radius
order statistics are not globally differentiable at ties. The TDA target itself
still has a hard 65-atom boundary, a separate possible source of target noise.

Static analysis is descriptive: several Al snapshots are ancestors of continuation
training data. Mg/Ta and the eligible Al temporal validation each have one source;
independent-source confidence intervals are not estimable for those metrics.
Report joint improvements honestly; a failed no-regression screen stays failed.

Generated findings: [support audit](../../output/mace_target_complete_20260908_context/support_audit/report.json),
[preflight](../../output/mace_target_complete_20260908_context/preflight.json),
[queue status](../../output/mace_target_complete_20260908_context/status.json),
[completed comparison](../../output/mace_target_complete_20260908_context/RESULTS.md).

New `src/models/encoders/mace_support.py` is maintained shared encoder code;
this directory contains experiment records and its unique reproducibility audit.
Logs and audit results under the output directory are disposable diagnostics.

The completed support audit checked 1,146,880 training views, 37,888 validation
views and 772,953 static-Al neighborhoods. Every supervised target atom has full
weight and a context path. Without pooling, even the corrected window leaves
3,023 Mg training target-atom occurrences, 82 Mg validation occurrences and two
Ta training occurrences without a central path. All 48 previously invisible
motion examples now change the embedding. Rotation, permutation and first-layer
preservation checks pass; boundary-exchange differences are at float32 numerical
scale (mean squared difference below 9e-13). Static coverage follows the analysis
adapter's radius sort; training coverage preserves the original target atom order.

Five focused unit tests pass. Training outcomes remain pending; access to the
missing information is not evidence of useful learned improvement.

[Performance proposal, September 8](PERFORMANCE_PLAN.md): measured runtime and
memory, geometry reuse, GPU-resident data, smooth context readouts, causal-history
forecasting and data diversity. The proposal does not alter the running queue.
