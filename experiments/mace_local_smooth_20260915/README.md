# Direct temporal training of an informative local state

**Discarded approach (16 September 2026):** replacement-embedding training on
frozen encoder features is no longer pursued. Scientific results and exact recipes
are retained. Embedding forecasting and native encoder training remain active.
See [scope and historical reproduction](../../docs/discarded_frozen_encoder_maps.md).

Question: can a nonlinear state map reduce the current encoder's temporal jumps
while retaining current local bond order and instantaneous topology?

This implements the first stage of the
[smooth-manifold proposal](../mace_velocity_20260915/LITERATURE_REVIEW_SMOOTH_MANIFOLD.md).
It freezes the completed coordinate/velocity MACE checkpoint and uses its
256-dimensional coordinate-derived structural block. One uncompressed reference
and maps with 8/16/32 coordinates receive matched current-physics supervision.
Compact models compare temporal weights 0/0.3/1/3/10 with two initialization seeds.
Temporal loss directly penalizes change relative to within-context spread;
a separate low-order term addresses disordered environments. There is no raw
teacher-retention term. Covariance regularization discourages collapsed or
redundant local coordinates.

All 1,125 retained records and 13,500 tracked pairs keep their preparation-level
split and source weights. Train-only target scales and 512 reference anchor
identities are preserved. The old test sources are explicitly development evidence
because their results already informed this research. Physical errors come from
jointly fitted heads; independent-observable/probe validation remains necessary.

Validation selects checkpoints using physical error plus the full temporal
penalty. A proposed preservation gate allows at most 10% relative increase in
each physical-error family versus the matched reference. Among passing candidates,
validation selects the lowest within-low-order RMS jump. The 0.10 jump target
and preservation gate are reported separately; failed candidates remain visible.

This snapshot stage does not train curvature, shared local motion directions,
history processing, or MACE weights. Later stages need consecutive observations.
A compact bottleneck is not evidence of a low-dimensional temporal manifold.

Recipe: [mace_local_smooth.json](configs/mace_local_smooth.json).
Calculations: [metric definitions](../../docs/metrics/mace_local_smooth.md).
Output: `output/mace_local_smooth/velocity-frozen-20260915/`.

Historical reproduction in conda `pointnet` from the archived source root:

```bash
python -m src.research.mace_local_state.run --config configs/analysis/mace_local_smooth.json --stage smooth-prepare
python -m src.research.mace_local_state.run --config configs/analysis/mace_local_smooth.json --stage smooth-fit
python -m src.research.mace_local_state.run --config configs/analysis/mace_local_smooth.json --stage smooth-evaluate
```

`smooth-all` performs all three stages. Preparation resumes verified source units;
fitting resumes completed saved epochs with identical identities. See [completed findings](RESULTS.md): no candidate passed the information gate.
The longer 32/64D comparison using [the capacity recipe](configs/mace_local_smooth_capacity.json)
retains 22 completed fits; final evaluation completion was not established. It is
no longer an active submission.
