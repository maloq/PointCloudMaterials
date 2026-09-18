# Static Al analysis of the active mixed GATr checkpoint

Question: how does the active temporal-backtracking GATr represent the same six
static Al configurations as the preceding encoder analyses?

The source is an immutable copy of `gatr-temporal-backtracking-20260918`'s
latest optimizer checkpoint at update 400, captured while training continued.
Its SHA-256 is `f2489e51bb7444193b6eeaf314945de9f4ca701841d9838e4e0852e62a22699c`.
It is not best-selected; no selection score is assigned to this snapshot.
The current fit uses mixed dynamic Al/Mg/Ti/Ta groups and temporal curvature
weight 21. The encoder still receives a single snapshot; temporal supervision
does not add history frames to its input.

Retain all six Al snapshots (166, 170, 174, 175, 177 and 240 ps), the 684,723-center
interior grid, seven spherical clusters and full standard analysis from
[the v6 protocol](STATIC_AL.md). Use raw z128, excluding grouped prediction
heads. Clusters are fitted independently and their IDs are specific to this run.
The current mixed fit excludes static observations, but the analysis is
descriptive and does not establish independence from checkpoint ancestry.

```bash
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_backtracking_static.json --stage export
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_backtracking_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_structural_gatr_backtracking_al.yaml
```

Results are written to
[`output/structural_static/gatr-temporal-backtracking-latest-20260918T1936/`](../../output/structural_static/gatr-temporal-backtracking-latest-20260918T1936/).
The pipeline publishes the gallery and metric definitions after completion.
Verification requires exact checkpoint/input tensors, batch replay and agreement
with an independently loaded, natively compiled copy of this exact encoder on
64 dynamic selection inputs. See [execution details](../../docs/structural_static_analysis.md).
