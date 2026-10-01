# TDA training and VCReg placement/scaling

## Full-data synthesis, 30 September

All seven pilots completed. The last topology-distance arm obtained mean-relative
validation error 0.3651 (TDA 0.3478), versus 0.3455 (TDA 0.3218) for block scaling alone.
The declared automatic winner is `tda-blocks`; its full-data fit started as 1015284.

The user requested a new large run combining the lessons. `MM-TDA-BLOCK-DIRECT-FULL`
retains structured reconstruction and block scales, moves VCReg onto the exported
state and uses the original D(D-1) covariance denominator. It retains variance .05,
covariance .01, mean anchoring .01, five-epoch ramps, batch 8192 and 20 epochs. The
topology-distance term remains off. Direct/original-strength VCReg gave the best
reconstruction of the four matched regularization pilots; block scaling improved
the separate projector-based TDA comparison. Their combination has not yet been
validated. Low effective rank is not by itself proof of information loss, and this
run does not promise to eliminate it. Evaluate reconstruction and rank jointly.

The already-running automatic winner continues as the full-data reference. The
new fit uses the identical rows, seed, architecture and LR schedule; only VCReg
placement and normalization differ. This joint change is not an isolated estimate
of either regularization effect. Both checkpoints use label-free validation NLL.
No test score was used to choose the new combination.

Question: does global row mixing and a topology-aware training objective improve
descriptor retention and the accessibility of structure in exported MACE states?

Recipe: [seven-arm study](../../configs/liquid_predictability/rich_tda_pilots_20260930.json).
Frozen calculations and input contract: [metric definitions](../../docs/metrics/rich_tda_objectives.md).
Operation: [workflow](../../docs/rich_tda_objectives.md).

| Arm | VCReg location | Covariance denominator | Descriptor treatment |
| --- | --- | --- | --- |
| embedding-pair | Exported state | D(D-1) | RH2 reconstruction |
| projector-pair | Projector | D(D-1) | RH2 reconstruction |
| embedding-channel | Exported state | D | RH2 reconstruction |
| projector-channel | Projector | D | RH2 reconstruction |
| tda-heads | Projector | D | Semantic heads and block balancing |
| tda-blocks | Projector | D | Above, common scale per block |
| tda-distance | Projector | D | Above, topology-distance supervision |

Every pilot uses the same nested 105,677 raw fitting patches, same source roles,
one seed, global batch 8192 and 20 epochs. The full fit uses the unchanged 1,056,768
RH2 rows and 20 epochs. Peak LR .01 remains fixed so this study isolates objective
choices, not LR tuning. All pilot methods are selected by common validation
descriptor NLL; test remains untouched until full fitting finishes.

The 2x2 controls isolate VCReg placement and denominator effects. The three TDA
arms form an incremental comparison. The first TDA treatment changes both the
head and semantic weighting. It is not an isolated architecture ablation.
The previous 60-epoch shard-shuffled RH2 is historical context, not a matched
causal test of the new sampler. A one-seed, short-update pilot establishes which
method won this screening protocol, not a universal optimum.

Primary endpoint: held-out final descriptor error relative to the fixed training
mean, overall and per family. Diagnostics: per-feature failures, TDA subgroups,
exported/projector effective rank and scalar regularization/task gradient balance.
Crystallization utility is a separate downstream question; reconstruction skill
alone does not establish precursor information.
