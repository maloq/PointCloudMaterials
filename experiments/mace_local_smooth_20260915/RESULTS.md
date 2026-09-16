# First frozen-state sweep: completed findings

**No compact candidate passed the information-retention gate on validation. No candidate reached a 0.10 normalized RMS jump.** The current encoder has not been replaced.

The sweep fitted 30 compact nonlinear maps plus two uncompressed references. Each used 300 full-batch epochs; compact maps were newly initialized, while reference readouts started from the existing trained head. The selected physics-only models all reached the last epoch, so this is not evidence that compression fundamentally cannot work.

## Explained comparison

Jumps measure local-state change over **0.75 ps**, divided by the representation’s training-reference spread. The disordered column restricts both endpoints and the reference population to local bond order qbar6 < 0.3; it is a low-order proxy, not a definitive liquid label. Physical error is the worst error ratio across bond order and three instantaneous-topology families, both overall and within this low-order population. A ratio of 1.10 is the largest allowed increase; 3.72 means 3.72 times the reference error in the worst family.

These illustrative variants are chosen from validation: the 32D physics-only model has the smallest worst physical-error ratio, weight 3 has the smallest low-order jump, and weight 10 has the smallest overall jump. None qualifies as an accepted model.

| Model | Split | Overall RMS jump | Disordered RMS jump | Worst physical error / reference |
| --- | --- | ---: | ---: | ---: |
| Current structural block | validation | 0.438 | 0.700 | 1.00× |
| Current structural block | development_test | 0.461 | 0.742 | 1.00× |
| 32D, physics only | validation | 0.685 | 0.873 | 1.24× |
| 32D, physics only | development_test | 0.720 | 0.905 | 1.26× |
| 8D, temporal weight 3 | validation | 0.220 | 0.373 | 3.61× |
| 8D, temporal weight 3 | development_test | 0.265 | 0.717 | 3.72× |
| 8D, temporal weight 10 | validation | 0.185 | 0.413 | 3.97× |
| 8D, temporal weight 10 | development_test | 0.227 | 0.766 | 3.93× |

## Interpretation and next experiment

Strong temporal penalties can make the overall score look better without stable, informative disordered-state coordinates. The weight-3 example improves validation disordered jump from 0.700 to 0.373, but its development-test value is 0.717 versus 0.742 for the reference, with a 3.72× worst physical error. The largest test jumps also become worse for several penalized models. This generalization gap deserves source-level follow-up; these data do not establish an irreducible physical limit.

The next controlled sweep uses 32/64 coordinates, weaker penalties 0/0.03/0.1/0.3/1, and 2,000 epochs with two seeds. All variants and the reference receive the longer budget. The hidden map has width 64 at its narrowest hidden layer, so 64 is a meaningful larger-state control. This separates optimization/compression problems from strong smoothing before adding trajectory curvature or history. Submitted detached as Slurm job **994010** with the existing implementation and a separate output directory.

Recipe: [capacity comparison](../../configs/analysis/mace_local_smooth_capacity.json). Output: `output/mace_local_smooth/velocity-frozen-capacity-20260915/`.

## Evidence and limits

- Same checkpoint, 1,125 source records, 13,500 tracked pairs, source-level splits and training-reference identities as the existing velocity experiment. Frozen feature replay relative error: 2.54e-6.
- All measured-velocity data in this source experiment are Al; no Zr result is implied.
- The old test split is development evidence because it already informed the research; no new blind generalization claim.
- Physical scores use jointly trained readouts, not independently fitted probes. No confidence intervals or independent physical-observable validation are included yet.
- MACE is frozen; the learned maps receive its coordinate-derived structural block. This stage does not yet use history, add a velocity-dependent structural state, constrain curvature, or establish a low-dimensional temporal manifold.
- Verification: 14 focused tests passed; metric documentation contracts passed for all 15 families.

Full numerical tables and archived definitions: `output/mace_local_smooth/velocity-frozen-20260915/tables/`. Plot: `plots/smoothness_information.png` in that run.

Definitions: [normalized RMS jump](../../docs/research_glossary.md#normalized-rms-jump), [direct temporal regularization](../../docs/research_glossary.md#direct-temporal-regularization), and [protocol calculations](../../docs/metrics/mace_local_smooth.md).
