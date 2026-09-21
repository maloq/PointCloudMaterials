# Independent-root BCR conditioning curves

Inherits noise loss, coordinate units, shell boundaries, structural targets and
paired-root bootstrap definitions from `bcr.md`. Noise keys and anchor order are
identical across arms/checkpoints. d0 and n_ref come only from12 training roots.
The frozen evaluation bank samples64 anchors from each of six development roots.

Absolute R_C, R_U, R_R and matched R_S are corruption-averaged per anchor, then
averaged within root with equal weight across roots. G=(R_other-R_C)/max(R_other,
1e-30). Negative gains remain negative. Four strict matched derangements per level
must use another root with identical temperature and training-fitted count/density/
q6 quintile bins. Unmatched examples are excluded from both members of paired
comparisons; coverage is shown. One thousand paired bootstrap draws resample roots,
not noise instances or atoms. Intervals exclude training-seed uncertainty. All-
population and q6<0.35 liquid diagnostics are separate; q6 is a fixed current-
structure measurement and is never a BCR objective or input.

The pilot G1 pass rule requires >=2 levels among .04/.08/.12 with liquid mean G_U
>=.05, lower95% bounds >0 over independently trained unconditional/frozen-random
controls and strict swaps, with >=50% strict coverage over >=4 roots. This is a
predeclared engineering criterion. It does not certify G2/G3 or crystallization skill.

Structural probe scaling/weights fit ten encoder-training roots; ridge penalties
use the remaining two training roots. All six development roots remain prediction
observations, never tuning observations. Small MLP follows the fixed BCR500-update
budget. Within-liquid diagnostics retain the same fitted readouts and targets.
Per-root errors accompany pooled standardized RMSE and per-target R2.

Relative update diagnostics every50 optimizer steps are Euclidean parameter-change
norm divided by the pre-update parameter norm, separately across encoder/decoder
parameters. Frozen encoder updates are exactly zero; these diagnostics are not
optimization targets. Curves retain each predeclared checkpoint; reconstruction
alone does not choose a 'best' code.


Table export: 2026-09-21T14:29:50.816497+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
