# Can selective physical information outperform wider dual-domain histories?

The completed first wave favors a simple relaxed reference and gives a modest,
uncertain benefit from quench differences. It does not support adding dense
history or a second original-encoder branch unchanged. Select the next input
family using **selection** Brier and the physical-error gate, then test missing
ordered-region geometry separately from optimization.

## Optimization:16 one-seed fits

[Configuration](../../configs/crystallization_transfer/literature_optimization_20260922.json).
Retrain cold, quench, quench+rates and original-current-descriptor controls with
a48-epoch cap and patience10. Test LR3e-5 against1e-4; short-horizon auxiliary
weights.25 and1; fully autoregressive training against scheduled teacher forcing;
weight decay1e-3 and1e-2 against1e-4; and head width64 against128. One low-LR
quench fit has a96-epoch cap and patience16. Encoders remain frozen; input sources,
targets, batch128 and seed20260919 remain fixed. Training is allowed to stop early.

These tests distinguish optimization, regularization, capacity and horizon effects.
The fully autoregressive control tests train/inference mismatch; lowering LR and
width tests whether the failed larger input combinations were optimization-limited.
No new dataset or relaxation is needed for this part.

## Ordered-region geometry:12 one-seed fits

[Configuration](../../configs/crystallization_transfer/crystal_front_20260922.json).
Compute descriptors from existing original-MD coordinates at each archived origin
and3 ps earlier. No future-confirmed labels or PTM classifications enter features.
For each atom use12 nearest bonds, each required to be within5 A; form normalized
q6 vectors, count bonds whose real inner product exceeds.7, and mark atoms with
at least7 such bonds as ordered. This is an operational bond-order criterion,
not a claim of an exact crystal/melt surface.

Inside12,20 and25 A balls around each tracked center, calculate ordered fraction,
coherent-bond fraction, nearest ordered-atom distance and an availability flag,
largest and nearest connected-component fractions, mean/std q6, center-to-context
q6 alignment, and invariant directional moments of ordered neighbors. Connectivity
is restricted to each ball; global cluster sizes cannot leak in. Evaluating a node's
coherent bonds needs a maximum10 A geometric halo, so total support reaches22,30
or35 A respectively. This changes predictor context, not the local MACE encoder.

| Fit group | Scientific motivation |
|---|---|
| Cold and quench controls | Separate new information from the changed auxiliary parameter budget. |
| Current descriptors at12/20/25 A | Determine whether nearby ordered regions or broader context explain onset. |
| Quench +25 A | Test complementarity of inherent structure and observed ordering geometry. |
| Repeated-current versus actual3 ps front changes | Test whether measured approach/growth adds information beyond the current region. |
| Quench +3 ps front changes | Test whether local quench sensitivity and context evolution complement each other. |
| Coherence-only versus geometry-only25 A | Distinguish bond-order strength/alignment from location, connectivity and anisotropy. |
| Quench + all three scales | Test multiscale structure without another encoder branch. |

All single-branch auxiliary heads have the same parameter budget within each
campaign; unused feature blocks are masked after training-only normalization.
Current/repeat/rate comparisons are separately trained. Components are truncated
at the observation boundary. Hard coherence thresholds and inherited float16
coordinate precision require sensitivity checks before interpreting descriptors
as a physical mechanism. No new MD or relaxation is produced.

## Evaluation and selection

Retain source splits, natural test prevalence,96 ps forecasts,12 ps Brier checkpoint
selection, paired source bootstrap and timing-with-misses from the first-wave protocol.
This remains exploratory on historical test sources. Promote using selection
Brier subject to the physical-validation tolerance; do not pick hyperparameters
from the displayed test table. Compare each new fit to its new same-campaign control.

Results update at
`output/crystallization_transfer/literature-optimization-20260922/RESULTS.md` and
`output/crystallization_transfer/crystal-front-20260922/RESULTS.md`.
**Update interpretation after both queues finish.** Descriptor preparation and
automatic validated handoff are documented in [operations](../../docs/crystallization_followup.md).
