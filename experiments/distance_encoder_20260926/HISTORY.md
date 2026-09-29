# Can preceding MD observations improve present crystal-front distance?

Train **CD-MACE128-H6** end to end on three shared-MACE embeddings at -6/-3/0 ps,
tracking the same atom. An ordered residual MLP combines embeddings and adjacent
differences, followed by the current-distance distribution head. Warm-start the
completed CD-MACE128 encoder/head; temporal fusion is new. Fine-tune all parameters.

Compare to **CD-MACE128-H6-repeat**, the same trainable architecture receiving
three copies of the current embedding, with identical eligible rows, initialization,
seed, optimizer and 12 epochs. Replay the original snapshot model on the exact
evaluation geometry as a historical reference. The repeated control distinguishes
historical information from additional parameters and fine-tuning time; its
redundant deterministic encoder calls are eliminated, not its gradient paths.

Use all eligible retained Al/Mg/Ti/Ta dynamic neighborhoods. Short history is
6 ps because the existing large coordinate bank has exact 3-ps cadence. Omit
the first two retained label frames per source equally in both arms. Preserve
Al64 all64 source roles and all original evaluation samples/path endpoints.

Keep the previous censored distance likelihood plus early-radius proper scores,
with likelihood-only selection after 12 epochs. No AP optimization, physical
reconstruction, regularization change or input condition covariates. Train with
one seed, global batch 1024, 512 sequences per GPU, using two GPUs.

Primary evidence: held-out distance NLL/Brier and spatial warning at probabilities
>.5/.75/.95, with missed paths and false alarms. Report current and historical
crystal visibility so memory of a previously visible crystal is distinguished
from evidence about an unseen front. Two-position alarms are primary. Onset AP
is not the research target here. Al held-out results cannot establish multi-material
generalization; external training branches share preparation ancestry.

[Metric definitions](../../docs/metrics/distance_encoder_history.md) ·
[Recipe](../../configs/distance_encoder/md_history6_20260926.json) ·
[Execution](../../docs/distance_encoder_history.md)

The audited eligible release contains **11,375,472 training sequences**
(Al 7,229,416; Mg 377,496; Ti 707,000; Ta 3,061,560) and
**190,080 Al selection sequences**. There are 117 train sources and 15 selection
sources; external branch counts do not imply independent ancestry.
