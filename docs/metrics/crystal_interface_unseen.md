# Dense non-visible-interface VCReg experiment

This extends [the interface target](crystal_interface.md) with an explicitly
requested population restriction. A query is eligible only if **none of the 25
input patches contains an interface atom with patch-relative radius <8 A**. This
matches the actual encoder graph producer: nodes at radius >=8 have zero atom
attributes and pooling weight, and no allowed message-passing edges. Context
placement uses distance-only covering and no labels. Only coordinates, patch
indices and relative patch positions enter forward(); distance/direction,
visibility, phase, frame, source and role are label/audit fields. Exclusion changes
the learning population; it is not a deployable interface-absence oracle input.

Expand the 150 existing Al trajectories using 64 evenly spaced integer frame
indices across each 801-frame timeline and 256 uniformly sampled atom identities
per selected frame. Draw without replacement, excluding previously sampled
uniform centers on that frame, using SeedSequence([20260928,source]). Candidate
selection does not inspect distance or phase. This gives 2,457,600 new candidates;
cache new candidates only when the interface is invisible. Keep every original
interface-dataset row, geometry, source role and scan path. New candidate counts
before visibility rejection are retained per source, frame and population.
No simulations or new source splits are introduced. Existing whole-source roles
remain 90/15/15/30. Sampling more centers does not create independent trajectories.

For each role, the pre-exclusion reference population is half fixed-at-risk and
half uniform centers, with equal source mass within each half. Uniform counts
include all newly proposed candidates, even those rejected before geometry
caching. Condition those probabilities on invisibility and renormalize once.
This preserves the intended conditional distribution, rather than accidentally
upweighting sources with very few eligible centers. No distance quotas are used.
All training draws have unit loss weight; validation uses the same conditional
weights. Calibration/test never enter fitting or checkpoint selection.

One distance+direction+VCReg fit starts from the same original supervised snapshot
parent as the preceding experiment, with fresh context head and optimizer. The
parent historically saw visible examples on its recorded training ancestors;
only this adaptation is restricted to invisible examples. The old interface model
is a frozen reference, not the initializer. No temperature/time/species inputs.

Global batch and microbatch are 512, split 256/256 across two GPUs. One nominal
epoch is 512 independent replacement updates; 16 blocks total 8192 updates and
4,194,304 draws. This is a declared expanded budget, not 16 exhaustive passes over
the new dataset. Select among blocks 12–16 by the conditional-population predictive
likelihood, excluding VCReg, exactly as defined in the interface objective.

VCReg means and centered covariance sufficient statistics use differentiable SUM
all-reduces across ranks. The summed adjoints plus averaged parameter gradients
give the same global-batch objective as a single process; do not average separately
computed per-GPU VCReg penalties. Both ranks use the same global sampler RNG and
take disjoint rank-strided halves of each draw. All parameter gradients are packed,
summed and divided by world size before clipping/AdamW; parameters unused on every
rank remain grad=None. The encoder's compiled patch operations stay local. Rank 0
alone writes checkpoints, logs and the single online W&B training run. Validation
is sharded across GPUs and its weighted sufficient statistics are summed.
See [PyTorch's gradient averaging semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
and [autograd-enabled collectives](https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py).

All cached rows retain predictions for transparent secondary comparisons. Primary
`unseen-distance/direction/reliability.csv` report only interface-invisible rows.
The original fixed liquid-at-risk track is intact; newly expanded uniform rows
are named separately. Original visible rows are diagnostic only. No row is silently
removed from the fixed benchmark. Rank tables add explicitly named unseen groups.
Physical readouts fit only eligible training rows. Temporal/noise checks are
anchored at eligible fixed test queries; the previous snapshot or perturbed context
may change visibility, so these are response diagnostics, not a hidden-interface
trajectory assay. Such label-side diagnostics do not train the encoder.

Primary `unseen-alarms.csv` truncates each original approach path **before its first
visible interface query or first crystal entry**. Two consecutive original query
positions must exceed the probability threshold. Never delete visible positions
and concatenate separated clear segments. Keep all original paths in the recall
denominator; also report the count with at least two eligible observations. Misses
remain misses. Warning distances are conditional on detection. The field
interface_visible_at_alarm is zero by construction, not a fitted performance score.

`paired-original-unseen-distance/direction/reliability.csv` evaluates the frozen
previous VCReg model and new adaptation on exactly the original interface-invisible
rows. Source-local parent indices pair records; identity, frame, atom, population,
path, role, distance and visibility must match, and every original invisible row
must be present exactly once. Verify historical prediction and checkpoint hashes.
Use the same per-population macro-source calculations as `point_tables`; keep
expanded uniform rows out of this comparison. This is a paired benchmark, not a
source-bootstrap confidence interval or a new fitted readout.

The companion overfitting audit keeps historical predictions and metric contracts
unchanged and runs locally; it creates no W&B diagnostic runs.
