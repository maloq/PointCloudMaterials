# Matched relaxed Ni response experiment

User choice, October9: continue the original MD state and relax only encoder
observations/targets. The completed epoch12 relaxed eSEN initializes all fits.
The previous unrelaxed campaign remains unchanged and supplies the matched rows.

Recipe: `configs/ni_response/esen_relaxed_original_md_20261009.json`.
Module: `src.research.ni_relaxed_response.workflow`.
Results: `${storage:training_storage}/ni_response/relaxed-esen-original-md-20261009-v1`.

Stages: indexed native-frame quenches; matched signed100fs original-state MD
followed by full-cell quenches; two-lane frozen target preparation; four1600-
update response/value fits; frozen first-appearance/32/64-atom probes; paired
comparison against the unrelaxed campaign. Definition of a default nucleus
remains64 atoms sustained1.5ps. Original labels and held-out pooling are retained.

The1930 requested native frames include310 exact-recipe shared-index candidates.
Remaining cells must be computed; birth-window-only availability never determines
the new training population. No pre-quantization local patch is replaced by a
float16 full-cell extraction. Physical trajectories never start from a quenched
configuration combined with hot velocities.

`submit --local` starts two workers on the current two idle GPUs and submits
one continuation per worker after the current allocation, using alternative GPU
partitions. Scientific checkpoints resume; completed quench and target packets
are reused. Each worker requeues only its receipt-bound job if more time is needed.
CPU evaluation depends on both lanes; a paired comparison depends on both
relaxed and unrelaxed evaluations. No unrelated job is stopped or modified.

[Precise mathematical and evaluation definitions](metrics/ni_relaxed_response.md).

October9 execution receipt: `technical/launch.json`. The relaxed checkpoint
finished12 epochs/62784 updates, SHA256
`09e1b31dddfdf0b71243c4cef59b323879f900d2e0eba44384509b9f95fc4826`.
Two local workers were launched on node58's GPUs0/1, with continuation jobs
1030833/1030834 after allocation1029087. Frozen evaluation1030835 and paired
comparison1030836 are queued behind their required training/evaluation jobs.

At14:47 CEST, SCRATCH filesystem operations were stalling even for directory
`stat`. The first native quench had converged and its native result was published
to the shared index, but verified float16 conversion stalled on NFS. Its partial
files are retained. The stalled diagnostic producer and its idle callback child
were stopped by their exact current-turn process identities; other jobs were
untouched. The local workers explicitly wait for SCRATCH availability, then
resume the preserved quench and run the full response/gradient and repeated-
quench diagnostic before scientific fitting. This is queued execution; no relaxed
motion-fit updates or prediction results are claimed at this stage.

At16:27 CEST, the storage gate had cleared, exposing a dump-reader precision
failure on the first perturbed initial cell: float64 bounds were rounded to
float32 before a 1e-10-angstrom identity check. Reading those bounds in float64
verified the original converged32000-atom result without changing its state or
relaxation settings. The corrected reader is frozen in `technical/code-box64`;
the original bundle and launch receipts are preserved in `technical/code` and
`technical/recovery-box64`. Only this campaign's waiting local worker and four
pending jobs were replaced. Current continuation jobs are1031193/1031194,
evaluation1031195 and comparison1031196; `technical/launch.json` is authoritative.
The resumed workers must complete the full response/gradient probe before fitting.

The resumed probe completed with finite gradients, and both lanes entered native
quench preparation. Its independent same-endpoint quench revealed a scientific
concern: identical input hashes and atom membership produced local coordinate
component RMS difference0.364 angstrom and embedding RMS difference1.177. Teacher
response RMS was0.716 for the position direction and0.999 for the velocity
direction. Both minima met the0.01-eV/angstrom convergence criterion, but their
total energies differed by2.315eV across32000 atoms. This is one repeated pair,
not a population noise estimate. The cause is not established; finite gradients
do not establish useful response supervision. Preparation continues with this
limitation recorded in `technical/repeatability-concern.json`; relaxed-response
benefit must not be inferred without signal/noise assessment and evaluation.

At21:40 CEST, all1930 native quenches were complete and79/100 paired response
groups were complete. Lane0 had stopped on one endpoint at the20000-iteration
limit (fmax0.106eV/angstrom versus required0.01); lane1 had finished its50 groups
and was waiting. The failed working directory and prior launch are preserved
under `technical/recovery-node60`. Both lanes were resumed on node60 under
allocation1031286 with identical settings and preserved complete packets. Only
this campaign's waiting remote lane and dependent pending jobs were replaced.
Current continuations1031900/1031901 follow that allocation; evaluation1031902
and comparison1031903 follow completion. A receipt-bound local watcher releases
evaluation early when both local lanes finish and cancels their then-unneeded
pending continuations, so completed training need not wait for allocation expiry.
The single-pair quench-repeatability concern above remains unresolved.

The later [controlled repeatability study](ni_relaxation_repeatability.md)
identified GPU neighbor summation order as a removable source of identical-input
variability. A private sorted-neighbor float32 variant gives exact repeated native
coordinates on all5 checked endpoints. It still shows strong sensitivity to tiny
input changes, including a float64-force reference. The completed89-quench assay
does not modify this campaign's existing frozen targets, checkpoints or recipe.
