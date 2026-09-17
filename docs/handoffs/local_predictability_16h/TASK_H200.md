# Receiving task: one H200, one seed, shared 16-hour ceiling

Copy the following task into the H200 session when ready to execute. Preparing
this file does not start the remote task.

---

Implement and run the H200 portion of the new local-predictability study in
PointCloudMaterials. Use one H200 and training seed **20260919 only**. Coordinate
with the local H100 task through exchanged files. The shared wall-clock ceiling
is **16 hours**, first complete report target hour 12, finish earlier if possible.
Record the shared start/deadline before execution; respect any shorter actual
allocation. Stop one hour before the deadline to preserve/export artifacts.

Read AGENTS.md, scripts/README.md, docs/local_predictability_16h.md,
experiments/local_predictability_20260917/README.md and
configs/local_predictability/two_gpu_16h.json. The JSON is a **planning spec**;
required loader/trainer/assay adapters are not implemented yet. Do not feed it to
the old trainer or allocation controller. Do not resume old width or simulation
queues. Optional extensions are disabled.

Your primary deliverable is a matched **physical-mean** native comparison:
current positions/velocities; real 12-ps atomic history; separately trained
repeated-current-frame 12-ps control. Width 16, two blocks, exported state 128,
17 Å observation, 5 Å cutoff. Train every encoder weight. Use current 128-component
physical reconstruction (weight 1) and simple linear future conditional means at
0.75/3/9/24/48/96 ps. No mixture head, whitening, slowness or crystal labels in
training or encoder selection. Crystal prediction is an external frozen-state
readout. H100 owns the supervised-onset comparison and shared assay.

1. Verify the portable 150-source manifest against the already uploaded raw Al
   trajectories. Preserve 90/30/30 folds and the declared validation halves. Report
   missing files precisely; never silently replace the sample with the old 450
   cached windows. Keep remote paths in machine.local.yaml. No bulk data reupload
   should be needed.
2. Own the native batched loader, simple-head and nested-history implementation;
   send its tested patch and hashes to H100 before either worker freezes native
   runs. Reuse maintained producers. Gate the entire extra temporal computation
   so alpha=0 exactly matches the current-state parent, including support and
   normalization effects. Verify outputs/shared gradients and gate learnability.
3. While H100 builds labels/descriptors, prepare the declared raw-window index
   and run small-set fitting on 32 windows from eight training sources. Require
   current standardized MSE <=0.1 and every block <=0.25. Diagnose failure; don't
   spend the remaining allocation on a blind sweep.
4. Use VRAM for bounded immutable input/graph residency and packed batches;
   effective batch eight with accumulation, full frame cadence, unchanged width.
   Time a tiny actual training workload, not the hardware benchmark suite. Report
   p90 update/validation cost and peak VRAM. Agree K=1024/2048/4096 with H100 before
   fits, reserving the full matched group and evaluation. At hour 4, if adapters
   or shared data aren't ready, report the revised feasible remainder.
5. Verify H100's frozen row/target/scaling release. Train one snapshot parent for
   K updates; clone identical weights into all three children, reset optimizers
   consistently and give **each child**, including continued snapshot, K updates.
   Freeze selection by physical validation error only. Keep the same seed and
   source-uniform sampling across children. This is four training stages and
   three final models, not a width sweep.
6. Compare native means to frozen-state ridge and matched nonlinear readouts and
   the input-packet reference on identical rows. Report present/future errors by
   block/horizon, state-mean and condition/time-stratified shuffle interventions,
   and present/future encoder gradient contributions. Use frozen assay readouts
   only after the general encoder checkpoint is fixed. Never claim physical
   unpredictability from a failed fit.
7. Run detached only after the new executable commands, tests, release hashes
   and real deadline are established; use explicit tracking/resume dependencies.
   Preserve source snapshots during live fits. No additional seeds, H48 extensions
   or new simulations. Publish core results immediately rather than filling time.

Return a tested code patch, complete checkpoints/resume state, paired predictions
and row IDs, tables with frozen metric definitions and implementation hashes,
source-bootstrap intervals conditional on this one seed, timing/memory records,
and an explicit incomplete/blocked-work ledger. Save under a fresh
output/local_predictability/ run. A baseline/fit-diagnostic report is the correct
outcome if the complete native comparison cannot meet its gates and deadline.
