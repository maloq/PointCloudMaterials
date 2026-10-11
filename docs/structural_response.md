# Broad structural starts for derivative motion

## First crystalline appearance, October 7

The new `first_appearance_20261007.json` recipe changes the prediction question:
given a currently crystal-free local observation, will any of its atoms first
become PTM-crystalline within 0.1/0.5/1/3/6 ps? A transient appearance counts.
No catalogue founder, eight-atom embryo, 64-atom size or eventual-establishment
condition selects the positive examples. The old unfiltered birth assay remains
unchanged. See [the exact target and metrics](metrics/motion_first_appearance.md).

The pinned consolidated dataset contributes all 114 velocity-bearing source
views across 106 recorded ancestors. Four fixed observation times and 128
uniformly drawn centers give 456 windows and 58,368 candidates before present
crystallinity and exact-identity checks. All candidates are drawn before future
labels are examined. This assay has an observed event prevalence, not four
negatives forced per positive. The original roles are retained; selection,
calibration and test are pooled only for evaluation. Eight original training
ancestors supply internal NLL selection. No explicit conditions enter readouts.

PTM is recomputed on original full periodic cells at every exact 0.1 ps frame;
existing component caches lose single crystalline atoms and cannot supply this
target. The consumed present atom identities are followed for 6.2 ps. A separate
same-atom, three-frame persistence label and a past-0.5-ps-clear diagnostic
subgroup distinguish brief appearances and recent recurrent ordering. The
primary eligibility rule uses only present crystal absence, with no external
neighborhood crystallinity, component-size or distance screening.

Compare all eight completed motion fits, both completed 12-epoch spatial starts,
qbar6, bond-order/TDA controls and instantaneous velocity summaries. Frozen
structural, motion-only and combined-code CatBoost probes select iterations by
training-only NLL. This does not start new encoder training or simulations and
does not load the broad spatial collection. The GPU produces frozen embeddings;
full-cell PTM and frozen diagnostic readouts use CPUs.

In `pointnet-torch214`:

```bash
python -m src.research.structural_response.appearance launch \
  --config configs/structural_response/first_appearance_20261007.json
```

The detached launch freezes source and metric definitions under
`${storage:training_storage}/structural_response/first-appearance-20261007-v1/technical/code`.
Its launch receipt identifies only this chat's process. Collection resumes by
verified window; retained readouts resume by model/space/target/horizon. Cache
inputs are registered as `al-first-crystal-010-20261007`. Intermediate status is
in `technical/collection-progress.json` and `technical/coordinator.json`; final
coverage, predictive scores, calibration and the scientific plot are grouped
under `tables/` and `plots/`, with `report.html` as the entry point.

The launch on node60 allocation 1026285 owns process 3479019. The first-window
label enumeration and actual spatial/motion GPU execution passed; numerical
receipts are in the run's technical folder. A separate owned publication process
3479729 waits for completed exported metrics, then updates the existing ten
training run IDs through `appearance_publish` without creating online probe runs.
Its frozen publication source, launch and failures are recorded separately.

This targets the next observed appearance after a crystal-free present, not an
atom's first-ever ordering over all unobserved history. Its horizon AP is not
directly comparable to AP at a lead before a retrospectively selected nucleus.

The user also requested these labels as a permanent supplement to
`al-nuclei-birth-010`. A single export is triggered after all label windows and
the cohort are sealed, independently of model evaluation. Its separate reader,
target semantics and storage are documented in the
[additional-label guide](datasets/nuclei_birth_labels.md). The experiment's
frozen producer and original nucleus labels remain unchanged.

## Completed motion fits and birth feature audit

All eight v4 fits completed 1,600 updates and their frozen birth evaluations.
The last VICReg value-joint fit resumed from update 413 on node60 with its original
microbatch 32, optimizer, RNG and frozen source. The unused owned successor job
1026802 was cancelled only after all eight completion receipts were verified.
See `technical/final-control-migration.json` and
`technical/training-campaign-complete.json` in the v4 output.

The user's high-AP audit preserves the exact unfiltered cohort. Recipes are
`configs/structural_response/birth_feature_audit_20261007.json` for CPU controls
and physical explanations, and `birth_embedding_audit_20261007_v3.json` for the
completed frozen embedding interventions. The output is
`structural_response/c3-l3-fast-20261007-feature-audit-v1` at training storage;
`report.html`, `plots/feature-audit.png` and grouped metric tables contain the
results. All 14,715 Q,V observations were reconstructed exactly; ancestry,
same-frame controls, exact coordinate overlaps and pretraining ancestry were
audited. No neighborhood filtering, relabeling or original metric replacement
was performed. Descriptor controls and surrogate explanations ran locally.

Frozen CPU producers are in `technical/code`. The two successive GPU diagnostic
exports retain their own producers under `analyses/*/technical/code` and their
own metric snapshots. The partial future-law v2 diagnostic stopped at an adapter
numerical-replay tolerance; it is not presented as a completed attribution.
V3 covers all eight models' structural and structural+motion readouts at 3 ps.
The current publication-only renderer is
`python -m src.research.structural_response.birth_audit_report --config
configs/structural_response/birth_feature_audit_20261007.json`.
See [metric definitions](metrics/motion_birth_attribution.md) and the scientific
findings in the experiment record.

## Fast motion campaign, October 7

Active recipe: `configs/structural_response/motion_fast_20261007_v4.json`.
It reuses the two completed 12-epoch spatial checkpoints and the sealed analytic
v2 MEAM teacher targets. Eight 1,600-update fits compare full-MACE joint adaptation
with a trainable residual adapter on cached MACE codes, each with and without
short derivative supervision. Student derivatives use centered float32 differences;
the physical teacher remains analytic. Full-batch latent gradient computation
replays encoder graphs in 64/32-row pieces. No broad structural replay is opened.
See the [v4 numerical definitions](metrics/structural_response.md).

Birth prediction uses `al-nuclei-birth-010` with **no additional neighborhood
filter**, as clarified by the user. Preparation retained 4,905 matched rows:
161 training physical births and 79 pooled-test physical births across 33 test
ancestors. There were 253 catalogue event records on velocity-bearing fixed-role
sources: four lack all required exact lead observations; 249 available records reduce
to 240 physical births. Fifteen duplicate positive/control pairs were removed.
Every retained case has exact 1/3/6 ps Q,V inputs. Controls are same-source random
nonbirth founder IDs, not certified continuously liquid neighborhoods. The
historical screened benchmark remains distinct.

The prepared input dataset is `al-motion-birth-010-nofilter-20261007`. Per-fit
birth readouts run automatically after future-NLL checkpoint selection, compare
structural, structural+motion and future-law features, and export AP/AUROC/NLL/
Brier with ancestor bootstrap intervals. Logistic/CatBoost probes stay local;
final metrics join the existing online scientific training run.

In `pointnet-torch214`, with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, use the existing
module `python -m src.research.structural_response.workflow` and recipe above:
`birth-inputs`, then `probe`, then `launch`. The probe records complete-update
timings, memory, secant/analytic comparisons and adapter checks without W&B.
Separate RTX6000PRO/H100, A100, A40 and L40S workers lease unfinished fits;
the latter three use half microbatch with full 256-sample updates. Node52 is
excluded following the documented CUDA initialization failures. The current
allocation starts a detached local worker, with an owned successor reservation.
The frozen source, ownership, jobs and actual progress live under
`structural_response/c3-l3-fast-20261007-v4/technical/` at training storage.

Launched October 7 at 13:17 Paris. The detached local worker is PID 3427000
inside the existing node60 allocation 1026285. Jobs 1026802–1026809 are recorded
in `technical/launch.json`; 1026802 is the local allocation's successor, while
1026806 started on node58. The A100, A40 and L40S lanes are separate reservations.
No unrelated allocation or simulation was changed. Read live progress receipts
for subsequent status, rather than treating this launch observation as current.

At 13:21 Paris, the user clarified that GPU types are alternatives. Redundant
pending A100/A40 jobs 1026803/1026804/1026807/1026808 were cancelled after checking
their launch ownership and pending state. The local/node58 joint fits and the
two running L40S workers on node39/node57 continue. The launch recipe is preserved;
`technical/gpu-pool-adjustment.json` records this operational change. Future
requests should use a combined compatible-partition list per required worker.
Both joint derivative fits reached ten optimizer updates, with their tenth
updates taking 3.74 and 3.62 seconds including checkpoint writes. This establishes
that scientific fitting started, not that a trained-model improvement was found.

All eight numerical probe arms passed. Across eight checked states, the centered
velocity derivative's relative error against exact AD was 0.042–0.075%.
Complete optimizer-update samples on RTX PRO 6000 Blackwell took 3.7–9.9 seconds
for warmed joint derivative steps, about 1.65 seconds for joint value steps,
and 0.14–0.16 seconds for cached-adapter steps. These short timings exclude cache
preparation, checkpoint I/O, validation and birth readouts; they are not completed
training results. The probe and frozen implementations are retained with the run.

At 14:39 Paris, both derivative adapters had completed all 1,600 updates but their
workers had failed during metric export: the bundle omitted
`configs/data/nuclei_birth.json`, a pinned metric-contract dependency. Checkpoints
and future predictions were intact. At 14:40, the missing metadata file was added
with the exact hash required by the original frozen contract; all 32 dependencies
then passed verification. No numerical code, checkpoint, data selection or metric
definition was replaced. See `technical/bundle-metadata-repair.json`.

Continuation jobs 1027017/1027018 were submitted through a single combined
`A100,A40,L40S` partition list per method and started on A100 nodes 06/54.
They resume the completed adapter checkpoints and then take remaining controls;
the two existing joint fits continue without interruption. See
`technical/metadata-repair-resume.json`. The repository launcher now copies all
metric dependencies outside its source/documentation directories and validates
the finished bundle before submission, preventing this packaging omission in
future revisions. The active run retains its original frozen source.

At 15:43 Paris, seven fits and their birth evaluations were complete. The sole
remaining VICReg value-joint control was running on A100 job 1027017 while the
local RTX was idle. Its saved checkpoint was verified at update 413; only that
owned job was cancelled, and detached PID 3449232 resumed on node60/GPU0 with
the same microbatch 32, optimizer/RNG state and frozen code. The launch receipt
and `technical/final-control-migration.json` record ownership and continuation.
Update 420 completed in 2.26 seconds, versus about 6.24 seconds on the A100.

## October 7 data revision

`configs/structural_response/motion_20261007_v3.json` prepares the next motion
revision with the completed spatial starts and **no broad structural replay**.
Both replay weight and interval are zero; the training implementation does not
construct the spatial replay model or open the packed million-row collection.
Motion fitting still uses the separate11,200 saved Q,V observations (6,080 fit,
1,024 inner selection,4,096 pooled held-out) and176 existing physical teachers.
The million-row collection remains provenance for spatial initialization only.
The short sampler now reads source/ancestor metadata from the physical bank,
checking job order against cached targets; this fixes the missing `source` field
that stopped v2 before its first optimizer update.

This superseded v3 recipe first pinned the consolidated `al-nuclei-birth-010`
source/event contract. Its proposed neighborhood screen was removed in v4;
use the active recipe above for new preparation.

The initial pin contains122 catalogue source views and295 established PTM event
records.114 velocity-bearing views from106 ancestors retain253 candidate event
records:169 original-training and84 pooled selection/calibration/test. These are
**before** exact1/3/6ps lead extraction, the crystal-free neighborhood screen,
matched liquid controls and physical event/view deduplication. They are not final
prediction-case counts. Eight ineligible views and their reasons are recorded.
The separate present-selected embryo-establishment target remains distinct.
Pin, sources, original roles and stable event IDs live in
`technical/nuclei-birth-dataset.json` and `technical/birth-cohort.json`.

This data revision is prepared, not launched. The derivative acceleration
proposal still requires measured full-update timings and numerical comparison.
The old v2 execution bundle retains its original replay settings and results.
Its remaining owned workers were cancelled when this data constraint changed;
`technical/data-revision-retirement.json` records ownership and exact job IDs.
Completed targets and partial feature packets are preserved for reuse. The
containing interactive allocation and other chats' jobs were not changed.

## Motion restart on October 7

Both v1 spatial fits completed all 12 epochs. The v1 motion numerical preparation
failed with CUDA OOM; none of its eight scientific motion fits started.
`configs/structural_response/motion_20261007_v2.json` defines a separate motion
revision, `structural_response/c3-l3-bf16-20261007-motion-v2`. It binds both existing
spatial checkpoint hashes and never restarts their pretraining or rewrites v1.
The eight comparisons and their 1,600-update budgets are preserved.

The repaired run keeps exact reverse-over-reverse AD through ordinary PyTorch
contractions and reduces graph microbatches. The forward-AD candidate was rejected
after a mismatch in its numerical comparison; it supplies no scientific targets. Large GPU joint microbatch is8; A100/A40/L40S use4, accumulated to256.
Joint derivative fits require at least64GiB actual device memory; smaller cards
run both frozen controls and the joint value control. Nested short-response
gradients and the densest observed graphs must pass the
local numerical probe before launch. Physical MEAM banks retain their recorded
float64 integration and float32 forces. No simulator rerun is needed.

Shared fixed-teacher and frozen-code caches are resumable, checksum-bound, and
limited to six entries with process leases. Per-fit `progress.json` distinguishes
teacher preparation from optimizer updates. Joint fits checkpoint every update;
workers reserve time based on measured update duration and can requeue themselves.
Only jobs in this revision's launch receipt may be requeued by these workers.
One queued successor follows the current local allocation; other requested GPU
partitions can start independently. Online W&B starts with scientific fitting.

From conda `pointnet-torch214`, set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the
installed e3nn constants loader, then run the existing workflow module with stages
`probe`, then `launch`, and config `configs/structural_response/motion_20261007_v2.json`.
The launch command starts a detached local motion worker and records the queued
GPU workers. Actual job IDs, process IDs and the frozen source are in this revision's
`technical/launch.json`; output from `squeue` and per-fit receipts determine status.

The validated restart launched on October7 at11:35 Paris: detached worker3410269
uses the current node60 allocation1026285. Queued jobs are1026447–1026449,
1026451–1026453 and L40S replacements1026464/1026465. Jobs1026450/1026454 failed
CUDA initialization on node52 before fitting; their replacements exclude that
host. The launch receipt preserves both attempts. The eight old v1 motion jobs
were cancelled only after the replacement launch; no unrelated jobs were changed.
At11:39, both joint derivative workers were producing shared tangent targets,
with the two A40 frozen workers waiting on those shared caches. Optimizer updates
had not begun at that observation. Later status must be read from live receipts.

## Original spatial campaign

The October 7 workflow trains two reusable structural checkpoints for 12 full
passes through the existing 1,056,768-row packed unrelaxed Al/Mg/Ti/Ta collection.
Each initializes four matched 1,600-update motion fits: value-only or analytic
response derivative supervision, with the structural encoder frozen or jointly
adapted. Scientific definitions: [metric contract](metrics/structural_response.md).

Recipe: `configs/structural_response/overnight_20261007.json`.
Implementation: `src/research/structural_response/`.
Use conda `pointnet-torch214` and module
`python -m src.research.structural_response.workflow` with `--config` pointing at
the recipe. Stages are `prepare`, `probe`, `reserve`, `start-local`, `launch`, and internal `worker`.
Preparation checks immutable input hashes and original ancestor roles. The probe
uses real observations, runs locally without W&B, checks precision, rotation
invariance, derivative backpropagation and effective-batch gradient computation.
Scientific training uses online W&B with stable resumable IDs.

Outputs follow `output/<question>/<run>/` semantics at the registered training
storage root: `structural_response/c3-l3-bf16-20261007-v1`. Each scientific fit has
its own `analyses/<fit>/technical/` checkpoints, progress, W&B receipt and completion
record. Tables carry the frozen metric definition and implementation hashes.
The shared `technical/launch.json` records every owned process and Slurm job.
The frozen execution bundle and recipe live in `technical/code`.

The current allocation performs numerical preparation. One queued GPU per
spatial method starts the full fitting stage after the current allocation, or
resumes its optimizer state if a checkpoint is already available.
Motion workers depend on completed spatial jobs; RTX6000PRO/H100, A100, A40 and
L40S are separate requests. The latter three use half microbatch with accumulated
updates, preserving effective batch 256. Joint motion geometry uses FP32 input
JVPs with ordinary PyTorch contractions at microbatch 16/8; latent-only frozen
training and spatial replay use 256/128. Structural pretraining uses 256. File leases prevent duplicate fits.
Owned batch workers checkpoint and requeue themselves before their allocation
expires if a fit remains incomplete. They never requeue or terminate the containing
interactive allocation. No unrelated training or MD jobs are changed.

Data reuse is explicit: packed spatial rows include Al geometry without velocities;
motion uses only the existing verified-potential velocity-bearing cohort. Its
11,200 observations include 6,080 fitting, 1,024 inner training-ancestor selection,
and 4,096 pooled original validation/calibration/test rows. True future coordinate
neighborhoods at 0.1/0.5/3/12 ps are extracted without interpolating snapshots.
The 176 saved full-cell 100 fs MEAM response banks are reused, not resimulated.
The new derived cache adds no independent trajectories or atoms. The separate
embryo-establishment cohort and new production birth search remain separate.

Status must come from the launch, per-fit progress, completion and Slurm receipts;
configuration alone is not evidence of a running or completed experiment.

Reusable exports are
`analyses/spatial-vicreg_descriptors/technical/best.pt` and
`analyses/spatial-masked_teacher/technical/best.pt` under the run root. Despite the
historical filename convention, these are the fixed final epoch-12 exports, not
validation-selected checkpoints. Restore `Geometry(checkpoint['config'])` from
`src.research.structural_response.model` and load `checkpoint['geometry']`.
The matching completion record binds the checkpoint SHA256. Future motion fits
must record that hash. The older local-response geometry class has a different
normalization and pooling contract and cannot restore these exports correctly.

cuEquivariance compilation is limited to four parallel compiler threads through
`CUEQUIVARIANCE_OPS_PARALLEL_THREADS`; this changes startup resource use, not the
objective or tensor precision. Motion jobs request 48 GB host memory because
higher-order kernel generation has a larger startup footprint than spatial fits.

Current successor reservations: spatial jobs **1025623** and **1025628**, 24 hours
each, after allocation 1023686 ends. Motion workers **1025624–1025627** and
**1025629–1025632** depend on their respective spatial job. Earlier unstarted
reservations were replaced during numerical preparation; their launch receipts
and frozen source are preserved in `technical/*preflight-v1*` and `*preflight-v2*`.
Structural and motion numerical gates are separate. A queued worker completes
its required gate if the preceding allocation ended before it finished.

Numerical preparation successor **1025645** runs on a large GPU for up to two
hours if the current allocation ends before the motion check finishes. It exits
without recomputing a completed check. All eight motion jobs additionally depend
on this gate; the two spatial fits can proceed independently once their own
checks have passed.
