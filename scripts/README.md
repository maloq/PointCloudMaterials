# Maintained commands

`python -m src.research.ni_latent_flow.queue prepare|launch|train|evaluate|kinetics|diagnostics|compare --config configs/ni_latent_sim/validation_20261018.json`
runs the expanded Ni validation follow-up; [workflow](../docs/ni_latent_validation.md).

`python -m src.research.ni_latent_flow.queue prepare|launch|train|evaluate|births|compare --config configs/ni_latent_sim/flow_20261018.json`
runs the fixed native Ni trajectory-method comparison; [workflow](../docs/ni_latent_flow.md).
`launch` detaches both allocated GPUs plus independent descriptor controls. Inspect
the existing launch receipt before resubmitting.

Use `python -m src.postprocessing --help` for the unified descriptor, search,
dataset, history and visualization commands; [module guide](../docs/postprocessing.md).

`python -m src.postprocessing al-gpu-search prepare|submit --config configs/analysis/al_nuclei_gpu_additions_20261008.json`
queues verified completed Al inputs through GPU extraction, unchanged CPU lineage
and dependent catalogue publication. The same interface owns `extract`, `track`,
`collect` and `publish` worker stages. Consult its recorded launch before any
resubmission; completed CPU searches are incorporated without recomputation.

`python -m src.postprocessing ni-gpu-search prepare|submit --config configs/analysis/ni_lowT_gpu_search_20261008.json`
switches the receipt-owned pending Ni CPU search to a GPU `resident`/`voronoi`
extraction job followed by unchanged CPU lineage tracking. It pins the original
campaign and reviewed GPU implementation, uses separate cache/result paths and
preserves all source roles and native frames. `extract` and `track` are worker
stages. [Launch and checks](../docs/simulations/ni_births_lowT_010_20261008.md#gpu-nucleus-search-october-8).

The same command with `configs/analysis/ni_lowT_gpu_search_400ps_20261008.json`
replaces the receipt-owned pending GPU search with the user's 400-ps hold revision.
It preserves original raw histories, caps every source view at 420 ps including
cooling and exports explicit zero-birth outcomes. Original plans remain frozen.
That analysis cap was subsequently withdrawn: use
`configs/analysis/ni_lowT_gpu_search_full_20261008.json` for all saved native
frames, including verified longer histories and retained post-stop chunks.
The simulation hold limit remains 400 ps; analysis has no time cap.
`configs/analysis/ni_lowT_gpu_search_handoff_20261008.json` retains that same
scientific analysis and prepared plans after moving the last Ni work off the
session's RTX6000 allocation. Its pinned handoff receipt replaces only producer
dependencies; the original allocation remains available to the user.

`src.simulation.campaigns.ni_births.handoff resume|fresh|gate --config CONFIG`
is the worker interface for the native-checkpoint handoff. Its receipt-bound
entry runs against the original frozen Ni producer, preserves the verified
prefix and matching NPT restart state, and enforces the 400-ps hold directly.
The current handoff is already submitted; see the campaign's launch receipts.

`python -m src.simulation.campaigns.ni_births.hard_stop prepare|submit --config configs/simulation/ni_births_lowT_hard_stop_400ps_20261008.json`
launches the receipt-owned live cap controller (`run` is its worker stage).
It verifies and publishes exact saved prefixes plus native cap checkpoints before
ending matching source children. Queue supervisors continue; actual native
overshoot and nonzero worker exits remain in the stopping audit. It has already
been submitted for this campaign; do not duplicate it.

`python -m src.postprocessing cache repack --plan EXTRACTION_PLAN --output NEW_PREPARATION --cache NEW_CACHE`
copies completed structural labels into selective detector arrays, with checksum
and plan checks and resumable chunks. It leaves original caches intact and does
not rerun PTM; use an external cache directory. See the module guide for wiring
the copied preparation into a new search.

`python -m src.postprocessing ni-search analyse --config FROZEN_CONFIG --plan-sha256 SHA256`
resumes an explicitly pinned Ni search with an audited equivalent implementation,
reusing completed chunks and recording actual execution hashes. See the module
guide for the frozen-plan and population checks.

Ni analysis uses bounded chunk scheduling; set `--workers` to the allocated
one-thread CPU workers. For a new campaign still producing trajectories, add
`--wait-for-jobs JOB1 JOB2` to admit completed, verified sources immediately
and wait for the remaining producers. Al history workers now checkpoint event
frames and share source-frame precursor preparation in the external cache.

For an existing birth-tail plan, `python -m
src.postprocessing histories assist --config <config>`
cooperatively processes unfinished, unlocked events. It preserves completed
packets and records the executing implementation. This does not submit or stop
other jobs. See [nuclei search performance](../docs/nuclei_search_performance.md).
New full-cell extraction retains atom-level PTM for reuse by tails. Search
configurations may set `descriptor_threads` (default 1); extraction launchers
allocate that many cores per frame worker, without increasing tracking cores.

`python -m src.simulation.campaigns.dense_al_birth_queue prepare|submit|repair|worker|collect`
executes the reserved V100 handoff for continuous Al birth-focused collection;
[protocol and receipts](../docs/simulations/al_dense_shooting_v100_20261007.md#birth-focused-handoff-october-7-evening).
`repair --launch LAUNCH` recovers the recorded missing-module startup failure in
a separate launch revision, after all original jobs have ended and before any
transferred trajectory has started. It retains source assignments and absolute
deadlines, checks frozen imports before GPU submission, and preserves the failed
launch and its code. Do not resubmit an already repaired queue.
`python scripts/convert_trajectory.py dense-al-birth DIRECTORY --delete-source`
verifies its variable-duration Q/V against the continuous timeline and stopping certificate.

`python -m src.simulation.campaigns.al_million_replay prepare|submit|worker`
recovers million-atom Al production Q/V at 0.10 ps through native checkpoint
replay with historical agreement gates;
[workflow](../docs/simulations/al_million_velocity_replay_20261007.md).

`python -m src.simulation.campaigns.ni_births.workflow prepare|submit-audit|submit|status --config configs/simulation/ni_births_010_20261007.json`
collects independent Ni NPT melts/cooling/holds with exact0.10ps Q/V and a dependent
full-cell nucleus-birth search; [workflow](../docs/simulations/ni_births_010_20261007.md).
The `ni_births_010_20261007_extension01.json` recipe uses the same commands for
three disjoint GPU lanes and adds an automatic `combine` stage that publishes
both verified birth catalogues without rerunning their analyses.
The `ni_births_lowT_010_20261008.json` recipe uses the same commands for a separate
36-source temperature screen at 950–1150 K on four GPU workers;
[protocol and launch](../docs/simulations/ni_births_lowT_010_20261008.md).
Passed numerical audits are reused through
`configs/simulation/ni_births_validation.json`: `submit-audit` returns the
existing receipt without submitting a job when compatible, and `submit` records
that reference. New seeds or campaign names do not require another audit.

`python -m src.simulation.campaigns.ni_births.allocation attach --config CONFIG --lane LANE --allocation JOB`
uses an explicitly authorized, idle GPU in the current allocation for queued Ni
sources. For a running lane, explicit `--worker-pid`, `--source-pid` and
`--source-index` identify its owned supervisor and current MD child: only the
supervisor pauses, the current MD finishes uninterrupted, and future sources
move to the attached GPU. The detached controller reuses its GPU type's reviewed
numerical validation and preserves
analysis dependencies on both executions. See the campaign's allocation receipt.

`python -m src.simulation.campaigns.ni_births.scale submit --config CONFIG --workers N --output NEW_LAUNCH_REVISION`
redistributes unstarted sources from receipt-owned running Ni lanes into new GPU
reservations. Existing native MD children continue while only their queue
supervisors pause; a detached controller verifies completion before releasing
those old allocations. The original frozen producer/configuration is reused,
source assignments remain disjoint, and the original analysis dependency is
extended to include all new workers and the drain controller. New reservations
use the recipe's combined compatible partitions and respect the user's GPU quota.

`python -m src.postprocessing ni-preview --config configs/simulation/ni_births_010_20261007.json --output OUTPUT`
exports a versioned progress plot and a descriptive, retrospective local-order
comparison; [definitions](../docs/metrics/ni_births_preview.md).

`python scripts/convert_trajectory.py native-qv INPUT.npz OUTPUT --delete-source`
verifies native float64 Q/V chunks, writes standard float16 observations with exact
identities/timelines and quantization receipts, and retains separate native restarts.

`python -m src.simulation.eam_gpu.benchmark --config configs/simulation/ni_eam_gpu_20261007.json` validates the pure-Ni CUDA EAM forces, analytic response derivatives and MD throughput against LAMMPS; [workflow](../docs/simulations/ni_eam_gpu_20261007.md).

`python -m src.simulation.campaigns.dense_al_shooting prepare|submit`
prepares additional dense-Al velocity reshoots and submits a gated V100 queue;
[recipe and commands](../docs/simulations/al_dense_shooting_v100_20261007.md).

`python -m src.postprocessing dataset refresh|status|verify` maintains the cumulative
Al nuclei-birth dataset with exact 0.10 ps views and native-only 0.30 ps data,
stable IDs, audited raw-cadence selection and incremental
updates from completed searches; [dataset guide](../docs/datasets/nuclei_birth.md).

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.continuous_motion.small_gpu launch --config configs/continuous_motion/small_gpu_20261006.json`
requests A40, A100 and L40S separately for the same1600-update fits, with128x2
accumulation and full256-sample noncollapse statistics; [workflow](../docs/continuous_motion.md#separate-smaller-gpu-requests).

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.continuous_motion.workflow prepare|launch|status --config configs/continuous_motion/all010_1600_20261006.json`
runs the1600-update fixed-step/continuous/derivative comparison with real Q,V,
long saved futures and short full-cell MEAM teachers; [workflow](../docs/continuous_motion.md).

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.response_dense_birth.pooled launch --config configs/analysis/response_all010_pooled_20261006.json` pools validation/calibration/test by user request, refits frozen readouts with training-only NLL selection, and audits neighborhood rejections. [Workflow](../docs/response_dense_birth.md#pooled-test-and-neighborhood-audit).

`python -m src.research.response_dense_birth.workflow launch --config configs/analysis/response_all010_birth_20261006.json` evaluates the twelve new response encoders on all available exact 0.10 ps v3 birth views with saved velocities, including partial archives and finer histories. It waits for newest labels without changing their jobs. The earlier `response_dense_birth_20261006.json` launch is preserved as superseded. [Workflow](../docs/response_dense_birth.md).

`python -m src.research.response_velocity.queue prepare|prepare-clustering|launch --config configs/simulation/response_native500_20261006.json` runs the native-velocity, source-temperature 0.5 ps response comparison; [workflow](../docs/simulations/response_native500_20261006.md).

`python -m src.postprocessing inventory --config configs/analysis/al_nuclei_dense_additions_extract_20261006.json`
verifies new production MD against executed MEAM commands, potential hashes and
prepared-liquid provenance, then freezes a fresh inventory and extraction plan.
Use `lineage_search prepare|submit --config configs/analysis/al_nuclei_dense_additions_20261006.json`
to run the unchanged birth protocol; [workflow](../docs/al_nuclei_lineage.md#october-6-verified-meam-production-addition).

`python -m src.research.nucleus_controls.ablation prepare|launch|summarize --config configs/simulation/nucleus_dynamics_ablation_20261006.json` runs the bounded box/thermostat comparison; [workflow](../docs/simulations/nucleus_dynamics_ablation_20261006.md).

`python -m src.postprocessing descriptors prepare --config configs/analysis/al_nuclei_dense_additions_extract_20261005.json`
freezes newly available native 0.10 ps observations, excluding the earlier population.
Then use `lineage_search prepare|submit --config configs/analysis/al_nuclei_dense_additions_20261005.json`
for detached extraction and revised 0.10/0.30 ps lineage searches;
[workflow](../docs/al_nuclei_lineage.md#new-dense-observations).

`python -m src.postprocessing search prepare|submit --config configs/analysis/al_nuclei_lineage_20261005.json`
restarts all frozen Al cadence views from existing structural classifications,
with separate birth and interaction fates; [workflow](../docs/al_nuclei_lineage.md).

`python -m src.research.response_birth launch --config configs/analysis/response_birth_20261004.json` runs frozen Q/QV response encoders on the original crystal-free nucleus-birth cohort with actual MD velocities and two GPU lanes; [workflow](../docs/response_birth.md).

`python -m src.research.response_future_similarity prepare|analyze|tracking --config configs/analysis/response_future_similarity_20261004.json` evaluates frozen response embeddings against independent future-shot banks. The `extract` stage imports the frozen campaign code; see [workflow](../docs/simulations/response_velocity_20261004.md#future-outcome-similarity).

`python -m src.simulation.campaigns.dense_al_gpu_transfer submit --config configs/simulation/al_dense_gpu_transfer_20261004.json` submits the captured unstarted-source hardware handoff; immutable preparation and previous launch receipts are required. [A100/L40S/A40 transfer](../docs/simulations/al_dense_gpu_20261004.md#october-4-hardware-transfer).

`python -m src.simulation.campaigns.dense_al_gpu prepare --config configs/simulation/al_dense_gpu_20261004.json` prepares a bounded dense-source transfer; `submit --launch LAUNCH` submits it as documented in the [GPU dense workflow](../docs/simulations/al_dense_gpu_20261004.md). Two GPU lanes retain the native NPT integrator, dense converter and stopping monitor.

`python -m src.research.meam_gpu.workflow launch --config configs/simulation/meam_gpu_blackwell_20261004.json --allocation ALLOCATION` freezes and detaches the CUDA force/response/MD campaign; `python -m src.research.meam_gpu.report --config configs/simulation/meam_gpu_blackwell_20261004.json` publishes current progress. [Workflow](../docs/simulations/meam_gpu_blackwell_20261004.md).

`python -m src.research.meam_md.prepare --config configs/simulation/meam_md_16k_20261004.json` prepares the shared periodic liquid; `python -m src.research.meam_md.workflow launch --config configs/simulation/meam_md_16k_20261004.json` runs gated CPU/GPU200ps trajectories; [workflow](../docs/simulations/meam_md_16k_20261004.md).

`python -m src.research.response_velocity.queue launch --config configs/simulation/response_velocity_20261004.json`
launches the fixed-initial-velocity MEAM comparison at0.1/0.5/1ps; [workflow](../docs/simulations/response_velocity_20261004.md).
The original large campaign is stopped. The new `configs/simulation/response_velocity_cuda500_float32_20261004.json`
recipe uses compiled CUDA MEAM,100/500fs and the full cohort, with longer runtime
reauthorized. `configs/simulation/response_velocity_three_hour_20261004.json`
retains the separate small pilot and its shared three-hour deadline.

`python -m src.research.response_meam.benchmark all --config configs/simulation/response_meam_torch_20261004.json` validates the differentiable Lee2003 Al PyTorch potential and compares complete local responses against MACE on one GPU; [workflow](../docs/simulations/response_meam_torch_20261004.md). `python -m src.research.response_meam.report --config configs/simulation/response_meam_torch_20261004.json` publishes saved timings.


`python -m src.postprocessing descriptors prepare|submit|progress --config configs/analysis/al_nuclei_search_20261003.json`
searches all compatible Al trajectories on separate exact0.75/0.1/0.3ps grids,
including paired0.1-to0.3 decimation; [protocol and execution](../docs/al_nuclei_search.md).

`python scripts/project.py relaxations scan|status|plan`
indexes existing relaxed cells and precise neighborhoods; `plan --requests REQUESTS.json --output MISSING.json`
separates reuse, native-checkpoint extraction and missing minimizations.
[Shared API and precision contract](../docs/data_storage.md#unified-relaxation-index).

`python -m src.research.response_geometry.clusters launch --config configs/analysis/response_geometry_clusters_20261003.json` and
`python -m src.research.response_geometry.queue prepare|submit --config configs/simulation/response_geometry_20261003.json`
compare unrelaxed response clustering spaces and queue expanded future-response training; [workflow](../docs/simulations/response_geometry_20261003.md).


`python -m src.data.al_pool refresh|status|select --config configs/al_research_pool.json`
maintains the shared Al inventory and joined appearance/fate observations,
with explicit cohort, input-domain and physical-cadence selection.
[Workflow](../docs/al_research_pool.md).

`python -m src.research.spatial_vicreg_bias.response_comparison restore|publish --config configs/analysis/response_cluster_comparison_20261003.json`
restores archived paired PaCMAP/MD viewers or publishes the selected local80
response encoder beside the retained descriptor and RH2 results. Frozen GPU
inference uses `python src/research/spatial_vicreg_bias/response_infer.py RESOLVED_CONFIG`.
[Workflow](../docs/spatial_vicreg_bias.md#october-3-response-encoder-addition).

`python -m src.research.birth_prediction.dense submit --config configs/birth_prediction/dense010_20261003.json`
searches completed native0.1ps Al descendants, then compares six-frame
preappearance descriptors against matched snapshot controls; [workflow](../docs/birth_dense010.md).

`python -m src.simulation.campaigns.dense_al_half_recovery prepare --config configs/simulation/al_main_half_recovery_20261003.json`
prepares native restart recovery of the stopped Al rerun queue with verified
binary observation prefixes; submit through the existing `dense_al submit`.
[Recovery contract](../docs/simulations/al_main_half_stop_20261001/README.md#october-3-quota-recovery).

`python -m src.postprocessing crop-windows --config configs/analysis/ta_appearance_crops_12ps_20261003.json`
saves separate±12ps Ta appearance-candidate windows from the verified20Å
movies; [workflow](../docs/ta_nucleus_crops.md).

`python -m src.postprocessing crops submit --config configs/analysis/ta_nucleus_crops_20261003.json`
exports separate native-cadence20Å Ta event movies with explicit window
truncation; [workflow](../docs/ta_nucleus_crops.md).

`python -m src.research.local_response_followup.queue prepare|launch --config configs/simulation/local_response_followup_20261003.json`
launches fifteen fixed-update response-weight fits and a separate GPU finite-displacement
forward campaign; [workflow](../docs/simulations/local_response_followup_20261003.md).

`python -m src.postprocessing.protocols.ta_content --config configs/analysis/ta_births_20261002.json`
summarizes saved Ta crystal fractions and conservative birth exclusions without
relabeling events; [definitions](../docs/metrics/ta_crystal_content.md).

`python -m src.research.birth_prediction.shooting_appearance submit --config configs/birth_prediction/shooting_appearance_20261003.json`
extends Al appearance prediction to shooting events and compares dissolution
with documented growth; [workflow](../docs/al_shooting_prediction.md).

`python -m src.research.local_response_audit --config configs/analysis/local_response_audit_20261003.json`
audits physical perturbation sizes, target/noise scales and competing gradients
at frozen local80 checkpoints; [definitions](../docs/metrics/local_response_audit.md).

`python -m src.research.birth_prediction.appearance_resume submit --config configs/birth_prediction/appearance_resume_20261003.json`
resumes unfinished full-cell quenches with an explicit execution-time extension;
[recovery and dependent CV](../docs/birth_appearance.md#nested-cross-validation).

`python -m src.research.response_force_engines.benchmark prepare|launch|summarize --config configs/simulation/response_force_engines_20261002.json`
compares persistent UNEP/MEAM force-only local responses with frozen MACE references;
[workflow](../docs/simulations/response_force_engines_20261002.md).
Use `python -m src.research.response_force_engines.parallel --config CONFIG --engine meam|unep --workers 3`
for persistent parallel replicas and `python -m src.research.response_force_engines.publish --config CONFIG`
for the saved-result figure and numerical-gate review (`--watch` waits for both workers).

`python -m src.research.birth_prediction.appearance_cv prepare|submit --config configs/birth_prediction/appearance_cv_20261002.json`
adds nested ancestry-grouped CV to the established/failed appearance comparison;
[workflow](../docs/birth_appearance.md#nested-cross-validation).

`python -m src.postprocessing.protocols.ta prepare|submit --config configs/analysis/ta_births_20261002.json`
searches full-cell Ta births and dissolved embryos; [workflow](../docs/ta_births.md).

`python scripts/convert_trajectory.py offline-dump SOURCE --delete-source`
archives an orthogonal position/velocity text dump with its actual timeline,
verified float16 observations by default, exact identities, float32 boxes and
quantization/checksum receipts. Use `--storage-dtype float32` for explicitly
requested float32 archives; existing float16 archives are not upcast.
Incomplete terminal-frame bytes are preserved separately. Native restarts and
scientific run states are unchanged. Execution inventory:
`docs/storage/al_float16_20261002/plan.json`.

`python -m src.postprocessing.protocols.shooting prepare|submit --config configs/analysis/shooting_births_20261002.json`
extends the full-cell birth/fate search to Al shooting futures; [workflow](../docs/shooting_births.md).

`python -m src.research.birth_prediction.appearance submit --config configs/birth_prediction/appearance_transfer_20261002.json`
compares original-birth and combined appearance predictors on failed embryos; [workflow](../docs/birth_appearance.md).

`python -m src.postprocessing.protocols.fates prepare|submit --config configs/analysis/nucleus_fates_20261002.json`
searches failed crystal embryos and appends fate labels to existing births; [workflow](../docs/nucleus_fates.md).

`python -m src.research.birth_prediction.merged prepare|submit --config configs/birth_prediction/merged_retest_20261002.json`
retests birth predictors with merged former validation/calibration/test and
training-only CV selection; [workflow](../docs/birth_prediction.md#merged-evaluation-retest).

`python -m src.research.local_response.queue prepare|submit --config configs/simulation/local_response_20261002.json`
runs gated local80 MLIP responses and twelve matched fits on two GPUs; [workflow](../docs/simulations/local_response_20261002.md).


`python -m src.research.birth_prediction.feature_families prepare|submit --config configs/birth_prediction/feature_families_20261002.json`
compares descriptor-family refits and matched permutations on original/relaxed
birth snapshots; [workflow](../docs/birth_prediction.md#descriptor-family-information).

`python -m src.research.response_performance.benchmark run --config configs/analysis/response_performance_20261002.json`
benchmarks numerical equivalence and full100fs oracle throughput; [workflow](../docs/response_atlas.md#october-2-oracle-acceleration-benchmark).
Use `src.research.response_performance.validate --config CONFIG` for isolated
repeats and acquisition-reuse verification, and
`src.research.response_performance.publish --root RUN_ROOT` for saved-result plots
(both invoked with `python -m`).

`python -m src.research.response_pacmap --config configs/analysis/response_pacmap_20261002.json`
plots matched held-out Al neighborhoods for response-trained MACE, MM-TDA and
GeoFormer; [transfer diagnostic](../docs/response_atlas.md#al-neighborhood-pacmap-transfer-comparison).

`python -m src.research.response_training.queue preflight|submit`
collects fixed-simulator responses and runs nine matched full-cell value/response
encoder fits with the float32/cuEquivariance/batch4 default;
[workflow](../docs/response_atlas.md#default-for-new-al256-response-runs).
An explicit `--config` selects a versioned recipe; preserve historical frozen runs.

`python -m src.research.predictive_followup.pacmap_views run --config configs/analysis/predictive_pacmap_20261001.json`
plots the saved shooting encoders with training-fitted PaCMAP, matched held-out
observations and a linked interactive comparison; [workflow](../docs/predictive_baseline.md#pacmap-views-of-the-completed-follow-up).

`python -m src.research.predictive_followup.queue prepare|encode|preflight|submit --config configs/predictive_baseline/followup_20261001.json`
compares matched frozen heads, Fourier supervision and nonnegative variance, including
MM-TDA-BLOCK-DIRECT-FULL; [workflow](../docs/predictive_baseline.md#matched-head-target-and-mm-tda-follow-up).
Use `python -m src.research.predictive_followup.lanes submit --config configs/predictive_baseline/followup_20261001.json`
for submission-count-limited clusters; five workers run the same48 fits sequentially.

`python -m src.research.birth_prediction.relaxed freeze|submit --config configs/birth_prediction/relaxed_temporal_20261001.json`
quenches all existing birth inputs and repeats paired frozen readouts, with train/test errors;
[workflow](../docs/birth_prediction.md#full-cell-relaxed-input-comparison).

`python -m src.research.predictive_baseline.queue prepare|preflight|submit --config configs/predictive_baseline/al480_20261001.json`
trains geometry-only MACE128 on fixed future-feature means from repeated Al shooting;
[workflow](../docs/predictive_baseline.md).

`python -m src.research.response_atlas.queue submit --config configs/response_atlas/feasibility_20261001.json` queues the [response-atlas feasibility study](../docs/response_atlas.md).


`python -m src.research.birth_prediction.temporal prepare|submit --config configs/birth_prediction/temporal_site_20260930.json`
compares fixed-endpoint history controls and persistent-site versus evolving
structural information; [workflow](../docs/birth_prediction.md#fixed-endpoint-and-site-persistence-comparison).

`python -m src.research.shooting_laws.queue bind|preflight|submit --config configs/shooting_laws/al480_20260930.json`
derives and compares physical future distributions from the 480-shot Al release.
`python -m src.research.shooting_laws.diagnostics bind|submit --config configs/shooting_laws/diagnostics_20260930.json`
scores the separate thermostat, nested-Al and Ta ensembles; [workflow](../docs/shooting_laws.md).

`python -m src.research.liquid_predictability.rich_objective_study submit-run --config configs/liquid_predictability/rich_tda_block_direct_full_20260930.json`
submits the full-data block-scaled descriptor encoder with direct VCReg;
[comparison and workflow](../docs/rich_tda_objectives.md#full-data-direct-vcreg-combination).

`python -m src.research.birth_prediction.leakage audit|explain|supplement|plot --config configs/birth_prediction/leakage_features_20260930.json`
audits retained birth inputs and explains frozen rich-descriptor predictors with
matched-group permutations and source uncertainty; [workflow](../docs/birth_prediction.md#leakage-and-feature-audit).

`python -m src.research.birth_prediction.extension prepare|submit --config configs/birth_prediction/drop_to_one_cv_20260930.json`
extends the existing birth histories down to one frame, runs five-fold grouped
readout CV and plots proper scores, AP, calibration and source uncertainty;
[workflow](../docs/birth_prediction.md#truncation-to-one-frame-and-readout-cross-validation).

`python -m src.research.birth_prediction.queue bind|submit --config configs/birth_prediction/preappearance_20260930.json`
compares rich-descriptor boosting, frozen rich/VICReg MACE and linear controls
on crystal-free histories before isolated establishment, with four matched
history endpoints; [workflow](../docs/birth_prediction.md).

`python -m src.research.liquid_predictability.rich_objective_study prepare|submit --config configs/liquid_predictability/rich_tda_pilots_20260930.json`
packs the existing fitting rows, runs seven matched 10% VCReg/TDA pilots and
queues a validation-selected full-data fit; [workflow](../docs/rich_tda_objectives.md).

`python -m src.research.liquid_predictability.rich_multimaterial_evaluate --run RUN_ROOT --batch-size 1024 --patch-chunk 256`
evaluates a stopped rich-descriptor encoder with its frozen producer and selected
checkpoint, updating the existing training W&B summary without opening a run;
[workflow](../docs/rich_multimaterial_encoder.md#detached-evaluation-after-stopping).

`python -m src.research.spatial_vicreg_bias.mace_checkpoint infer|publish --config configs/analysis/mace_rich_current.json` evaluates the current selected rich-MACE checkpoint in both comparison viewers; [workflow](../docs/spatial_vicreg_bias.md#current-rich-mace-checkpoint-in-the-interface-viewer).

`python -m src.research.spatial_vicreg_bias.compact_view_assets --publication REPO_BUNDLE`
creates lossless compact display assets from existing sample/embedding exports on
Slurm CPU. Follow with `comparison_layout`; [viewer workflow](../docs/spatial_vicreg_bias.md#responsive-rendering).

`python -m src.research.spatial_vicreg_bias.sample_lattice --publication REPO_BUNDLE --workers 4`
exports PTM-oriented ideal-lattice overlays for saved samples on Slurm CPU.
`embedding_travel infer|publish` exports frozen vector probes and spatial/feature
neighbor graphs; run inference from the recorded frozen model source on Slurm GPU,
then refresh with `comparison_layout`. [Definitions](../docs/metrics/interactive_structure_paths.md).

`python -m src.research.spatial_vicreg_bias.sample_environments --source DENSE_BUNDLE --publication REPO_BUNDLE --dataset matched|static`
exports up to five real 80-atom neighborhoods per cluster from saved dense labels
on Slurm; follow with `comparison_layout` to show the rotatable examples. No fitting.

`python -m src.research.spatial_vicreg_bias.comparison_layout --publication REPO_BUNDLE --dataset matched|static`
refreshes the compact PaCMAP/MD layout and dataset selector using saved page
payloads, without recomputing projections, clusters or correspondence metrics.
It also publishes the complete viewer as `heldout-al.html` or `al-static.html`
directly in the experiment output folder; MACE and checkpoint-explorer publication
keep those entry pages current too.

`python -m src.research.spatial_vicreg_bias.checkpoint_explorer --dense-config RESOLVED_CHECKPOINT_DENSE_CONFIG`
publishes the matched GeoFormer / descriptor explorer with epoch selection,
advanced training variants, matched cluster colors and paired full-snapshot MD.
Run inside Slurm after frozen dense inference for its declared checkpoints.
[Workflow](../docs/spatial_vicreg_bias.md) · [definitions](../docs/metrics/checkpoint_cluster_matching.md).

`python -m src.research.spatial_vicreg_bias.paired_pacmap --source COMPLETED_STATIC_BUNDLE --publication REPO_STATIC_BUNDLE`
publishes saved neural/descriptor 3D PaCMAP with optimal matched colors,
membership matrices, overlap bars and dense MD. Assignments use saved dense
contingencies; no neural or cluster refitting. [Definitions](../docs/metrics/cluster_color_matching.md).

`python -m src.research.spatial_vicreg_bias.static_md submit --config configs/analysis/static_al_interface.json`
submits the six relaxed Al snapshots (166–240 ps) for frozen NN / rich-descriptor
cluster comparison, paired dense MD views and PaCMAP; [workflow](../docs/spatial_vicreg_bias.md).

`python -m src.analysis.al_replay --config configs/analysis/al_replay_completed20_20260929.json`
compares old/new Al trajectories from the same native preparations at exact shared
times; [paired protocol](../experiments/al_replay_20260929/README.md).

`python -m src.analysis.al_duration --config configs/analysis/al_duration_all150_20261001.json`
audits all 150 historical Al sources and the captured complete dense descendants
for late transformation, bulk plateaus and hypothetical early stopping;
[protocol and findings](../experiments/al_duration_20261001/README.md).

`python -m src.analysis.al_half_transform --config configs/analysis/al_half_transform_20261001.json`
evaluates the user-defined 50%-crystal stopping target from the frozen duration
observations, including confirmation and a 6 ps prediction tail; same research protocol.

`python -m src.simulation.campaigns.dense_al prepare-half-stop --config configs/simulation/al_main_half_stop_20261001.json`
versions only unstarted Al sources to halfway/peer stopping, with native-state
monitoring and verified variable-length exports; [workflow](../docs/simulations/al_main_half_stop_20261001/README.md).
`python scripts/convert_trajectory.py dense-al-half RUN_DIR --delete-source`
is its maintained converter for both 0.1 and 0.01 ps.

`python -m src.research.spatial_vicreg_bias.island_audit --config configs/analysis/interface_pacmap.json --output OUTPUT_DIRECTORY` runs the screenshot-specific descriptor-island audit inside a Slurm CPU allocation; [definitions](../docs/metrics/pacmap_islands.md) and [findings](../experiments/spatial_vicreg_bias_20260929/PACMAP_ISLANDS.md).

`python -m src.research.spatial_vicreg_bias.md_space --config RESOLVED_SUBMISSION_CONFIG --dense-config RESOLVED_DENSE_CONFIG` publishes two independent dense MD panels with cluster colors matching PaCMAP. `python -m src.research.spatial_vicreg_bias.dense_md prepare|infer --config RESOLVED_DENSE_CONFIG` computes their full-snapshot descriptors or frozen neural assignments on Slurm. [Workflow](../docs/spatial_vicreg_bias.md).

`python -m src.research.spatial_vicreg_bias.pacmap_views submit --config configs/analysis/interface_pacmap.json` submits 2D and interactive 3D PaCMAP of frozen encoder/projector and rich descriptors; no neural training. [Workflow](../docs/spatial_vicreg_bias.md).

`python -m src.research.spatial_vicreg_bias.correspondence submit --config configs/analysis/interface_cluster_correspondence.json` submits independent rich TDA/bond-order/CNA clustering and interface-local correspondence to frozen neural clusters; [protocol](../experiments/spatial_vicreg_bias_20260929/INTERFACE_CORRESPONDENCE.md).

`python -m src.research.spatial_vicreg_bias.queue submit --config configs/spatial_vicreg_bias/al64_20260929.json`
submits matched GeoFormer neighbor-alignment fits and physical cluster readouts;
[workflow and existing submission](../docs/spatial_vicreg_bias.md).

`python -m src.research.liquid_predictability.rich_multimaterial_queue launch --config configs/liquid_predictability/rich_multimaterial_h100_20260929.json --allocation ALLOCATION` starts a detached H100 fit from sealed descriptors and queues a checkpointed two-GPU continuation; [workflow](../docs/rich_multimaterial_encoder.md).


`python -m src.research.liquid_predictability.rich_encoder probe|submit|worker|train --config configs/liquid_predictability/rich_mace256_20260929.json`
runs 60 full passes of raw rich-descriptor learning with MACE256/L3/Z256;
[workflow](../docs/rich_descriptor_encoder.md).

`python -m src.research.liquid_predictability.control_relaxation submit --config configs/simulation/liquid_full_relaxation_20260928.json` submits full-cohort relaxation and its dependent study; [operations](../docs/simulations/liquid_full_relaxation_20260928.md).

`python -m src.research.liquid_predictability.control_queue submit --config configs/liquid_predictability/controls_relaxed_20260928.json`
queues known/null signal controls, rich-feature MACE training and paired relaxed-input/label comparisons; [workflow](../docs/liquid_controls.md).


`python -m src.research.liquid_predictability.descriptor_baselines submit --config configs/liquid_predictability/descriptor_baselines_20260928.json`
adds mean, affine ridge and linear-logit controls to the frozen GPU descriptor
comparison; [workflow](../docs/liquid_descriptors.md#mean-and-linear-controls).

`python -m src.research.liquid_predictability.descriptor_queue submit --config configs/liquid_predictability/descriptors_al64_20260928.json`
compares rich TDA, bond order, CNA and geometry descriptors with GPU boosted trees
and a small MLP on the fixed liquid-only cohort; [workflow](../docs/liquid_descriptors.md).


`python -m src.research.encoder_mechanisms.results --config CONFIG --output OUTPUT`
summarizes completed matched encoder trajectories and adaptation using saved
metrics and source-paired predictions, with frozen metric definitions and
scientific plots; [workflow](../docs/encoder_mechanisms.md).

`python -m src.research.liquid_predictability.queue launch --config configs/liquid_predictability/al64_20260928.json`
runs physical profiles, descriptor/frozen controls, joint distance-only VCReg MACE,
source learning curves and a visible-crystal control; [workflow](../docs/liquid_predictability.md).

`python -m src.research.crystal_vector.interface_queue launch --config configs/crystal_vector/liquid_distance_20260928.json --gpu 0`
reuses the expanded cohort for strict liquid-only external-crystal localization,
with two GPUs, batch 512 and VCReg; [workflow](../docs/crystal_liquid_distance.md).

`python -m src.research.crystal_vector.feature_audit probes|interventions --config configs/analysis/crystal_feature_dominance_20260928.json`
audits dominant features with training-only readouts and frozen-head interventions;
[local diagnostic workflow](../docs/crystal_interface_unseen.md#feature-dominance-audit).

`python -m src.research.crystal_vector.interface_queue launch --config configs/crystal_vector/interface_unseen_20260928.json --gpu 0`
prepares dense interface-invisible contexts on CPUs, then runs one detached
two-GPU VCReg fit with global batch 512; [workflow and saved-prediction audit](../docs/crystal_interface_unseen.md).

`python -m src.research.crystal_vector.interface_queue submit --config configs/crystal_vector/interface_al64_20260928.json`
prepares the separate interface-layer targets and queues three joint encoder/context
treatments, including crystal-interior evaluation; [workflow](../docs/crystal_interface.md).

`python -m src.research.crystal_vector.queue prepare|launch|worker|train|evaluate --config CONFIG`
runs the joint snapshot crystal-distance/vector experiment with random batches;
[workflow](../docs/crystal_vector.md).

`python -m src.simulation.campaigns.dense_al prepare|prepare-continuation|submit|worker|collect`
reruns the main Al preparations at exact 0.1 ps, preserving ancestor roles;
`convert_trajectory.py dense-al RUN_DIR` verifies their canonical exports.
[Campaign and execution](../docs/simulations/al_main_010ps_20260927/README.md).
`convert_trajectory.py dense-al-001ps RUN_DIR` verifies their float16 exports in
bounded chunks. [Protocol and execution](../docs/simulations/al_main_001ps_20261001/README.md).

`python -m src.research.distance_encoder.material_queue submit|worker --config CONFIG --allocation JOB --gpu INDEX`
fine-tunes CD-MACE128-D6 separately for Al and Ta, followed by paired parent
evaluations; [workflow](../docs/distance_encoder_material_finetune.md).

`python -m src.analysis.cluster_explorer --config configs/analysis/cd_mace128_explorer.json --stage compute|render`
builds the offline linked static explorer over published standard/order analyses;
[workflow](../docs/analysis_visualization.md#linked-structure-exploration).
The [CD-MACE128 static recipe](../docs/structural_static_analysis.md#cd-mace128-distance-supervised-encoder)
exports and verifies the completed distance-trained encoder before analysis.

`python scripts/project.py simulations --details --output DIRECTORY` exports
measured simulation holdings, per-trajectory timelines, sampled velocity fields,
formats and disk allocation. It reads producer records and array headers; it does
not delete, convert or rerun simulations. [Current inventory](../docs/simulations/inventory_20260926/README.md).

`python -m src.research.distance_encoder.history_queue submit|worker|prepare --config configs/distance_encoder/md_history6_20260926.json` runs the end-to-end MD-history distance comparison; [workflow](../docs/distance_encoder_history.md).

`python -m src.research.encoder_mechanisms.workflow submit|check|worker --config configs/encoder_mechanisms/alignment_readout_20260926.json`
runs the matched alignment/readout/adaptation queue;
[execution, resume and CPU birth audit](../docs/encoder_mechanisms.md).
`python -m src.research.encoder_mechanisms.recovery submit --config CONFIG --output OUTPUT`
resumes its saved encoder checkpoints with isolated evaluation processes and
the originally declared adaptation fits; completed analyses keep their provenance.

`python -m src.research.distance_encoder.queue submit|worker --config ...`
submits the CD-MACE128-VC comparison; `distance_encoder.local` evaluates a single
local embedding without context predictors. [Workflow](../docs/distance_encoder.md).

`python -m src.simulation.campaigns.position_shooting prepare|submit|worker|collect`
queues replicated full-cell Ta shots through the elemental producer, with verified
STORE publication and a dependent failure-preservation job. See the
[shooting record and CLI arguments](../docs/simulations/ta_shooting_20260926.md).

`python -m src.research.distance_encoder.prepare plan|lane|seal` attaches distance
labels to the dynamic multi-material release. `distance_encoder.train` runs
distributed end-to-end training; `distance_encoder.context` evaluates/refits the
selected context predictors. [Workflow](../docs/distance_encoder.md).

`python -m src.analysis.native_mace_adapter --config configs/analysis/native_mace_epi_latest_static.json --stage export|verify`
bridges native self-supervised MACE checkpoints into the full static gallery;
then run `python -m src.analysis.pipeline configs/analysis/static_native_mace_epi_latest_al.yaml`.
See [checkpoint fidelity and the six-snapshot protocol](../docs/structural_static_analysis.md).

`python -m src.analysis.cluster_order --config configs/analysis/mace_epi_epoch12_order.json --stage compute|render`
analyzes full-source PTM and a fixed stratified sample over saved clusters, then
renders representatives, detected order and the multi-sample summary. CPU only;
no encoder or clustering fit. [Workflow and figures](../docs/analysis_visualization.md#local-order-figures).

`python -m src.research.spatial_distance.queue prepare|worker --config configs/analysis/spatial_distance_20260926.json`
trains continuous crystal-distance readouts; `spatial_distance.confidence`
exports fixed probability-threshold diagnostics. [Workflow](../docs/spatial_distance.md).

`run_lammps_campaign.py birth-sources prepare|submit|worker` produces fresh Al
birth-screen sources on CPUs; `convert_trajectory.py birth-pair RUN_DIR` verifies
their paired precision exports. See the [uniform-temperature campaign](../docs/simulations/al_birth_uniform_20260926.md).

`python scripts/experiment_registry.py publish --plan configs/analysis/result_publication_20260926.json`
publishes grouped galleries over retained artifacts without fitting or metric
recomputation. `publish --record PATH/to/run.json` refreshes one existing gallery,
including its interactive HTML entry pages; `--include-paper-svg` exposes retained
cluster-proportion paper SVGs. `results` refreshes explicit receipts in the existing SQLite
catalogue; `verify-results --record PATH/to/run.json` checks evidence hashes and
links. `build --from-snapshot` repairs saved inventory grouping/classification
while retaining its observation date. See the
[result system](../docs/research_results_system.md).

`python -m src.research.encoder_quality.queue submit|worker|report --config configs/analysis/encoder_quality_latest_20260926.json`
evaluates eight frozen native MACE checkpoints; [checks, Slurm execution and results](../docs/encoder_quality.md).

`python -m src.postprocessing.protocols.harvest --config configs/analysis/nucleus_harvest_20260926.json`
audits training-source birth coverage, causal eligibility, competing contacts
and bond order; [CPU operations](../docs/nucleus_harvest.md).

`python -m src.research.spatial_approach.queue prepare|worker|collect --config configs/analysis/spatial_approach_20260926.json`
runs fixed-snapshot spatial warning-distance probes; [workflow](../docs/spatial_approach.md).

`python -m src.postprocessing.protocols.external_data|external_audit`
processes large Al/Mg/Ti/Ta histories in
parallel chunks; [CPU commands and deadlines](../docs/crystallization_origin_cpu.md).

`python -m src.postprocessing.protocols.extract|audit --config configs/analysis/crystallization_origin_20260925.json`
audits full-cell crystal ancestry and label availability without fitting;
[commands and definitions](../docs/encoder_research/crystallization_origins.md).

`python -m src.data.fixed_cohort.prepare --config configs/fixed_cohort/al64_v1.json --workers 4`
builds the immutable 64-center Al benchmark and source-disjoint structural
pretraining corpus; [data contract and loaders](../docs/datasets/fixed_al64.md).
CPU preparation only; no training, new simulations or online tracking.

`python -m src.research.equivariant_context.queue prepare|check|submit|worker|collect --config configs/equivariant_context/comparison_20260925.json` prepares the shared-encoder invariant/vector/tensor/hierarchical predictor comparison; [operations](../docs/equivariant_context.md). Preparation and checks never submit or train.

`python -m src.research.supervised_onset.campaign check|submit|collect --config
configs/supervised_onset/information_20260925/campaign.json` runs the small/500k/1M/2M
comparison through the existing supervised trainer; [operations](../docs/supervised_capacity.md).

`python -m src.research.supervised_onset.queue prepare|preflight|launch|worker|collect`
runs the distinct crystallization-supervised likelihood protocol (AP reporting only) on the existing
larger paired cohort; [operations](../docs/supervised_onset.md).

`python -m src.research.prediction_context_audit --config configs/analysis/prediction_context_20260925.json`
logs the actual prediction inputs of275 saved AP comparisons, checking probe
widths and retaining evidence hashes; [policy and ledger](../docs/encoder_research/prediction_context.md).

`python -m src.research.geoframe_evolution.train|reference|queue|precursors`
reproduces the epoch-34 GeoFrame recipe and measures interfaces, defects,
liquid order and prediction along training; [commands](../docs/geoframe_evolution.md).

Run from the repository root with `conda run -n pointnet-torch214 python …` for new
GPU work. Exact resumes of existing jobs retain their recorded `pointnet`
environment; see [the PyTorch upgrade](../docs/pytorch214_upgrade.md).
Commands contain argument parsing and forwarding only; scientific implementation
belongs in `src/`. Run-specific settings belong in configuration, not copied runners.
See [workflow details](../docs/workflows.md), [experiment records](../experiments/README.md)
and [the output layout](../docs/research_layout.md).

`python scripts/experiment_registry.py encoders` refreshes the offline encoder
research catalogue and SQLite evidence database from registered local/archived
reports and tables. It performs no training or inference. See the
[encoder research handbook](../docs/encoder_research/README.md).

`python -m src.research.local_predictability.report --config
configs/analysis/local_predictability_report.json --output output/local_predictability/NEW-SUMMARY`
collects completed H100/RTX results and scores deferred observability/readout
predictions on CPU. It verifies paired rows, exports source intervals and figures,
and refuses to overwrite an existing report. It does not fit models. See the
[summary metric definitions](../docs/metrics/local_predictability_summary.md).

| Command | Workflows / implementation |
| --- | --- |
| `project.py datasets --refresh [--output docs/datasets]` | Searchable dataset registry, metadata cards, potentials, current schemas and discovery audit; [start here](../DATASETS.md), `src/project_runtime/dataset_registry.py`. |
| `benchmark_hardware.py [storage,cpu,gpu,all]` | Defaults to all; prints/saves a standard results table. Optional `--cpu-ranks 1 8 24` runs a CPU sweep. Synthetic inputs; `src/hardware_benchmark/`; [usage](../docs/hardware_benchmark.md). |
| `experiment_registry.py build` | Refresh the searchable experiment, simulation and ideas dashboard; `src/experiment_runner/registry.py` |
| `experiment_registry.py storage` | Human-readable size report and large-file CSV; `src/experiment_runner/storage.py` |
| `experiment_registry.py clean [--root output/RUN]` | Preview removable inference caches; add `--apply --inactive` only for inactive runs. Metadata and reconstruction evidence are preserved. |
| `experiment_registry.py metrics-docs` | Check that metric descriptions match their implementing source files. |
| `experiment_registry.py run --spec SPEC` | Track an explicit command, config/source snapshots and outcome; `tracking.py` |
| `experiment_registry.py status --record RECORD` | Inspect tracked execution state. |
| `experiment_registry.py prune --plan PLAN` | Preview a hash-verified explicit deletion plan; `--apply` executes it. |
| `experiment_registry.py pack-logs --before ISO_TIME` | Preview lossless log archiving; `--apply` verifies archives before deleting originals. |
| `experiment_registry.py relocate-caches --plan PLAN` | Preview verified shared-cache relocation; details in [registry guide](../docs/output_registry.md). |
| `run_experiments.py --plan PLAN` | Local/Slurm training plans, explicit resume, collection; `src/experiment_runner/cli.py` |
| `run_lammps_campaign.py WORKFLOW` | Simulation families; `src/simulation/campaigns/` |
| `convert_trajectory.py FORMAT_OR_AUDIT` | Verified producer-specific conversions; [formats and provenance](../docs/trajectory_conversion.md) |
| `inspect_temporal_lammps_dataset.py` | Dataset inspection/cache preparation; `src/data/inspect_temporal.py` |
| `run_shooting_ablation.py METHOD --config CONFIG` | Distributional, spatial, geometry, multiscale, dynamical, short-horizon, encoder-finetune, temporal-pretraining; `src/temporal_vamp/commands/` |
| `run_predictive_atlas.py METHOD --config CONFIG` | Frozen, history, finetune, temporal-encoder protocols; `src/temporal_vamp/commands/` |
| `analyze_geoframe.py WORKFLOW` | Stability, variability and representation comparisons. |
| `train_geoframe_spatiotemporal.py` | Config-selected training; `src/training_methods/spatiotemporal.py` |
| `run_temporal_vamp.py`, `evaluate_temporal_vamp.py` | VAMP training/evaluation; `src/temporal_vamp/` |
| `run_shooting_predictor.py`, `evaluate_shooting_ablation0.py` | Baseline shooting prediction/evaluation. |
| `run_nested_committor.py`, `run_predictability_map.py` | Configured committor/predictability analysis. |
| `run_temporal_encoder_pretraining.py`, `run_ordinary_temporal_embedding_cache.py` | Temporal pretraining/embedding production. |
| `analyze_shooting_branch_outcomes.py` | Branch statistics. |
| `plot_experiment_summary.py`, `plot_grouped_metric_csv.py` | Summary/CSV plotting; `src/analysis/plots/` |
| `plot_homogeneous_checkpoint.py`, `render_shooting_dynamics_gifs.py` | Simulation visualization. |

Current training and analysis use existing module entry points:

`python -m src.research.encoder_context.queue prepare|check|launch|report --config
configs/encoder_context/al64_20260925/campaign.json` runs the fixed-Al64 encoder
initialization comparison with vector/harmonic predictors and complete evaluation;
[operations](../docs/encoder_context.md). Tests/preflight remain local-only.

`python -m src.research.onset_horizons --config configs/analysis/onset_horizons_20260924.json`
replays saved native and joint-head onset probabilities at3/6/12 ps, with3 ps
primary and explicit event counts. It preserves all historical model selectors
and performs no training; [definitions](../docs/metrics/onset_horizons.md).

The structural-reconstruction, AP-tuned robust-onset/spatial-hierarchy, and
conditioned neighborhood-JEPA trainers were retired on 2026-09-30. Their numerical
reproduction uses recorded frozen producers; [retirement and retained tools](../docs/architecture_retirement.md).

`python -m src.research.structural_state_onset_review --run RUN --output NEW-OUTPUT`
audits completed repaired-encoder onset predictions and exports paired whole-source
uncertainty for12ps AP, Brier, binary log loss and event-time NLL. It never refits
models; see [metric definitions](../docs/metrics/structural_state_onset_review.md).

`python -m src.research.relaxed_encoder.queue evaluate --config
configs/analysis/relaxed_encoder_expanded.json` runs extraction and frozen readouts
without encoder training. Explicit exclusions are bound to the frozen plan;
see [stopping fits and evaluating](../docs/relaxed_encoder.md).

`python -m src.research.relaxed_encoder.evaluation freeze|reuse|build|worker|report
--config configs/analysis/relaxed_encoder_large_test.json` runs the larger fixed-grid
crystallization assay with whole-source uncertainty and explicit event coverage;
see [operations](../docs/relaxed_encoder.md) and the
[scientific protocol](../experiments/relaxed_encoder_large_test_20260921/README.md).

`python -m src.research.relaxed_encoder.interim prepare|extract|probe|report --config
configs/analysis/relaxed_encoder_interim.json` evaluates finished encoders on an
existing fixed assay cohort while the larger evaluation is preparing. It retains
source splits and matched linear/MLP probes; see [interim readouts](../docs/relaxed_encoder.md).

`python -m src.research.relaxed_encoder.recovery cpu|cuda|build --config CONFIG --name RETRY_NAME`
repairs cells from an existing archived restart plan and rebuilds data. It does
not submit encoder training; [retained operations](../docs/architecture_retirement.md).

`python -m src.research.gatr_conditional_information.comparison --config configs/analysis/conditional_information_local_last.json`
repeats the assay for a pinned pair of final native MACE/GATr checkpoints, using
each model's frozen training implementation and common radial controls. Stages
`prepare`, `extract`, `probe`, `report` preserve separate extraction and scoring;
the same run also repeats trajectory stability. See the
[latest-checkpoint protocol](../experiments/gatr_conditional_information_20260918/LOCAL_LAST622.md).

`python -m src.research.gatr_conditional_information --config configs/analysis/gatr_conditional_information.json`
tests the frozen exported GATr state beyond radial structure and density on
node07/A100. Stages `prepare`, `probe`, `report` resume verified extraction,
nested source-held-out readouts and figure/table export. The separate command
`python -m src.research.gatr_conditional_information.spatial --config configs/analysis/gatr_conditional_spatial.json`
extends same-frame radial matching to the existing dense snapshots with frozen
probe settings; see the [scientific protocol](../experiments/gatr_conditional_information_20260918/README.md).

`python -m src.research.gatr_equivariant --config configs/analysis/gatr_equivariant.json`
audits frozen GATr multivector directions on the requested node07/A100. Stages
`temporal`, `spatial`, `report` support resuming extraction; default `all` also
runs readout interventions. It exports source-level angular and spatial-order
metrics and an offline 3D viewer; see the
[scientific protocol](../experiments/gatr_equivariant_20260918/README.md).

`python -m src.research.trajectory_stability --config configs/analysis/trajectory_stability.json`
compares the latest selected MACE/GATr states with TDA, SOAP, bond-order, radial
and angular descriptors on matched, identity-preserving Al trajectories. Stages
`prepare`, `encode` and `report` support separate CPU/GPU execution. It exports
source intervals, full trajectories and exact metric definitions; see the
[scientific protocol](../experiments/trajectory_stability_20260918/README.md).

New geometry-protected 2× snapshot encoders use the prepared recipes in
`configs/shared_pretraining/geometry_fp32_2x/`; see
[architecture, precision and validation](../docs/shared_pretraining_geometry_fp32_2x.md).
These recipes have not been submitted. Reuse the existing `shared_pretraining`
profiler and runtime; no additional training entry point is needed.

The existing v3 VICReg restart uses its frozen code and the command `python -m src.training_methods.shared_pretraining.queue
submit --plan configs/shared_pretraining/restart_b1024_lr002_bf16/campaign.json`.
It starts only the two fresh structural fits in the recorded existing allocations;
see [the restart workflow](../docs/shared_pretraining_restart_20260918.md).
The separate `shared_pretraining.profile` command requires `--precision float32|bf16`
and measures actual batch updates after warmup. It is never called by training.
The original shared-campaign/H200 recipes below are paused historical recipes;
reproduce them with their frozen code, not the current normalized architecture.

`python -m src.training_methods.shared_pretraining.queue submit --plan
configs/shared_pretraining/campaign.json` starts the approved 12-epoch structural
fits, initialized causal continuations and frozen analyses, with detached local
workers and dependent Slurm jobs. It freezes executable code, uses online W&B,
cosine warmup and allocation-aware exact resume. Its data command is
`python -m src.training_methods.shared_pretraining.data --config
configs/shared_pretraining/data.json --workers 6`; see the
[shared campaign workflow](../docs/shared_pretraining_20260918.md).

`python -m src.training_methods.shared_pretraining.queue serial --plan
configs/shared_pretraining/h200_batch1024/campaign.json --deadline-utc ISO_TIME`
runs the batch-1,024 H200 pipelines sequentially without Slurm, checkpoints at
the deadline and resumes completed/partial stages. See the
[H200 handoff](../docs/h200_shared_pretraining_task_20260918.md). Its separate
`shared_pretraining.profile` preflight accepts `--memory-limit-gib 80`; scientific
training never runs hardware benchmarks.

`python -m src.data.structural_pretraining.prepare --config
configs/structural_pretraining/data.json --workers 6` freezes a training-only
multi-material radius calibration and prepares raw snapshot/three-frame inputs,
spatial/temporal neighbor pairs and instantaneous physical/TDA anchors. See the
[structural pretraining workflow](../docs/structural_pretraining_20260917.md).

New structural fits use `src.training_methods.shared_pretraining.queue` and
`configs/shared_pretraining/al_stable/`, with physical/instantaneous-TDA anchors,
full-batch representation statistics and training-only head calibration.
The earlier `configs/structural_pretraining/{mace_vicreg,gatr_vicreg,gatr_lejepa}.json`
are immutable historical run recipes; exact resumes require their frozen source.
The standalone trainer now also requires explicit `materials` and
`head_calibration_rows` for newly created recipes.

`python -m src.data.relaxed_targets prepare|run|status --config
configs/simulation/relaxed_tda_al.json` produces resumable full-cell relaxed-TDA
labels on denser native training windows and completed Al shooting trajectories.
CPU workers share an immutable, ancestry-aware queue; see
[target generation and resume](../docs/relaxed_tda_targets.md).

`python -m src.research.backbone_tda --config
configs/local_predictability/backbone_v2/tda_snapshot.json` compares frozen physical
MACE/GATr snapshot states with matched source-held-out instantaneous-TDA readouts.
See [execution/resume](../docs/backbone_tda.md) and the
[scientific protocol](../experiments/local_predictability_20260917/TDA.md).

`python -m src.research.local_predictability.backbone_v2 --config
configs/local_predictability/backbone_v2/rtx6000_screen.json --stage screen`
runs fresh MACE/cuEquivariance and axial GATr gates, explicit workload profiles,
and a matched one-seed physical snapshot screen. The `fit` stage uses the common
physical/onset trainer without profiling. See [execution/resume](../docs/backbone_v2.md)
and the [scientific protocol](../experiments/local_predictability_20260917/BACKBONE_V2.md).

`python -m src.research.backbone_repeats --config
configs/local_predictability/backbone_v2/h100_repeats.json --stage run` follows the
H100 screen with matched GATr onset parent/snapshot/history/repeated-frame fits,
using the existing v2 trainer and completed MACE reference. `--stage compare-screen`
only checks and exports the completed H100 speed/physical comparison. See the
[H100 extension](../docs/backbone_v2.md#h100-comparison-and-onset-repeats).

`python -m src.research.local_predictability.plan --config
configs/local_predictability/two_gpu_16h.json --output /tmp/local-predictability-queue.json`
validates the one-seed, two-worker 16-hour **planning specification** and writes
job identities. It never trains, submits or resumes jobs. Execution uses
`python -m src.research.local_predictability.data --config configs/local_predictability/h100_execution.json`
for the outcome-independent full-timeline release, followed by `audit` with the
same configuration and `baselines --config configs/local_predictability/rtx6000_baselines.json`
in that package. The maintained tracking wrapper waits for explicit stage records.
Native integration follows the H200 tested implementation; see the
[H100/H200 handoff](../docs/local_predictability_16h.md).

`python -m src.research.local_predictability.observability --config
configs/local_predictability/rtx6000_observability.json` fits current-state and
true-future packet diagnostics on the frozen native window grid. It saves model
checkpoints and predictions for later analysis; future inputs are diagnostic only.
See the [observability protocol](../experiments/local_predictability_20260917/OBSERVABILITY.md).

`python -m src.research.local_predictability.native_queue --config CONFIG
--stages snapshot history12 --worker h100 --resume` assigns disjoint frozen
continuations to workers and raises the process open-file limit for all 150
memory-mapped sources. The RTX worker uses `--stages repeat12 --worker rtx6000`.
`raw_observability --config configs/local_predictability/rtx6000_raw_observability.json`
in the same package trains the all-state raw-atom current-label check.
`native_readouts --config configs/local_predictability/rtx6000_native_readouts.json`
extracts completed onset states and fits fresh linear/MLP predictors. These
diagnostics preserve predictions for later interpretation. The
[operational queue](../docs/local_predictability_16h.md) uses explicit dependencies.

`run_lammps_campaign.py memory-sources prepare --config
configs/simulation/predictive_memory_precision.json --run-name NAME` stages fresh,
split-assigned independent Al sources. `memory-sources run-worker --campaign-root
ROOT --worker-index INDEX --workers COUNT` runs a fixed shard in a 48-rank CPU
allocation. Its paired exports use `convert_trajectory.py memory-pair RUN_DIR`.
See the [simulation record](../docs/simulations/predictive_memory_precision_20260917.md).

`python -m src.data.predictive_memory.prepare --config configs/predictive_memory/pilot.json`
audits existing trajectories and caches strictly partial observations and continuous
physical targets. `python -m src.training_methods.predictive_memory.train --config
configs/predictive_memory/pilot.json --history-ps 48 --velocity` trains the new
label-free memory pilot. `python -m src.training_methods.predictive_memory.compare
--config configs/predictive_memory/pilot.json` compares the eight completed matched
fits. See [workflow and resume](../docs/predictive_memory.md).
For new batched training, use `configs/predictive_memory/batched.json` with the
cuEquivariance backend enabled and the same CLI. It exposes effective/micro/evaluation
batch sizes and bounded GPU input
residency. Current checkpoints use format 2; earlier runs use their original commit.
The comparison's `--modalities xv` option evaluates the four-fit velocity-input
replicate from `configs/predictive_memory/replicate-xv-seed20260918.json`.
`python -m src.training_methods.predictive_memory.diagnose --config CONFIG
--modalities xv` audits frozen heads under constant-state interventions and fits
train-only ridge controls for physical future/present prediction. It writes
separate diagnostics and does not retrain or replace the native encoder.
The `configs/predictive_memory/optimization/` recipes use the same trainer for
matched 12,000-update original/stronger-present-loss fits across two seeds.

`python -m src.research.memory_report --config configs/analysis/memory_research_report.json
--output output/predictive_memory/NEW-SNAPSHOT` freezes completed causal/memory
results, reported H200 means, diagnostic tables and plots without retraining.
It refuses to overwrite a prior snapshot and does not average incomplete seed
cohorts. The dated interpretation is reviewed separately; see the
[consolidated report](../output/predictive_memory/research-summary-20260917/RESULTS.md).

`python -m src.research.mace_velocity causal-prepare --config configs/mace_causal/pilot.json`
prepares identity-preserving physical history/future examples.
`causal-train --config configs/mace_causal/pilot.json --variant D --device cuda:0`
trains the native tensor MACE state; variants A–E and `repeated_anchor` share
data and training controls. See [workflow](../docs/mace_causal.md).

Frozen-feature replacement-embedding workflows were [discarded](../docs/discarded_frozen_encoder_maps.md)
on 16 September 2026. Their source snapshot, recipes and results remain available
for historical reproduction. Embedding forecasting and native encoder training
remain active.

```bash
python -m src.training_methods.contrastive_learning.train_contrastive --config-name vicreg_mace_relaxed
python -m src.analysis.pipeline configs/analysis/static_topology.yaml --checkpoint CHECKPOINT --output-dir output/QUESTION/RUN
python -m src.training_methods.embedding_forecast --config CONFIG.json --stage train
python -m src.data.topology_views --config-name vicreg_mace_relaxed
```

Joint MACE context checkpoints use the existing `src.research.mace_context.run`
stages `static-export` and `static-verify`, followed by `src.analysis.pipeline`
with `configs/analysis/static_mace_context_al.yaml` or `static_mace_context_zr.yaml`.
See [static context analysis](../docs/mace_context_static.md).

Selected snapshot structural GATr–VICReg encoders use
`python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_static.json
--stage export|verify`, then the same `src.analysis.pipeline` command with
`configs/analysis/static_structural_gatr_al.yaml`. The adapter preserves the
native full neighborhood, species and fixed material scale; see
[structural static analysis](../docs/structural_static_analysis.md).

The newest Al-only v6 GATr uses the same commands with
`configs/analysis/structural_gatr_v6_static.json` and
`configs/analysis/static_structural_gatr_v6_al.yaml`, in `pointnet-torch214`.
Verification includes replay of the saved compiled selection features.

The frozen latest mixed GATr checkpoint uses the same stages with
`configs/analysis/structural_gatr_backtracking_static.json`, then the pipeline
with `configs/analysis/static_structural_gatr_backtracking_al.yaml`. The recipe
pins an immutable `last.pt`; verification compares its own native compiled
encoder, since a latest optimizer checkpoint is distinct from a selected best.

The final local MACE/GATr step-622 reruns use the same stages with
`configs/analysis/structural_{mace,gatr}_local_latest_static.json` and
`configs/analysis/static_structural_{mace,gatr}_local_latest_al.yaml`.
Execute in each result's frozen `technical/runtime`; see the
[local static protocol](../experiments/shared_pretraining_20260918/STATIC_AL_LOCAL.md).

The matching Al-only v6 MACE checkpoint uses those same export/verify stages
with `configs/analysis/structural_mace_v6_static.json`, then the pipeline with
`configs/analysis/static_structural_mace_v6_al.yaml`. It preserves MACE's native
per-observation graphs and cuEquivariance/BF16 execution in `pointnet-torch214`.

`python -m src.research.mace_context.cluster_diagnosis --config
configs/analysis/mace_context_clusters.json` runs saved-feature clustering
ablations and a matched GeoFrame V2 comparison. See the
[liquid-cluster diagnosis](../experiments/mace_context_clusters_20260915/README.md).
The follow-up `python -m src.research.mace_context.cluster_probe --config
configs/analysis/mace_context_clusters.json` tests frozen features with spatially
separated physical readouts using the same diagnostic samples.

The MACE preparation command is
`python -m src.research.spatiotemporal.prepare_spatiotemporal_vicreg_views --config CONFIG`.
Forecast embedding storage migration uses `convert_trajectory.py embedding-cache`;
see the [conversion protocol](../docs/trajectory_conversion.md).
Forecast training accepts `--runtime-config RUNTIME.json` with
`{"loader":"resident","log_every_steps":100}` to retain stored embeddings on
the compute device and report batch progress. The default `mmap` loader remains
available. Explicit `--resume-transition TRANSITION.json` permits a reviewed,
exact old/new implementation hash transition while restoring the saved training
state and preserving the scientific configuration.
`python -m src.training_methods.embedding_forecast.handoff --plan PLAN.json
--variant NAME` replaces a tracked forecast process inside its existing Slurm
allocation. Its plan supplies exact job/node identities, the original execution
record and checkpoint directory, and the replacement run specification. See the
[GPU restart record](../experiments/embedding_forecast_20260911/GPU_RESIDENT_RESTART_20260913.md)
for the plan schema, allocation holding, and guarded collection semantics.
Observed-history sweeps use the same forecast training command with per-context
configuration files. The distinct paired comparison is
`python -m src.research.forecast_context.compare --plan PLAN.json`; it verifies
matched windows and normalization across histories and exports source-weighted
scores, paired intervals and plots. Inputs are completed forecast artifacts and
a plan listing their configs, seeds and result root. See the
[context experiment](../experiments/forecast_context_20260913/README.md).
Local crystallization assessment uses the exact tracked center atoms and existing
PTM implementation: `python -m src.research.forecast_crystallization.local_data
--config CONFIG`, then `local_predict` and `local_analyze` in the same package.
Inputs are an existing forecast cache, source trajectories and frozen checkpoints;
outputs are local physical labels, a train-only crystal readout, paired forecasts,
transition/timing tables and plots. See the
[local crystal assay](../experiments/forecast_crystallization_20260913/README.md).

Spatial/mixture forecasting uses the same training command. Runtime setting
`validation_residency: host` retains validation inputs in RAM to leave GPU space for
the training split and large batches. `validation_residency: staged_device` copies
that split to the GPU once per validation pass and releases it before backpropagation;
see [performance measurements](../docs/forecast_performance.md). Prepare periodic
observed-frame neighbor indices with
`python -m src.training_methods.embedding_forecast.spatial --config CONFIG`.
Its config declares the existing embedding cache, simulation source manifest,
neighbor count and output. Training variants select spatial inputs and a Gaussian
mixture over the complete future path; spatial training requires the resident loader.
`prepool_embeddings: true` stores hashed same-frame neighbor means in the cache
dtype, allowing broad neighborhoods to share the same GPU batch cost. Preparation
publishes `status.json` for queued consumers and accepts resolved storage tokens.
The [spatial-context recipes](../configs/embedding_forecast/spatial-context/analysis.json)
compare 32/128/512 centers against the existing 8-center control.
Learned individual-neighbor attention uses a separate geometry sidecar, prepared by
`python -m src.training_methods.embedding_forecast.spatial --stage geometry --config
configs/embedding_forecast/spatial-attention/geometry.json`. Its training configs
select `spatial_attention` and `data.geometry_cache`; the same forecast training and
physical-analysis commands apply. `training.micro_batch_size` accumulates gradients
within the unchanged effective batch. Runtime `validation_residency: swap_device`
moves attention training tensors to host during validation; `evaluation_batch_size`
sets the validation/test batch independently. See the
[attention protocol](../experiments/forecast_spatial_attention_20260914/README.md).
After collection, `python -m src.research.forecast_spatial_mixture.compare --stage
summarize --plan configs/embedding_forecast/spatial-context/report-20260914.json`
creates a fresh six-figure gallery of spatial size, paired uncertainty, timing,
history, trajectory quality and temperature strata from completed artifacts.
For completed fits, `python -m src.research.forecast_spatial_mixture.evaluate
--plan PLAN --run NAME --seed SEED` reuses the frozen local PTM assay and scores
both mean paths and probabilistic crystal readouts. Then `compare --plan PLAN` in
the same package verifies matched windows/scales and exports the full paired study.
See the [spatial/mixture experiment](../experiments/forecast_spatial_mixture_20260913/README.md).
`python -m src.research.forecast_spatial_mixture.trajectories --config CONFIG`
visualizes measured local point-cloud evolution, observed embedding trajectories,
and alternative futures from completed mixture checkpoints using one train-fitted UMAP.
The full-trajectory overview separates occupancy from time; feature heatmaps explain
changes in the frozen crystal score with training-ranked channels. The
[`trajectory-visuals.json`](../configs/embedding_forecast/trajectory-visuals.json)
recipe exports PNG/PDF figures, a GIF, offline 3D time sliders and exact plotted
arrays to a fresh output. It uses CPU inference on selected diagnostic examples.
`--stage plots` refreshes figures from that output's retained extraction without
loading checkpoints or simulation trajectories again.
`python -m src.training_methods.embedding_forecast.allocation --plan PLAN` runs
explicit training/analysis module commands serially inside an existing allocation.
Its plan specifies node/job identity, prerequisite status files, exact arguments,
completion states and an allocation deadline; it never submits or implicitly resumes
a job. An optional `status_path` publishes progress at an existing dependency's
location after its previous controller has been retired and its status archived.
The paired collector accepts explicit `comparison_pairs` and `reference_runs`
with retained `local_directory` paths for the [short-history extension](../experiments/forecast_spatial_mixture_20260913/SHORT_HISTORY.md).

Recorded older protocols live in [src/research](../src/research/README.md); their
configuration and findings stay with the dated experiment record.

The frozen forecast encoder assay runs with
`python -m src.research.mace_encoder_diagnostics.extract --config
configs/analysis/mace_encoder_diagnostics.json --stage all`, followed by
`python -m src.research.mace_encoder_diagnostics.analyze --config
configs/analysis/mace_encoder_readout.json`. It tests input invariance, geometric
and membership changes, storage precision, TDA generalization and physical time
dependence using retained checkpoints and trajectories. See the
[scientific protocol](../experiments/mace_encoder_diagnostics_20260914/README.md).
`python -m src.research.mace_encoder_diagnostics.verify --config
configs/analysis/mace_encoder_diagnostics.json` checks producer labels, replays
stored forecast embeddings and validates sibling velocities and timelines.

`python -m src.research.mace_tda_ridge_audit.run --config
configs/analysis/mace_tda_ridge_audit.json` re-encodes the original six single-frame
VICReg/TDA checkpoints, checks the training loss with target interventions, and
compares repository and independent ridge calculations for projector and encoder
features. `--stage summarize` rebuilds its tables and paired plot. See the
[audit protocol](../experiments/mace_tda_ridge_audit_20260914/README.md).

The same entry point with `--config configs/analysis/mace_tda_initialization.json
--stage initialization` compares frozen original MLIP and three random MACE
encoders with the audited trained encoder features. It retains the original ridge
protocol and adds a validation-only regularization sensitivity analysis.
With `--config configs/analysis/mace_tda_direct.json --stage direct`, it compares
verified random/MLIP embedding distances and top-10 neighborhoods directly with
the three TDA blocks on held-out structures. This stage fits no readout or scaling;
it reports both global and within-frame geometry, including shuffled controls.

Completed one-off simulation specializations, recovery scripts and temporary
launchers were removed on 2026-10-11. Use the maintained campaign dispatcher and
conversion commands; recover older implementations from Git when designing a
new shared workflow. Historical protocols, arguments and completion receipts
remain in [simulation records](../docs/simulations/README.md).
The historical Aluminum shell launcher requires `PYTHON` and an explicit
`CAMPAIGN_CONFIG`, and accepts `DEVICES`; restore its dependency tree as described
in the [config index](../configs/README.md). The historical FactorVAE queue and
spatiotemporal trainer accept `--config-dir` for restored recipes.
All maintained simulation recipes live in `configs/simulation/`.

`project.py` manages machine settings (`paths`, `doctor`), dataset IDs (`datasets`), JSON resolution (`resolve`), verified selected exports (`bundle`, `verify-bundle`), full checkout snapshots (`snapshot`), completed simulation publication (`publish-simulation`), and stopped failure archives (`archive-failed-simulation --inactive`). Inputs and copy semantics are documented in [portability](../docs/portability.md); implementation is in `src/project_runtime/`.

`project.py simulations --output docs/simulations` exports readable collection and
producer-outcome CSV indexes from `configs/datasets.json`; implementation is in
`src/project_runtime/simulation_inventory.py`. It preserves failed/duplicate attempt
evidence and never treats outcome records as counts of independent simulations.

Source ownership and retained historical import/command paths are documented in
[the source refactor record](../docs/src_refactor.md#completion-pass-starting-at-d119fd2).

`python -m src.research.mace_context.run --config configs/analysis/mace_context.json --stage STAGE` runs the [complete-context MACE pilot](../experiments/mace_context_20260914/README.md). Stages are `prepare`, `verify`, `frozen`, `train`, `labels`, `summarize`, and `smoothness`; extraction/training require an explicit `--mode`. Existing allocation plans can run these commands on assigned GPUs. The CPU-only `smoothness` stage reuses all seven completed feature extractions and writes a separate run suffixed `-smoothness`, with temporal normalization checks, paired source intervals and boundary curves.

The [information-recovery follow-up](../experiments/mace_context_recovery_20260914/README.md)
uses the same module with `--config configs/analysis/mace_context_recovery.json`
and stages `recovery-readouts`, `recovery-verify`, `recovery-train --variant
dual_ssl|dual_physics`, and `recovery-summarize`. Cached readouts compare linear
and nonlinear decoders and inner/center fusion. The distinct joint experiment
trains a shared backbone with or without encoder gradients from physical targets.
# Coordinate/velocity local-state research

The same `src.research.mace_velocity` entry point has `data-prepare`, `data-smoke`
and `data-study` stages for the distinct **end-to-end native encoder** training-data
comparison, using `configs/analysis/mace_data_amount.json`. This updates MACE itself;
it does not train a map on frozen features. See [run instructions](../docs/mace_data_amount.md).

Use `python -m src.research.mace_velocity STAGE --config configs/analysis/mace_velocity.json`
for the separate local phase-space protocol (`inventory`, `prepare`, `verify`,
`teacher`, `train`, `evaluate`). The maintained conversion dispatcher adds
`python scripts/convert_trajectory.py paired-velocity --positions P --velocities V --output O --atoms N`
for atom/time-matched legacy dumps. See [the run instructions](../docs/mace_velocity.md).

`python -m src.research.mace_causal_comparison --config configs/mace_causal/comparison.json`
collects the completed matched-seed pilot into paired physical-error tables and
plots; it verifies source/center/anchor/target identity across every readout.

`python -m src.research.mace_velocity causal-benchmark --config configs/mace_causal/runtime-benchmark.json --device cuda:0`
measures original versus packed/resident execution on verified histories.
`causal-probe --probe-modes linear nonlinear` selects cheap frozen readouts;
`--probe-modes state_constant state_history` selects the matched history-access pair.
See [execution and data handoff](../docs/mace_causal_runtime.md).

The compiled VICReg restart uses the existing shared-pretraining queue and separate profiler; see [the recipe](../docs/shared_pretraining_compiled_repair_20260918.md). No benchmark runs inside training.

The active Al-only VICReg repair uses `configs/shared_pretraining/al_stable/`
with the existing shared-pretraining queue; see [stability and calibrated
checkpoint export](../docs/shared_pretraining_al_stability_20260918.md).

The GATr continuation uses the same `shared_pretraining.queue submit --plan
configs/shared_pretraining/broad_full_tda/campaign.json`. It submits CPU expansion
through the existing `src.data.structural_pretraining.prepare --config` command,
then a GPU job dependent on successful target completion. See the
[full-TDA workflow](../docs/shared_pretraining_broad_full_tda_20260918.md).

The dynamic-only mixed-material GATr recipe reuses `shared_pretraining.queue
submit --plan configs/shared_pretraining/gatr_mixed_triplets/campaign.json`.
See [grouped normalization and detached execution](../docs/shared_pretraining_mixed_triplets_20260918.md).

`python -m src.research.crystallization_transfer.queue submit --config configs/crystallization_transfer/mace_20260919.json` freezes and detaches four existing GPU lanes for the expanded MACE onset study; [protocol](../experiments/crystallization_transfer_20260919/README.md), [execution](../docs/crystallization_transfer_20260919.md).

For context-radius and training/data-size sweeps, use the same crystallization
transfer queue with `--config configs/crystallization_transfer/mace_scaling_20260919.json`.
It waits for the original GPU lanes and reuses their immutable cache; see the
[scaling protocol](../experiments/crystallization_transfer_20260919/README.md).

Completed crystallization queues can be compared without further fitting:
`python -m src.research.crystallization_transfer.report --initial INITIAL_OUTPUT
--scaling SCALING_OUTPUT --output SUMMARY_OUTPUT`. This verifies paired test
identities and exports all fits, horizons, source-bootstrap NLL differences and
scaling curves; [metric definitions](../docs/metrics/crystallization_transfer_summary.md).

The corrected trainable-encoder attention study uses the same queue with
`--config configs/crystallization_transfer/mace_adaptive_20260919.json`; see the
[adaptive protocol](../experiments/crystallization_transfer_20260919/ADAPTIVE.md).
The transfer queue's `continue` command accepts the separate operational recipe
`configs/crystallization_transfer/mace_adaptive_continuations_20260919.json` to
submit dependent slots loading the existing frozen campaign.

`python -m src.research.crystallization_paths.queue submit --config
configs/crystallization_transfer/mace_paths_20260919.json` adds frozen-MACE direct,
autoregressive, mixture and diffusion trajectory forecasts on the same source
split. It uses existing simulation data, GPU-resident timelines and an independent
detached queue; see [execution](../docs/crystallization_paths_20260919.md) and the
[scientific protocol](../experiments/crystallization_transfer_20260919/PATHS.md).

The same path queue accepts `configs/crystallization_transfer/mace_path_refinement_20260919.json` for 30 targeted screens and five promotions. `python -m src.research.crystallization_paths.diagnose` replays completed forecasts on selection sources; see [execution](../docs/crystallization_paths_20260919.md).

`python -m src.training_methods.neighborhood_jepa.prepare --config configs/neighborhood_jepa/data_20260920.json`
builds the tracked paired-graph release reused by retained label-free studies.
The old conditional JEPA training and profiling entry points are retired;
[retained encoders and data tools](../docs/architecture_retirement.md).

Newly trained paired-MACE snapshot checkpoints use
`python -m src.analysis.neighborhood_adapter --config CONFIG --stage export|verify`,
followed by the analysis pipeline. The adapter reads `mace_paired_epi_v1` and
exports `neighborhood_snapshot_encoder`; old JEPA checkpoint schemas are retired.
See [current encoder APIs](../docs/architecture_retirement.md).


`python -m src.research.crystallization_information --config configs/analysis/crystallization_information_short.json`
measures short-horizon crystallization information missing from frozen exports using
matched feature add-backs and physical decoders. See [workflow](../docs/crystallization_information.md).

`python -m src.research.relaxed_encoder.queue cpu|build|evaluate|report|benchmark --config CONFIG`
retains paired-data preparation and frozen-checkpoint diagnostics. Encoder fitting,
expanded training submission, and the context-night continuation queue are retired;
[details](../docs/architecture_retirement.md).

`python -m src.hardware_benchmark.relaxation --config configs/benchmarks/relaxation_cuda.json --backend {cpu,v100,h100} --binary /path/to/lmp`
benchmarks paired full-cell MEAM relaxation and target fidelity, separately from
training; see [GPU relaxation](../docs/hardware_benchmark.md#full-cell-meam-relaxation-on-cuda).

`python -m src.research.relaxed_encoder.accelerated --config configs/analysis/relaxed_encoder_accelerated.json --backend {h100,a100,v100} --lane NAME`
adds validated CUDA MEAM producers to the shared CPU task queue; see
[GPU production](../docs/relaxed_encoder.md#gpu-production-alongside-cpu-workers).

`python -m src.training_methods.bcr prepare|verify|train|evaluate|probes|select --config CONFIG` runs the isolated bottleneck-conditioned reconstruction workflow; see [execution](../docs/bcr.md) and [protocol](../experiments/bcr_v1_20260921/README.md).

`python -m src.research.bcr_pilot.queue prepare|preflight|submit --config configs/bcr/pilot_20260921/study.json` runs the independent-root three-arm conditioning pilot; [operations](../docs/bcr_pilot.md).

`python -m src.research.crystallization_paths.figures --config configs/analysis/crystallization_figures.json`
exports paper-style PNG trajectory comparisons, training-fitted UMAP projections,
and real-point-cloud context diagrams from completed forecasts. CPU replay only;
no predictor fitting. Separate captions and input identities accompany the gallery.
Use `--stage render` to adjust appearance without repeating extraction or UMAP.

`python -m src.research.structured_context.queue verify|verify-pipeline|submit|worker|report --config configs/crystallization_transfer/symmetric_mace_gatr_20260921.json` runs the symmetric-context forecast comparison; see [workflow](../docs/structured_context.md).

`python -m src.research.structured_context.figures --config configs/analysis/structured_context_figures.json`
replays the completed eight forecasts, regenerates PNG/UMAP/context figures, and
compares the same onset events and matched controls across forecast origins.
Use `--stage render` for appearance-only changes; [metric definitions](../docs/metrics/structured_figures.md).

The selected relaxed cold-vic-temp01 model uses `src.analysis.neighborhood_adapter` export/verify with `configs/analysis/relaxed_best_static.json`, followed by `src.analysis.pipeline configs/analysis/static_relaxed_best_al.yaml`. See [protocol](../experiments/relaxed_encoder_expanded_20260921/STATIC_AL.md).

Relaxed symmetric-context MACE forecasts use the module entry point
`python -m src.research.structured_context.relaxed`; see
[operations](../docs/structured_relaxed_context.md) and the
[research protocol](../experiments/structured_relaxed_context_20260921/README.md).
The active archived-data route is `python -m src.research.structured_context.reuse`;
it replaces fresh-quench preparation for this comparison and runs no simulations.

`python -m src.research.bcr_followup.queue preflight|submit|run|report --config configs/bcr/followup_20260922/study.json` runs the eight-hour frozen-BCR follow-up and paired relaxed-data audit; [operations](../docs/bcr_followup.md), [scientific protocol](../experiments/bcr_followup_20260922/README.md).

Matched relaxed-versus-original MACE PNG analysis:
`python -m src.research.structured_context.reuse_figures --config configs/analysis/relaxed_reuse_figures.json`.
Use `--render-only` to redraw checked artifacts without GPU inference or UMAP fitting.
Captions and the HTML gallery are saved alongside the PNG files.

`python -m src.research.crystallization_followup.queue prepare|verify|submit|worker|report --config configs/crystallization_transfer/literature_followup_20260922.json`
runs the paired-input, actual-time and dense-history frozen-MACE experiments;
[workflow](../docs/crystallization_followup.md), [literature and protocol](../experiments/crystallization_followup_20260922/README.md).

`python -m src.research.crystallization_followup.front --config configs/crystallization_transfer/crystal_front_20260922.json --after output/crystallization_transfer/literature-optimization-20260922`
derives observed bond-coherent cluster descriptors from existing MD, then validates
and launches the front queue after the optimization queue; no new simulation.

`python -m src.research.bcr_followup_analysis --config configs/analysis/bcr_followup_20260922.json` analyzes completed BCR interventions, frozen structural probes and fresh decoders, with root-bootstrap uncertainty and a static summary figure; [findings](../experiments/bcr_followup_20260922/RESULTS.md).

`python -m src.research.liquid_geometry.study prepare|model|latest|checkpoints|report --config configs/analysis/liquid_geometry_20260922.json` runs frozen metric interventions, exact-cohort forecast associations and native early/late checkpoint diagnostics; see [workflow](../docs/liquid_geometry.md).

Native snapshot physical screen: `python -m src.research.encoder_screen.queue --config configs/encoder_screen/screen_20260923.json --lane NAME`; [preparation, cached figure pass and continuation](../docs/encoder_screen.md).

The mixed GeoFrame/reconstruction-MACE parameter-search launcher is retired.
The standalone GeoFrame fit and saved-result analysis modules remain;
[historical campaign reproduction](../docs/architecture_retirement.md).

`python -m src.research.mace_epi.queue submit --config configs/mace_epi/campaign.json` runs the paired MACE VICReg/Epi comparison; see [operations](../docs/mace_epi.md) and [scientific protocol](../experiments/mace_paired_epi_20260923/README.md).

`python -m src.research.nucleus_recovery prepare|launch|summarize --config configs/simulation/nucleus_recovery_20261004.json`
runs known-nucleus saved-state recovery and paired Q/V rounding controls; [workflow](../docs/simulations/nucleus_recovery_20261004.md).

`python -m src.research.nucleus_controls.queue prepare|submit|assist|summarize --config configs/simulation/nucleus_controls_20261005.json`
compares dissolved embryos, size/temperature matched establishing seeds and present-only random liquid under a shared 16-hour cap; [workflow](../docs/simulations/nucleus_controls_20261005.md).

`python -m src.research.nucleus_controls.negative prepare|launch|summarize --config configs/simulation/nucleus_negative_extra_20261005.json`
adds 64 liquid ancestors and fresh shots for the 12 failed embryos under a separate three-hour cap; [workflow](../docs/simulations/nucleus_negative_extra_20261005.md).

`python -m src.research.nucleus_controls.bias --config configs/analysis/nucleus_continuation_bias_20261005.json`
audits paired structure and thermodynamics with common observation precision.
`python -m src.research.nucleus_controls.calibration prepare|launch|summarize --config configs/simulation/nucleus_rate_calibration_20261005.json`
runs the present-selected birth-rate check; [workflow](../docs/simulations/nucleus_continuation_bias_20261005.md).
`assist --frozen-code FROZEN_CODE --gpus 4` adds leased workers within that same deadline; scheduler partitions and excluded nodes are explicit optional arguments.

`python -m src.research.ni_ordering_flow.queue launch --config configs/ni_latent_sim/ordering_20261019.json` runs the matched overnight history/context queue.

`python -m src.simulation.campaigns.ni_ordering_branches launch --config configs/simulation/ni_ordering_branches_20261019.json` starts the separate Ni continuation pilot.
