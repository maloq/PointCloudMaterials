# Datasets: start here

**Ni observed-history/context extension (October19, collecting):** `ni-ordering-histories-20261019` retains21,600native windows and19,800birth-assay rows, extends observed history to1.2ps at0.1ps, and adds12A regional features. Same central atom; nearest80 is reselected at every observed frame for native and birth inputs. Training ancestry28/test62 unchanged. [Workflow](docs/ni_ordering_flow.md).

**Ni native source expansion (October 18, complete):**
`ni-native-expansion-20261018` measures 7,200 additional fixed-central-atom
windows from the existing thirty October 12 histories: 240 windows per source,
four observations through the present and sixty future observations at exact
0.1 ps spacing. Anchor times and atom IDs are selected without future or phase
labels. The source roles remain 12 train / 18 test; this creates no new MD.
The matched original-window comparison uses 80 windows per training source:
1,280 windows / 16 original sources versus 2,240 / 28 original plus additional
sources. All 4,320 additional test windows remain available alongside the
10,560 existing native test windows. Normalization reuses the original sixteen
training sources unchanged. The added heldout sources appeared in earlier birth
assays; they are not a previously untouched benchmark. Completion and checksums:
`${storage:training_storage}/ni_latent_flow/validation-20261018-v2/technical/expansion-complete.json`.

**Ni spatial context (October 15, preparing):** `ni-spatial-context-20261015`
adds 87 regional density/motion measurements at 8/12/20 Angstrom to the existing
14,400 native windows and 19,800-row birth assay. The deduplicated extraction
covers 87 existing sources, 52,578 source frames and 999,971 center observations.
No new MD, independent sources or birth filtering; train-only transforms and
original ancestry are retained. [Workflow and completion receipts](docs/ni_spatial_scales.md).

**Ni kinetic-latent measurements and annotations (October 15):**
`ni-kinetic-measurements-20261015` reuses all 14,400 native fixed-central-atom
measurement windows through 6 ps, with train-only affine/asinh normalization.
Train: 3,840 windows / 16 ancestors; pooled test: 10,560 / 44. Exact full-cell
PTM-derived center labels, local fractions and tracked component sizes are a
separate resumable annotation stage. The existing 19,800-row 90-source nucleus
assay remains separate. No new raw MD or independent sources are created.
[Workflow and provenance](docs/ni_kinetic_latent.md).

**Ni phase-model birth readouts (October 13):** `ni-phase-birth-probes-20261013`
reuses the original sixty-source nucleus assay and adds the completed thirty-source
October 12 catalogue. Exact 1/3/6 ps leads, persistent 32/64-atom nuclei, and
the phase models' 28-train/62-test ancestry. [Workflow](docs/ni_phase_sim.md#nucleus-readout-follow-up).

**Dense Ni phase-space teachers (October13, complete):** `ni-phase-nve-20261013` contains630 deterministic100-fs full-cell continuations for direct Q,V and latent-tangent learning, with fixed80 atom identities every5fs. Train320/28 sources; test310/62, preserving the pooled split. [Workflow](docs/ni_phase_sim.md).


**Ni density rates and physical-source expansion (October13, preparing):**
`ni-density-rates-measurements-20261013` reuses300 physical anchors and adds451,
with original244 future measurements plus74 initial density-rate features.
`ni-density-rates-response-20261013` stores the new100-fs full-cell continuations.
Training banks are80/16,320/16 and320/28 states/sources;310 test anchors cover all
62 nontraining sources. New experiments default to train/test only, preserving
all original roles as provenance. [Workflow](docs/measurement_sim.md#density-rates-source-expansion-and-smaller-steps--october-13).

**Thirty additional Ni melts (October 12, simulations complete):**
`ni-births-lowT-010-20261012` adds five fresh 32,000-atom sources at each of
950/1000/1050/1100/1125/1150 K. All 30 sources completed, with 73,780 frames and
7.375 ns summed production history. New source IDs and seeds
are disjoint from the previous 60 sources, with 12 train / 6 selection / 12 test
roles fixed in advance. Positions and velocities are saved every 0.10 ps;
holds are capped at 400 ps, with the existing crystallinity stop.
`ni-lowT-20261012-gpu-full-v1` is the GPU descriptor and CPU lineage search,
requeued after a missing frozen audit file stopped the initial extraction;
new birth counts are not yet available. [Protocol and current job receipts](docs/simulations/ni_births_lowT_010_20261012.md).

**Direct Ni measurements (October 11):** `ni-direct-measurements-20261011`
contains 14,400 existing fixed-atom sequences remeasured as BO40/TDA113/density74/velocity17,
plus 300 dense physical-response anchors. No learned encoder or new MD; ancestry, roles
and native cadences are retained. See [workflow](docs/measurement_sim.md).


**Dense Ni response paths (October11, complete):**
`ni-dense-path-response-20261011` retains physical local80 values and tangents
every5fs through100fs from the original300 anchors and60 Ni sources. It reuses
original split roles and generating potential, and is not a new independent
source cohort. Motion fitting reuses the expanded14,400 fixed-atom sequences.
All300 physical paths and encoded derivative target packets are now available.
[Protocol and storage](docs/ni_latent_sim.md).

**Ni motion collection expanded threefold (October10, preparing):**
`ni-latent-sequences-regimes3x-20261010` retains4,800 original local sequences and
adds4,800 liquid-persistent plus4,800 advancing-front windows from the same60 MD
ancestors. Every sequence follows one fixed central atom ID at native0.1ps,
with0.3ps history and6ps future. Existing source roles and birth probes remain
unchanged. [Sampling and readiness receipts](docs/ni_latent_sim.md).

**Compact Ni simulator pilot (October10):** `ni-latent-sequences-20261010`
prepares exact0.1ps local trajectories from the60 pinned Ni sources, preserving
ancestry and roles. `ni-latent-lifting-20261010` records separate physical
continuations for an empirical decoder diagnostic. [Protocol](docs/ni_latent_sim.md).

**Ni quench repeatability (October9):** `ni-quench-repeatability-20261009` contains
controlled repeats of five existing Ni response endpoints. It preserves source
ancestry and is a numerical diagnostic, not new independent motion-training data.
[Protocol](docs/ni_relaxation_repeatability.md).

**Matched relaxed Ni response (October9):** `ni-response-esen-relaxed-20261009` preserves the unrelaxed motion/evaluation rows and original MD labels, while quenching encoder observations and targets. [Definition and execution](docs/ni_relaxed_response.md).

**Ni eSEN response (October 9):** `ni-response-esen-20261009` prepares 19,200 native0.10ps Q,V observations from all60 completed Ni trajectories. Separate evaluations cover first crystalline appearance and persistent32/64-atom liquid-origin nuclei. Original ancestry/roles retained; no relaxed inputs or broad spatial replay. [Protocol](docs/ni_response.md).

**Ni birth-window GPU relaxation (October 8):**
`ni-birth-relaxed-windows-20261008` prepares fixed-box full-cell FIRE minima for
native 0.10 ps windows from -6 to +6 ps around 70 established PTM liquid-origin
births in the original, extension and completed low-temperature catalogues.
Overlapping source frames are deduplicated; original ancestry/roles remain.
Native restart states enter the shared relaxation index. This is a requested
derived collection; per-cell receipts determine availability.
[Protocol, precision and execution](docs/simulations/ni_birth_relaxation_20261008.md).

**Al 0.10-ps birth update, October 8:** the seven completed original V100
branches are incorporated without recomputation: `al-nuclei-birth-010` now
contains **129 views / 312 established liquid-origin PTM records**. The
[dataset guide](docs/datasets/nuclei_birth.md) records its fingerprint. All
152 main dense simulations and all 33 birth-focused branches are complete.
The remaining 45 main trajectories plus those 33 branches are being analyzed
on two A100s (165,828 frames), with dependent CPU lineage and automatic
catalogue publication. Source ancestry, roles and native-only `030` are retained.
[Queue and provenance](docs/al_nuclei_lineage.md#october-8-completed-data-incorporation).
Earlier dated entries below describe their original launch state.

**Current spatial pretraining (October 8):** `spatial-training-ni20` now points to
the **completed crystal/interface extension**: 1,373,798 training rows, including
317,030 Ni (23.08%). It retains the previous fitting rows and adds 52,838 Ni
observations, half crystal interiors and half interfaces. Validation and test
are merged into one 248,081-row evaluation set, including 9,830 Ni after 1,638
phase-selected additions. Training/evaluation ancestry remains disjoint. All
198 bond-order/CNA/TDA targets are prepared; 197 pass the training-variation mask.
[Current protocol and consumer configuration](docs/datasets/spatial_ni_enriched.md).

**Original 0.75-ps Al campaign retired (October 7):** the production arrays of
all 150 trajectories were deleted at the user's request, releasing **99.624 GiB**.
Prepared liquids, native melt restarts, source identities and historical analyses remain.
The 150 production `final.restart.bin` files were subsequently deleted too.
The three registry entries are `retired` / `provenance_only`; their remaining
single-frame preparation archives are not usable production histories.
Use the separate consolidated 0.10/0.30-ps birth datasets for new birth work.
[Removal record and scope](docs/storage/al075_removal_20261007/README.md).

**Birth-focused V100 handoff (October 7 evening):**
`al-birth-focused-v100-20261007` replaces 33 unstarted branches from the earlier
40-branch V100 plan, retaining parents, seeds and roles. Q/V starts at initialization
every 0.10 ps; a coarse cluster-growth trigger retains 12 ps follow-up. Three
completed and four active original branches keep their protocol. Four replacement
jobs share the existing lane deadlines; planned branches are not completed data
or new independent ancestors. Birth labels remain not yet searched.
[Protocol, initial-state audit and jobs](docs/simulations/al_dense_shooting_v100_20261007.md#birth-focused-handoff-october-7-evening).
The original seven branches have since completed. The replacement launch failed
before MD because its frozen snapshot omitted a scheduler helper; the
[repair revision](docs/simulations/al_dense_shooting_v100_20261007.md#missing-dependency-repair-october-7-evening)
requeues the same 33 unstarted branches and retains the original deadlines.
The launch's `repair.json` identifies current receipts; there is no new ancestry.

The [seven original completed branches](docs/al_nuclei_lineage.md#seven-original-v100-branches-october-7)
have a completed native 0.10 ps birth/fate search: 25,867 frames from six melt
ancestors. All 28 source/detector tasks finished at 01:03 Europe/Paris on October 8.
PTM finds 17 established liquid-origin root families, plus three separately
established subcritical composites. Cumulative-dataset inclusion remains pending;
the separate 33 replacement branches are not included in this result.

**Additional labels for `al-nuclei-birth-010`:** the permanent
`al-nuclei-birth-010-first-crystal-v1` supplement stores first local PTM appearance
and three-frame persistence at 0.1/0.5/1/3/6 ps horizons. It joins by source,
frame and center atom, preserving original ancestry and roles. One final export
follows label-collection completion; missing windows and null out-of-risk targets are not
negatives. Existing established-nucleus labels and the parent fingerprint remain
unchanged. [Reader and publication guide](docs/datasets/nuclei_birth_labels.md).

**First crystalline appearance assay (2026-10-07):**
`al-first-crystal-010-20261007` prepares outcome-independent local80 Q,V panels
from all 114 velocity-bearing views in the pinned `al-nuclei-birth-010` source
contract (106 recorded ancestors). Original 0.1 ps full-cell PTM labels identify
the next first crystalline atom at 0.1/0.5/1/3/6 ps horizons, including transient
appearances. The present crystal-free risk set is a new target, not a replacement
or extra neighborhood filter for the existing established-birth benchmark.
Preparation is running; final eligible row counts are not yet known. See
[workflow](docs/structural_response.md#first-crystalline-appearance-october-7).

**Million-atom Al velocity recovery:** [native replay](docs/simulations/al_million_velocity_replay_20261007.md)
targets the existing 400 ps production first, saving Q/V every 0.10 ps. Every
section must pass historical-coordinate and native-endpoint agreement gates.
`al-million-velocity-replay-20261007` is a recovery collection, not a new independent
melt or an automatic replacement of the original nucleus labels; completion is
recorded in its live status. The preceding 300 ps melt is deferred.

**Ni nucleus-birth collection, expanded to 24 independent melts:**
[campaign](docs/simulations/ni_births_010_20261007.md) registers the original
`ni-births-010-20261007` and its 12-source `ni-births-010-20261007-extension01`:
32,000-atom Ni melts, six per 1100/1200/1300/1400 K temperature, isotropic NPT,
positions and velocities every 0.10 ps. **All 24 simulations and both searches
are complete**; the combined catalogue finished October 8 at 02:40 Europe/Paris.
It contains 110,384 frames, 11.036 ns summed saved duration and 42.40 GB of usable
trajectory files. There are 11 established liquid-origin PTM records, all at
1100 K, with detector sensitivity and composite histories retained separately.
[Combined source plan](output/ni_nucleation/campaign-20261007-extension01/technical/combined-source-plan.json).
The combined birth-table export preserves component datasets, ancestry and
original roles. This remains separate from the consolidated Al birth population.
See the [completion and storage breakdown](docs/simulations/ni_births_010_20261007.md#october-8-completed-collection-and-search).

**Additional Ni low-temperature collection, launched October 8:**
`ni-births-lowT-010-20261008` adds 36 independent melts, six each at
950/1000/1050/1100/1125/1150 K, with the same 32,000-atom NPT and native 0.10 ps
positions/velocities. The additional-GPU dispatch has six active Ni GPUs at its
initial check, with further workers queued under the per-user GPU limit;
automatic birth analysis waits for all original and additional workers.
At user request its queued CPU descriptor search was replaced by GPU
`resident`/`voronoi` extraction followed by unchanged CPU lineage tracking;
The current derived result ID is `ni-lowT-20261008-gpu-full-v1`: the user
subsequently capped holds at **400 ps at target temperature (420 ps with cooling)**.
A detached controller caps active sources; completed longer raw histories remain
intact. At the user's clarification, analysis uses **all saved frames**, including
longer histories and verified retained tails beyond 420 ps; it has no time cap.
Earlier crystallinity stops remain in effect. Zero established births are valid
censored observations and are retained explicitly. Earlier GPU search revisions
are superseded planned destinations, not additional datasets of results.
Replicates 06–11 and preparation seeds are disjoint from the
earlier 24 sources. This planned population is separate from the completed
24-source catalogue; new births are not yet available.
[Protocol, roles and launch receipts](docs/simulations/ni_births_lowT_010_20261008.md).

**Ni potential available:** the unchanged `Ni1_Mendelev_2012.eam.fs` author file
is registered as `potential-ni-mendelev2012`, with potential identity
`ni-mendelev2012-eam`. [CUDA engine and numerical validation](docs/simulations/ni_eam_gpu_20261007.md).
The short engineering reference cell is not a Ni precursor production dataset.

**New dense Al velocity reshoots:** [V100 campaign](docs/simulations/al_dense_shooting_v100_20261007.md)
prepares 40 branches from 20 existing liquid parents, with 0.10 ps positions and
velocities. Four V100 workers are now authorized: the original worker plus
three added lanes, sharing the same 40 branches. Completed children are registered
under `al-dense-shooting-v100-20261007`. They share original melt ancestry and
roles. New birth labels are not available until extraction and search finish.

**Nuclei birth dataset for new experiments:** use **`al-nuclei-birth-010`** or
**`al-nuclei-birth-030`**. The 0.30 ps name now means **uniformly saved native
0.30 ps data only**, excluding subsampled finer data and irregular shooting
prefixes. The 0.10 ps population retains its exact views of finer native sources.
The [dataset guide](docs/datasets/nuclei_birth.md)
provides one cumulative event/source index, established and dissolved-embryo
selections, and a single update command. Earlier dated searches are provenance;
consumers no longer assemble them. Unsearched holdings stay explicit in coverage.
The [crystalline precursor tails](docs/datasets/nuclei_birth_histories.md) are
attached histories of the same established births; their ongoing collection is
separate from the sealed event catalogue and does not create new birth labels.

**Reusable spatial-to-motion training:** [October 7 workflow](docs/structural_response.md) reuses 1,056,768 unrelaxed Al/Mg/Ti/Ta patches and the existing 88-view/84-ancestor verified-MEAM velocity cohort. Exact saved future coordinates are prepared; no new MD or independent ancestry. Two 12-epoch spatial starts feed matched derivative/value motion fits. Per-fit receipts determine training status.

The new motion revision reuses only the completed spatial checkpoints: no
million-row structural loader or replay during motion fitting. Birth evaluation
is pinned to `al-nuclei-birth-010`; the catalogue contract and eligibility counts
are separate from the11,200-row label-free motion training/evaluation cohort.
The prepared `al-motion-birth-010-nofilter-20261007` inputs apply no additional
neighborhood screen:161 training and79 pooled-test physical births, exact1/3/6ps
observations and same-source nonbirth controls. [Active fast motion workflow](docs/structural_response.md).


**Separate embryo-to-nucleus cohort:** [workflow](docs/continuous_motion.md#present-selected-embryo-establishment) collects present 8–63 atom embryos from the 92 pinned velocity-bearing views, with common 1/3/6 ps future-establishment labels. Collection completed: 34,060 deduplicated observations; pooled test contains 1,776 embryo families across 29 ancestors, including 29/54/67 positive families at 1/3/6 ps. Existing strict birth metrics are preserved.

**Additional October7 birth data:** [new search](docs/al_nuclei_lineage.md#october-7-additional-nucleus-birth-data) queues22 newly completed native0.10ps production trajectories with velocities:17 train,1 validation,4 test.43192 frames; birth and strict-input eligibility counts pending. Existing source ancestry and earlier releases remain unchanged.


**Continuous-motion birth transfer:** [1/3/6ps workflow](docs/continuous_motion.md#nucleus-birth-transfer-at-1-3-and-6-ps) extracts new exact lead observations from the92 pinned velocity-bearing views. All nine frozen encoders and 171 readouts completed. The strict cohort retains 16 training births and 7 pooled-test births across 4 held-out melt ancestors; readout selection used training ancestors only.


**Continuous-motion teachers:** [workflow](docs/continuous_motion.md) reuses88 verified-MEAM velocity views from84 Al64 ancestors, with outcome-blind local80 sampling. New100fs full-cell tangent banks,0.1–12ps saved future targets, nine matched fits and separate completion receipts; no new independent ancestry.

**Pooled birth test:** [derived evaluation](docs/response_dense_birth.md#pooled-test-and-neighborhood-audit) combines validation/calibration/test by user request while preserving original roles. It reuses the exact all010 cohort and audits every candidate's neighborhood rejection. Readout selection is confined to original training ancestors; no dataset arrays or historical splits are rewritten.

**All available 0.10 ps response-to-birth evaluation:** [October 6 workflow](docs/response_dense_birth.md) completed 92 catalogue views with saved velocities across three v3 releases, including older partial histories and exact 0.10 ps subsampling of finer histories. The original separate-role evaluation completed all 153 fits at 15:07 Paris; current Q/QV encoders are frozen. Million-atom and older 100k-atom records without velocities are excluded from this matched comparison.

**Native-velocity response training:** [October 6 workflow](docs/simulations/response_native500_20261006.md) retains 135 full periodic parents with actual MD velocities/source temperatures, 1000 fs damping and user-requested float64 integration. Targets reach 0.5 ps; twelve fits and native-velocity all64 clustering use a new nine-hour budget.

**Verified MEAM production birth search:** [October 6 addition](docs/al_nuclei_lineage.md#october-6-verified-meam-production-addition) checks new native LAMMPS Al production against executed potential commands, potential-file hashes, liquid-start provenance and exact 0.10 ps timestamps. Recovery/control purposes and previously searched inputs are excluded; matched 0.30 ps views use the same v3 criterion.

**Box/thermostat continuation controls:** [nine-hour workflow](docs/simulations/nucleus_dynamics_ablation_20261006.md) adds 384 paired 12 ps branches on the same 64 source states; NVT box/timestep controls and weaker BAOAB damping, with the previous 192 continuations reused.

**Continuation-rate calibration:** [paired audit and present-selected embryos](docs/simulations/nucleus_continuation_bias_20261005.md) separates selected-negative evidence from an actual rate comparison. A new 64-state cohort is frozen using present 8–63 atom embryos and past history only; NPT and BAOAB use source temperature and common observation precision. Its ancestry overlaps earlier collections.

**Additional negative continuations:** [three-hour supplement](docs/simulations/nucleus_negative_extra_20261005.md) selects 64 additional original liquid ancestors and adds fresh noise shots for 12 previously selected failed embryos. Selection uses present structure only; repeated shots remain distinct from independent starting states. This is a separate native 0.75 ps release, not all-Al completion.

**Revised Al nucleus search:** [lineage search and new dense additions](docs/al_nuclei_lineage.md) separates birth origin from later fragmentation, coalescence and fate. The frozen rerun and newly available 0.10/0.30 ps observations have separate releases and launch receipts.

**Dense Al GPU transfer:** [workflow](docs/simulations/al_dense_gpu_20261004.md) moves five unstarted 70,304-atom sources to CUDA MEAM forces with the original native isotropic NPT, 2 fs integration, 0.1 ps observations and stopping rule. Existing ancestors/roles retained; two GPU lanes and the remaining CPU queue are tracked separately.

**Differentiable CUDA MEAM:** [workflow](docs/simulations/meam_gpu_blackwell_20261004.md) reuses the16k liquid and four response parents; adds a separate seeded-interface growth control. Numerical gates, response timings and per-arm MD completion are explicit; this does not establish melting-point or NPT equivalence.

**Periodic16k MEAM comparison:** [workflow](docs/simulations/meam_md_16k_20261004.md) prepared one new liquid ancestor for matched200ps CPU/GPU450K trajectories,0.1ps observations and native1ps states. Preparation and1ps conservation gates passed; both long arms launched, with completion recorded per arm. CPU/GPU descendants are not independent ancestors.

**Initial-velocity response pilot:** [workflow](docs/simulations/response_velocity_20261004.md) retains the stopped 135-parent full periodic campaign and runs a separate three-hour subset: 12/4/4 train/selection/test ancestors, two sampled velocities and two shots per bank, with unchanged 0.1/0.5/1 ps horizons. Four encoder fits and all64 analysis follow collection. Remaining MEAM checks were skipped at the user's request. No new independent ancestry.

**New CUDA MEAM velocity responses:** the [0.5 ps recipe](configs/simulation/response_velocity_cuda500_float32_20261004.json) restores the original90/15/30 roles and four sampled velocities per geometry, collecting100/500fs targets with compiled analytic derivatives. Float32 forces, simulation states, tangents and observables are explicit; only the backend scalar energy reduction remains float64. The stopped mixed-precision bank is preserved separately. Twelve fits and all64 clustering follow collection. Longer runtime is explicitly reauthorized; no new independent ancestors.

**Differentiable MEAM benchmark:** [workflow](docs/simulations/response_meam_torch_20261004.md) reuses four original local80 train ancestors and their noise streams for same-GPU MEAM/MACE timings. Stores current768-feature responses; no new independent MD ancestry or trained encoder.


**New full Al embryo catalogue:** [three separate cadence protocols](docs/al_nuclei_search.md)
search original full cells without crystal-free-history or liquid-control eligibility.
Native0.1 trajectories also enter a paired exact0.3 search; frozen coverage records
all included, unavailable and incompatible collections.

**Relaxed data:** use the [unified relaxation index](docs/data_storage.md#unified-relaxation-index)
to find existing full cells and exact float32 neighborhoods, or plan only missing
work. New experiments share native restart states for extracting new centers
without repeating a minimization.

**Expanded unrelaxed response collection:** [workflow](docs/simulations/response_geometry_20261003.md) prepares720 train geometries across the same90 ancestors, retains original15/30 selection/test queries and adds three common-path future-feature targets. Collection is running, not yet complete.


**All Al means all Al:** the [shared Al pool](docs/al_research_pool.md) joins
appearance/fate labels and prepared inputs with selectable cohort metadata.
Unprocessed Al collections remain in its inventory, with their eligibility state
explicit. Never report one finished sub-study as completion of all Al work.

**Native 0.1 ps Al appearance assay:** [six-frame workflow](docs/birth_dense010.md)
freezes 24 completed 520 K descendants for a separate birth search and strict
crystal-free input screen. Old 0.75/0.30 ps event labels are not reused.

**Ta appearance cutouts at±12ps:** [eight separate20Å-radius candidates](docs/ta_nucleus_crops.md#separate-12-ps-appearance-windows)
reuse verified native0.1ps movies, explicitly record incomplete time windows,
and provide a ZIP package. The previous18ps exports remain separate.

**Ta event crops:** [separate20Å movies](docs/ta_nucleus_crops.md) export appearance
candidates and verified dissolutions at0.1ps spacing. Requested±18ps windows
are explicitly truncated to available24ps shooting trajectories.

**Local80 finite disturbances:** [workflow](docs/simulations/local_response_followup_20261003.md) registers384 new MACE forwards from four existing training parents; larger displacements and24/36Å convergence, with no new independent ancestry.

**Al shooting prediction extension:** [appearance and fate workflow](docs/al_shooting_prediction.md)
screens 77 new established and 30 dissolved candidates, using matched 1.5 ps
input spacing across earlier and shooting cohorts. Fate fitting excludes
unresolved endings; the first extension uses unrelaxed coordinates.

**Ta births and fates:** [full-cell search](docs/ta_births.md) covers six earlier runs and24 shooting futures, retaining six parents in one preparation-ancestry group. Exact0.5ps protocol; candidate labels only.

**Al shooting births and fates:** [full-cell search](docs/shooting_births.md) covers completed shooting branches, preserving parent/source ancestry and counting nested continuations once. Separate exact 0.30 ps protocol; candidate labels only.

**Failed-embryo appearance histories:** [workflow](docs/birth_appearance.md) extends the birth assay with eligible candidates from the nested 184/28 fate catalogue; original source roles and old rows are unchanged. Original and relaxed input variants share labels.

**Nucleus fate labels:** [CPU audit](docs/nucleus_fates.md) inventories smaller disappearing crystal episodes and adds growth/dissolution/censoring sidecars to existing Al64 birth rows. Original labels and source roles remain unchanged.

**Local80 response pilot:** [workflow](docs/simulations/local_response_20261002.md) registers new MACE teacher queries from135 original fixed-Al64 MEAM sources, preserving roles and center identities; nested moving environments are gated before training.


**Atomistic response-training collection:** [workflow](docs/response_atlas.md#atomistic-response-training)
collects fresh fixed-MACE values/responses for56 full-cell Al256 synthetic
configurations, with32/8/16 train/selection/test roles and shared FCC ancestry.

**Value-only predictive baseline:** [workflow](docs/predictive_baseline.md) derives a fixed
274-coordinate target from the sealed Al480 local assay; all historical rows and
source roles are retained. No new MD or Al64 resplit.

**Response-atlas numerical pilot:** [workflow](docs/response_atlas.md) reuses Al480 outcomes and adds a separately registered fixed-MACE Al256 perturbed-FCC response collection.


**Shooting future-law assay:** [workflow](docs/shooting_laws.md) reuses the four
compatible 48 ps Al campaigns (40 parents, 480 futures), with separate CSLD,
nested-Al and Ta diagnostic collections. No new simulations or Al64 resplit.

**Pre-appearance birth classification:** [workflow](docs/birth_prediction.md)
derives matched crystal-free histories and continuously liquid controls from
the completed original Al64 birth catalogue, preserving all source roles.
This event-enriched track is distinct from the historical all64 window benchmark.

**Spatial VICReg mechanism cache:** [workflow](docs/spatial_vicreg_bias.md) derives
128-atom parents, exact 80-atom views and physical/TDA references at fixed Al64
anchors. No resplit, future labels or crystal-based encoder-training filter.

**Multimaterial rich-descriptor training (prepared cache complete):** [workflow](docs/rich_multimaterial_encoder.md) derives 442 local targets from the raw Al/Mg/Ti/Ta pool; a fixed 1,056,768-patch subset gets 60 epochs with correlation/angular order 3 and batch8192.


**RD-MACE256 rich-descriptor training:** [workflow](docs/rich_descriptor_encoder.md) reuses all 183,596 unrelaxed training contexts and the exact existing 3,536-feature bank; no new data or source split.

**Liquid sensitivity and paired archived relaxation:** [workflow](docs/liquid_controls.md) freezes the available raw/relaxed intersection, with both original and recomputed relaxed crystal labels. Synthetic labels are separately identified.


**Rich liquid descriptors:** [CPU derivation](docs/liquid_descriptors.md) reuses the sealed
Al64 crystal-free observations for TDA, CNA, bond-order and geometric feature banks.
No new trajectories or source split.


**Liquid predictability descriptors and clearance:** [study workflow](docs/liquid_predictability.md)
derives geometry-only summaries and label-side observation clearance from the sealed
dense Al64 cache. No new trajectories or source split; CPU preparation is resumable.

**Strict liquid-only distance view:** [LCD-MACE128-VC](docs/crystal_liquid_distance.md)
reuses the dense Al64 context cache, retaining only liquid queries with no
established crystal in any observed patch for fitting. A crystal must exist
elsewhere for localization; crystal-absent cases are a separate evaluation.

**Dense interface-invisible contexts:** [256 centers × 64 frames × 150 existing Al sources](docs/crystal_interface_unseen.md)
proposed 2,457,600 extra contexts, retaining new geometry only when all input
patches exclude the interface. The sealed cache contains 1,133,196 contexts; original rows and
source/ancestor roles remain intact. Added centers are not independent sources.

**Crystal-interface targets:** [CIV-MACE128](docs/crystal_interface.md) retains fixed
Al64 rows and adds uniform held-out centers for interior evaluation; derived from
existing trajectories and PTM lineages, with no new simulation or source split.

**New Al 0.1 ps production:** [150 prepared-liquid reruns with velocities](docs/simulations/al_main_010ps_20260927/README.md)
retain the original melt ancestry and frozen roles; 2 fs integration, originally 600 ps measurement.
These are new descendants, not a replacement of the frozen 0.75 ps benchmark.
The [October 1 continuation](docs/simulations/al_main_010ps_20260927/README.md#october-1-continuation)
keeps the 20 completed descendants, recovers three native restarts and queues the
127 unstarted sources with the same ancestor roles.
**Additional 0.01-ps Al observations:** [two descendants](docs/simulations/al_main_001ps_20261001/README.md)
reuse source 886 (400 K) and 1004 (520 K), both train ancestors, with velocities.
They do not add independent melt lineages or alter the 0.1-ps source contract.

**Shorter pending Al sources:** [confirmed-halfway stopping and peer caps](docs/simulations/al_main_half_stop_20261001/README.md)
versions the 128 unstarted trajectories (126 main and two additions), preserving
21 completed and three already running. Keep velocities, actual endpoints,
termination/censoring and the frozen ancestor roles; the original Al64 release
is unchanged. Registered collection: `al-main-half-stop-20261001`.

**Material-specific distance adaptation:** the [Al/Ta fine-tuning protocol](docs/distance_encoder_material_finetune.md)
retains fixed Al64 roles and uses newly simulated Ta velocity branches from a known
parent for a separately labeled conditional-generalization assay.

**Six-frame MD distance training:** [nominal 0.75-ps and exact 0.10-ps experiments](docs/distance_encoder_history.md)
record actual spacing and separate fixed-Al versus external-ancestry evaluation contracts.

**Regional emergence:** [training-source harvest audit](docs/nucleus_harvest.md)
reuses existing Al trajectories for precursor eligibility, event windows and
control diagnostics. This is an analysis collection, not a released benchmark.

**New Al comparisons:** [fixed 64-center benchmark and large structural-pretraining
sets](docs/datasets/fixed_al64.md), with frozen 90/15/15/30 source roles and an exact
historical 16-center comparison track.

**Completed structural expansion:** [256 native centers and proportional
multi-material sampling](docs/datasets/structural_multimaterial_256.md), preserving
the fixed prediction benchmark.

**[Open the searchable dataset registry](docs/datasets/index.html)** ·
[Browse the Markdown cards](docs/datasets/README.md) ·
[Download CSV](docs/datasets/datasets.csv) · [Registry JSON](docs/datasets/registry.json)

**[Potential files, hashes and provenance](docs/datasets/potentials.md)**

The registry covers **Al, Mg, Ti, Ta, Zr and Al–Ni structures**: raw dynamics,
static structures, synthetic geometry, training caches, physical targets and
potential files. Filter by material, potential or use classification. Each card
links to producer records, array schemas, hashes and known provenance.

Read the classification before using a dataset. A complete binary is not
necessarily an independent trajectory, an accepted protocol or a training input.
Duplicate exports, incomplete preparations, removed Zr dynamics, shared ancestry,
metadata gaps and reported H200 copies remain visible.

Refresh from the repository root in `pointnet`:

```bash
python scripts/project.py datasets --refresh
```

This reads metadata, potential files, array headers and filesystem sizes. It does
not alter datasets, load full trajectories, submit jobs or fit models. Pages show
an observation at the displayed timestamp, not live job status. Unregistered
directories are listed for review.

**Register new data in [configs/datasets.json](configs/datasets.json)**, including
storage role, material, generating potential, provenance and ancestry when known.
Keep unknown values explicit. See [the registry guide](docs/datasets/GUIDE.md).

Response-atlas follow-up (2026-10-01): stronger toy controls and fixed-parent shot precision; see `experiments/response_atlas_20261001/FOLLOWUP.md` and `docs/response_atlas.md`.

**Planned fast Al256 response collection:** `response-atlas-al256-training-fast-20261002` registers the new float32/cuEquivariance default, with the same 56 generating geometries as October1 and separate stochastic streams. This is a registered recipe, not a completed collection or new independent ancestry; see [the workflow](docs/response_atlas.md#default-for-new-al256-response-runs).

Birth-centred history supplement: `al-nuclei-birth-010-histories-v1` is collecting 295 original established-birth records, with precursor tails and appearance-observation links. It does not add nucleus events. See [definitions and access](docs/datasets/nuclei_birth_histories.md).

**Ni phase-model observed histories:** `ni-phase-history-native4-20261013` adds
exact native Q,V at -0.3/-0.2/-0.1/0 ps to all630 existing phase teachers and
19652 unique nucleus inputs. Same current80 atom IDs and28 train/62 test source
ancestors; no additional MD or independent trajectories. See
[history workflow](docs/ni_phase_sim.md#history-attention-workflow).

**Ni dense measured targets:** `ni-phase-measurements-20261014` derives 244
measurements and 91 smooth measurement responses at all 21 saved times of the
same 630 physical teachers. Values/JVPs use float64; no new MD or independent
ancestors. Complete before training, with checksummed cache and training-only
normalizers. See [workflow](docs/ni_phase_sim.md#measurement-supervision-workflow).
