# Ni low-temperature nucleus-birth campaign, October 8

The user requested additional Ni simulations, including temperatures below
1100 K. This campaign adds 36 independent melts: six each at 950, 1000, 1050,
1100, 1125 and 1150 K. The temperature grid and seeds were fixed before outcomes.
The earlier 24-source campaign found established liquid-origin records only at
1100 K; this grid extends below that temperature and resolves the gap towards
1200 K. The objective is to compare liquid ordering, embryo emergence and growth
across temperatures, without changing the event thresholds to increase counts.

**Current duration policy:** at the user's subsequent request, active and future
sources are capped at **400 ps at target temperature, 420 ps including cooling**.
Earlier crystallinity stops still apply. At the user's clarification, **analysis
uses every saved frame with no time cap**, including histories longer than 420 ps.
Zero established births remain valid censored results at each actual endpoint.
See [full saved-history analysis](#full-saved-history-analysis-user-clarification) below.

## Protocol and ancestry

The [recipe](../../configs/simulation/ni_births_lowT_010_20261008.json) preserves
the original physical protocol: 32,000 Ni atoms, the pinned
`Ni1_Mendelev_2012.eam.fs` potential, zero-pressure isotropic NPT and a 1 fs
integration timestep. Each source has its own 2300 K, 40 ps melt, which must pass
the original liquid-quality gates. Cooling takes 20 ps, followed by a hold of
up to 500 ps. The original stopping rule retains 10 ps after reaching 50%
whole-cell crystallinity, subject to the duration cap.

Positions and velocities are saved every native 0.10 ps. Integration remains
float64, GPU force arithmetic float32, archived Q/V float16 and boxes float32;
timesteps and atom IDs remain exact. Verified conversion receipts retain
quantization errors and checksums. Native restart files preserve integration
precision. New simulation staging is on SCRATCH; completed trajectories and
native checkpoints are published to STORE.

Source IDs use replicates 06–11, with roles train/selection/test repeated twice
at every temperature. All 36 source IDs and preparation seeds were checked
against both earlier campaigns and are disjoint. These are fresh independent
melts, with no inherited parent coordinates or velocities. This is a separate
temperature-screening collection, not an extension with an identical temperature
grid. The completed historical 24-source catalogue remains its own population.

Cooling and isothermal births retain separate labels. The four established
structure detectors and lineage definitions are unchanged: primary PTM
establishment requires 64 connected atoms sustained for 1.5 ps; composite,
interface-associated and uncertain-origin records remain separate. Birth counts
are not independent-melt counts, and establishment is not a committor estimate.
The full Q/V supports subsequent prestructured-liquid analysis; the automatic
search itself does not establish whether liquid precursors predict births.

## Execution

Submitted October 8, 12:10 Europe/Paris, with four detached GPU workers and a
dependent 12-CPU birth-search job. Each worker receives nine sources and requests
one GPU from the combined A40/A100/L40S/H100/RTX6000PRO alternatives. All four
initial allocations are A40. The GPU wall-time limit is 18 hours; this is a
scheduling ceiling, not a completion estimate. Historical A40 throughput suggests
roughly 12–14 hours for all nine full-length sources per worker, potentially much
less when crystallinity stopping fires. This initial estimate is superseded by
the additional-GPU dispatch below. Analysis follows the simulations.

| Stage | Slurm job | Initial node |
| --- | --- | --- |
| GPU worker 00 | 1028472 | node26 |
| GPU worker 01 | 1028473 | node28 |
| GPU worker 02 | 1028474 | node38 |
| GPU worker 03 | 1028475 | node38 |
| Birth analysis | 1028476 | dependency on all four workers |

The launch receipt establishes current-chat ownership. The code, configuration,
machine mapping and metric definitions were frozen before submission. Numerical
validation is reused from the passed audits. The only mismatch in the validated
MD file was relocation of two imports inside `audit()`: an
[AST and exact-hash review](../metrics/ni_validation_import_review_20261008.json)
confirms unchanged production functions. Reuse records bind both implementations
and the review checksum; no new numerical audit was executed.

Dataset ID: `ni-births-lowT-010-20261008`.
Plan identity: `ec82cb7932b179f693809805cc45ee69501ff3b24069f18d0ef86ceb3597ef03`.

- [Source plan](../../output/ni_nucleation/campaign-lowT-20261008/technical/plan.json)
- [Launch and job receipt](../../output/ni_nucleation/campaign-lowT-20261008/technical/launch.json)
- [Validation reuse](../../output/ni_nucleation/campaign-lowT-20261008/technical/validation-reuse.json)
- Archive: `${storage:archive}/simulations/ni-births-lowT-010-20261008`
- Analysis: `${storage:training_storage}/nuclei_search/ni-lowT-20261008-v1`
- Disposable analysis cache: `${storage:cache}/nuclei-search/ni-lowT-20261008-v1`

Maintained commands, using conda `pointnet-torch214`:

```bash
python -m src.simulation.campaigns.ni_births.workflow prepare --config configs/simulation/ni_births_lowT_010_20261008.json
python -m src.simulation.campaigns.ni_births.workflow submit --config configs/simulation/ni_births_lowT_010_20261008.json
python -m src.simulation.campaigns.ni_births.workflow status --config configs/simulation/ni_births_lowT_010_20261008.json
```

Submission has already happened; do not resubmit these sources. The maintained
command rejects a duplicate launch. Current completion and failure states belong
to the launch's `status-*.json` and per-source archive receipts.

## Additional GPUs, October 8 at 12:17 Europe/Paris

The user requested more GPUs. The 32 unstarted sources were redistributed into
eight new GPU reservations, four sources per worker. The four original native
simulations continue uninterrupted. Only their queue supervisors are paused;
the drain controller verifies each current simulation's completed trajectory
before releasing its original allocation. No source, seed, physical setting,
saved cadence, producer or metric definition changed.

New GPU jobs are 1028483–1028490, with CPU drain controller 1028491. The original
analysis job 1028476 now depends on the four original allocations, all eight
new reservations and the drain controller. All new jobs request one GPU from
the same combined compatible partition list, with an eight-hour wall-time limit.

At the initial dispatch check, jobs 1028483 and 1028484 had started on A40 nodes
27 and 30. Together with the original four this gives **six active Ni GPUs**.
The other six reservations were pending with `QOSMaxGRESPerUser`; GPUs allocated
to other work also count against that limit. As original simulations finish,
their released slots can start waiting Ni workers. Eight new reservations do
not imply eight simultaneously available additional GPUs.

The revised conservative simulation estimate at 12:20 is **6–9 hours remaining**,
approximately 18:30–21:30 Europe/Paris on October 8, at the observed capacity and
native throughput. This includes expected queueing for the remaining workers;
it is not a guaranteed deadline. Crystallization stopping or additional available
quota can shorten it. Birth analysis follows and is not included in this estimate.

- [Redistribution, assignments and ownership receipt](../../output/ni_nucleation/campaign-lowT-20261008/technical/scale-01/launch.json)
- [Complete dependency dispatch](../../output/ni_nucleation/campaign-lowT-20261008/technical/dispatch.json)
- Per-new-worker progress: `technical/scale-01/worker-*/source-progress.json`
- Original-source drain: `technical/scale-01/drain/lane-*/drain-progress.json`

The maintained command was:

```bash
python -m src.simulation.campaigns.ni_births.scale submit --config configs/simulation/ni_births_lowT_010_20261008.json --workers 8 --output output/ni_nucleation/campaign-lowT-20261008/technical/scale-01
```

It has already executed; do not resubmit it. The unchanged original four-lane
configuration remains the physical source contract. Actual scheduling and
per-worker assignments are recorded in the separate dispatch revision.

## GPU nucleus search, October 8

At the user's explicit request, the pending CPU extraction/search job 1028476
was replaced at 13:01 Europe/Paris by GPU extraction 1028535 and CPU lineage
tracking 1028536. The [GPU search recipe](../../configs/analysis/ni_lowT_gpu_search_20261008.json)
pins the same 36-source campaign identity and the reviewed descriptor code.
MD jobs and the original simulation configuration are unchanged.

Extraction requests one GPU from the A100/H100 alternatives, four host CPUs and
32 GB memory. It starts after all remaining MD workers finish. Original producer
job IDs remain in the receipt; already-finished jobs that have expired from Slurm
are omitted from new scheduler dependencies. Tracking requests twelve CPUs and
64 GB and depends on successful GPU extraction. The six-hour extraction and
24-hour tracking wall-time limits are scheduling ceilings, not timing forecasts.

The wrapper calls the existing `gpu-extract` implementation in `resident` mode
with `voronoi` PTM ordering, 16-frame chunks, two-frame prefetch and one compression
worker. Every native 0.10 ps observation is retained. No CPU structure extractor
is called. CPU work after extraction is chronological family tracking and table
publication, using the same four detectors and event thresholds.

The [GPU audit](../postprocessing_gpu.md) found identical PTM labels and events
in the audited windows, with small bond-order differences. This is recorded as a
distinct numerical producer, not asserted to be bitwise identical for every
new trajectory. Both extraction and tracking plans bind the actual GPU code,
audit, input identities and chunk checksums. Completed raw trajectories pass
checksum and actual-timeline checks before admission; failed/missing sources
remain explicit coverage records rather than negative birth examples.

A manual compatibility check consumed an already-audited 64-frame Ni GPU cache
through the unchanged tracker for all four detectors successfully. It did not
rerun GPU extraction and its diagnostic records are excluded from campaign births.
The final preparation verified five completed new sources; remaining inputs will
be admitted after MD. Frozen imports, the reviewed 25-file GPU implementation,
source plans and metric contracts were checked before execution.

- [Launch and replacement jobs](../../output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-01/launch.json)
- [Replacement receipt](../../output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-01/replacement.json)
- [GPU-cache/lineage compatibility check](../../output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-01/gpu-lineage-compatibility.json)
- Dataset/result ID: `ni-lowT-20261008-gpu-v1`
- Results: `${storage:training_storage}/nuclei_search/ni-lowT-20261008-gpu-v1`
- Cache: `${storage:cache}/nuclei-search/ni-lowT-20261008-gpu-v1`

The original CPU-only analysis location remains a superseded planned destination.
Historical completed Ni catalogues and their frozen CPU provenance are retained.

## 400-ps hold cap and analysis revision

The simulation stop policy below remains active. Its initially capped analysis
was superseded by the following full saved-history revision before execution.

The user explicitly chose 400 ps at target temperature, not 400 ps including
cooling. The [policy](../../configs/simulation/ni_births_lowT_hard_stop_400ps_20261008.json)
therefore sets a 420-ps saved endpoint. The original frozen 500-ps-hold recipe
and already-completed longer raw histories remain as historical provenance.
The earlier stop at 50% crystalline plus 10 ps remains subject to the new cap.
An unconfirmed or absent birth at the endpoint is not a failed simulation and
does not cause an automatic extension.

Detached CPU controller **1028622** checks the launch-owned source workers.
The already-loaded MD code has no live duration hook, so the controller pauses
the matching source child after the cap checkpoint is saved, verifies the
complete native 0.10-ps prefix and Q/V checksums, and publishes the exact prefix
and native checkpoint before ending that child. The GPU queue supervisor then
continues to its next assigned source. Native logged overshoot is recorded
separately, including a run already past the cap when the controller started;
the published capped data never include those later frames. Worker nonzero-exit
receipts are preserved with an explicit intentional-stop interpretation.

The first controlled source, `ni-T1050-melt08`, published **4,201 frames from
0 to 420 ps** with the verified `isothermal-000420000.restart`. It had already
reached 520 ps in native execution while final publication was pending; that
100-ps overshoot remains in its stopping certificate and logs. Its supervisor
was observed starting `ni-T1125-melt09` afterward. The archive's native structural
and thermodynamic logs preserve their actual longer times; analysis uses the
explicit capped Q/V observations.

At 14:08 Europe/Paris the pending GPU search jobs 1028535/1028536 were replaced
by **GPU extraction 1028625** and **CPU lineage 1028626**. Extraction waits for
all eight remaining MD workers and successful completion of controller 1028622.
Tracking waits for successful GPU extraction. Resource settings and audited
GPU descriptor implementation are unchanged. The earlier frozen search plans
remain as a superseded, unexecuted revision.

The [new search recipe](../../configs/analysis/ni_lowT_gpu_search_400ps_20261008.json)
uses a separate result/cache ID, `ni-lowT-20261008-gpu-400ps-v1`. Every completed
source is analyzed through min(raw endpoint, 420 ps), with exact native indices,
unchanged source ancestry and roles, and no outcome-based source filtering.
Nineteen completed sources passed checksum, native-timeline, frame-view and
frozen-plan checks at submission. Three historical 520-ps trajectories are
viewed through 420 ps while their complete raw arrays remain intact. Inputs
that finish later receive the same checks before extraction.

`analyses/births-v1/tables/observation_outcomes.csv` will contain one row per
completed source and detector, including explicit zero established-birth counts.
It separates cooling/hold outcomes, records the observed duration and original
raw termination, and distinguishes no observed establishment from a claim that
crystals can never form. Failed/missing sources remain separate coverage records.
Births requiring persistence beyond the observation endpoint remain unconfirmed.
No new birth counts are available until the queued search finishes.

At 14:07 Europe/Paris: **19/36 completed, eight running on eight GPUs, nine not
yet started**. This is a timestamped execution snapshot, not final coverage.

- [Controller ownership and launch](../../output/ni_nucleation/campaign-lowT-20261008/technical/hard-stop-400ps/launch.json)
- [Controller live status](../../output/ni_nucleation/campaign-lowT-20261008/technical/hard-stop-400ps/status.json)
- [New search launch](../../output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-400ps-01/launch.json)
- [Native observation-window verification](../../output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-400ps-01/window-verification.json)
- [Scheduler and frozen-execution verification](../../output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-400ps-01/queue-verification.json)

Maintained commands (already submitted; do not repeat):

```bash
python -m src.simulation.campaigns.ni_births.hard_stop prepare --config configs/simulation/ni_births_lowT_hard_stop_400ps_20261008.json
python -m src.simulation.campaigns.ni_births.hard_stop submit --config configs/simulation/ni_births_lowT_hard_stop_400ps_20261008.json
python -m src.postprocessing ni-gpu-search prepare --config configs/analysis/ni_lowT_gpu_search_400ps_20261008.json
python -m src.postprocessing ni-gpu-search submit --config configs/analysis/ni_lowT_gpu_search_400ps_20261008.json
```

## Full saved-history analysis, user clarification

The user clarified that trajectories already longer than 420 ps must be analyzed
in full. There is now **no analysis time cap**. The simulation stop controller
1028622 continues enforcing the requested 400-ps hold limit; it was not restarted
or changed. Its historical policy text is qualified by the separate
[analysis-policy amendment](../../output/ni_nucleation/campaign-lowT-20261008/technical/hard-stop-400ps/analysis-policy-amendment.json).

At 14:19 Europe/Paris, the still-pending capped search jobs 1028625/1028626 were
replaced by **GPU extraction 1028644** and **CPU lineage 1028645**. Extraction
retains dependencies on remaining MD producers and successful controller
completion. The [current recipe](../../configs/analysis/ni_lowT_gpu_search_full_20261008.json)
uses `ni-lowT-20261008-gpu-full-v1` for separate results and caches. Descriptor
implementation, scientific event thresholds, cadence and source roles are unchanged.

Preparation verified all native frames of 22 completed sources. Four have saved
histories through **520 ps (5,201 frames each)**: `ni-T1000-melt07`,
`ni-T1050-melt08`, `ni-T1125-melt06` and `ni-T1150-melt06`. For
`ni-T1050-melt08`, the 1,000 already-converted frames beyond its capped publication
were still present on SCRATCH. Their checksums, exact native timeline and atom
IDs were verified, and the 20 chunks were copied without requantization into
its STORE `hard-stop/saved-tail/` archive. The analysis appends those parts to
the original 4,201-frame publication, preserving its stop certificate and source
identity. Boundary geometry at 420/420.1 ps and the final 520-ps frame was checked.
These are retained observations, not another simulation or extra independent melt.

Later completed sources receive the same full-history handling, including any
verified saved post-cap chunks. The source plans bind all part manifests and
the tail archival receipt. Observed duration and censoring now use the actual
saved endpoint; establishment after 420 ps is eligible whenever those frames
exist. Zero-birth rows remain explicit, with `analysis_capped=false`.

- [Current launch](../../output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-full-01/launch.json)
- [Full-history and boundary verification](../../output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-full-01/full-history-verification.json)
- [Queue verification](../../output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-full-01/queue-verification.json)

Maintained commands (already submitted):

```bash
python -m src.postprocessing ni-gpu-search prepare --config configs/analysis/ni_lowT_gpu_search_full_20261008.json
python -m src.postprocessing ni-gpu-search submit --config configs/analysis/ni_lowT_gpu_search_full_20261008.json
```

## RTX6000 handoff, October 8

The user requested moving Ni off the RTX PRO 6000 on node60. That allocation,
1028490, also hosts the active Codex session and separate TensorNet training.
Only the Ni source process was ended. Its queue supervisor remains paused,
holding no CUDA context, so the session allocation stays alive. Do not resume
that obsolete supervisor while the replacement jobs own its remaining source IDs.

The active `ni-T1100-melt10` resumed on **A100 node07, job 1028835**, from the
native float64 **210-ps** checkpoint. Its 2,101 saved prefix frames, native
restart and original monitoring logs were verified and archived on STORE before
the old MD child was terminated. Initialization on the A100 confirmed the
matching `fix integrate` NPT restart state and unchanged native positions,
velocities, box and timestep. The continuation was observed beyond 225 ps.
Unsaved progress after the checkpoint is replayed; original pre-handoff logs
are retained separately. The final trajectory includes the checkpoint frame
once, with exact native 0.10-ps Q/V throughout.

The final unstarted source, `ni-T1150-melt11`, started independently on
**L40S node50, job 1028836**, using its original seed and role. Both relocated
workers enforce the **400-ps hold / 420-ps total** duration directly, retaining
the earlier crystallinity stop. Existing numerical validation is reused; force
and integration precision remain unchanged. Restart verification checks state
restoration and does not rerun the potential numerical audit.

The earlier CPU cap controller 1028622 was retired after the native duration
limit moved into the two workers. Completion gate **1028837** verifies their
outputs and all 36 source completion records. The full-history GPU search uses
the [handoff recipe](../../configs/analysis/ni_lowT_gpu_search_handoff_20261008.json)
and still analyzes every saved frame. Prepared scientific plans and result/cache
locations are reused. Dependencies follow the two replacement MD jobs and their
gate; they no longer wait for the node60 allocation to end.

- [Ownership, replacement jobs and frozen entry](../../output/ni_nucleation/campaign-lowT-20261008/technical/rtx-handoff-01/launch.json)
- [Native restart-state verification](../../output/ni_nucleation/campaign-lowT-20261008/technical/rtx-handoff-01/restart-verified.json)
- [Local Ni GPU-context release](../../output/ni_nucleation/campaign-lowT-20261008/technical/rtx-handoff-01/rtx-released.json)
- [Preserved prefix and checkpoints](../../output/ni_nucleation/campaign-lowT-20261008/technical/rtx-handoff-01/prefix.json)
- [Current analysis launch](../../output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-full-handoff-01/launch.json)

The frozen handoff entry uses the original simulation code bundle with the
[explicit execution recipe](../../configs/simulation/ni_lowT_rtx_handoff_20261008.json).
It has already launched; do not start duplicate resume/fresh workers.

The resumed A100 source subsequently finished at **234 ps**, retaining the
normal 50%-crystalline plus 10-ps stop. Final output has 2,341 exact native
observations; the saved 2,101-frame prefix is checked separately for unchanged
Q/V and boxes. The final 1150 K source continues on L40S. Current full-history
search jobs are **1028840** (GPU extraction) and **1028841** (CPU lineage).
