# Ni birth campaign: numerical gates and event definitions

This family binds the new Ni producer, its native-NPT adapter, verified Q/V
conversion and the existing v3 extraction/tracking calculations. It is not an
extension of the frozen Al source population. No predictor is fit.

Force RMS, energy per atom and scalar pressure errors compare identical native
LAMMPS `eam/fs` and GPU-EAM configurations. Final position RMS uses minimum-image
displacement after 100 native NPT steps; volume error is abs(V_native/V_GPU-1).
Wall times include host/native integration and copies; graph NVE timings from
the earlier engine assay are not used as NPT throughput.

Native thermodynamics logs use LAMMPS metal units: temperature K, pressure bar,
volume Angstrom³, energies eV. PTM monitoring fractions are whole-cell counts
of FCC/HCP/BCC divided by atom count on native observations, with 3.2-Angstrom
connected components. MSD is the native center-of-mass-corrected full-cell mean
squared displacement since 20 ps of melting. Density changes remain part of NPT.

Stored observations use verified float16 Q/V, float32 bounds, exact steps/IDs.
Quantization RMS/max errors compare every encoded component against the native
float64 value; coordinate errors use minimum images. Velocities are Angstrom/ps.
Structure extraction for final births uses the archived observations and records
that precision separately from native-coordinate stopping diagnostics.

Ni extraction uses PTM RMSD 0.10 and 3.2-Angstrom neighbors. The four outputs are
PTM FCC/HCP/BCC and q6 coherence with (bond correlation, solid degree) thresholds
(.65,6), (.70,7), (.75,8). Connected components with at least four atoms are
tracked. The lineage equations are the maintained v3 implementation described
in [Al lineage definitions](al_nuclei_lineage.md), applied to the explicitly
new Ni population: four shared atoms and 50% continuation, 8-atom appearance,
64-atom connected establishment for 1.5 ps, 128-atom growth, 1.5-ps reset and
prehistory. Interface proximity is 7.0 Angstrom in Ni. Composite/split/absorbed
families retain their distinct origin/fate and are not counted as new root births.
The same 32/128-atom and 1.5/3-ps persistence sensitivity fields are retained.
These thresholds define operational events and are not a critical-size estimate.

Event onset phase is cooling if first_time_ps < 20, otherwise isothermal;
onset_time_from_hold_ps = first_time_ps - 20. No cooling observation is relabelled
as a constant-temperature equilibrium state. Temperature is an audit field only.

Counts are grouped by physical source, temperature, detector, onset phase,
origin and whether establishment_confirmation_ps is present. Detector totals are
correlated views. Sources correspond to independently seeded full-cell melts;
multiple events within a source do not increase independent-melt counts.
Coverage explicitly includes complete, failed, partial and not-started sources.
MD completion and structural-search completion are separate statuses. The
collection's maximum 500-ps hold and causal crystallinity stopping yield censored
observation windows, not an unbiased nucleation-rate estimator.

The October 7 cleanup correction closes intermediate source memmaps before
deleting their verified files on NFS, and distinguishes post-completion cleanup
errors from MD failures. It changes no integration, observation or event
calculation. The running collection and its queued search retain their original
frozen producer and metric hashes; post-run reconciliation verifies completed
payloads and keeps the original nonzero-exit receipts separately.

The 12-source extension uses the same scientific definitions with new independent
preparation seeds and source IDs (replicates 3–5). Worker lanes partition sources
without duplication. The combined 24-source export concatenates the verified
component coverage/events/counts tables and adds their dataset IDs. It records
each input table's frozen contract and checksum, checks source identities are
disjoint and complete in coverage, and checks extraction/tracking configurations
agree apart from storage paths. No numerical event calculation, ancestry split
or original table is changed. Counts from different detectors remain correlated
views, not additional independent samples. Missing/failed sources remain in
coverage and prevent a combined scientific-completion claim.

Numerical validation receipts are now reused for matching potential, numerical
settings and MD/force/conversion implementation hashes. The reuse record links
the original metrics and their original GPU; it does not produce new metric rows
or claim a fresh numerical execution. Allocation attachment additionally checks
the GPU model against the reviewed receipt. Trajectory integrity and liquid
preparation gates still apply to each newly generated source.

The October 8 low-temperature campaign adds six independent melts at each of
950/1000/1050/1100/1125/1150 K (replicates 6–11), with unchanged integration,
observations, stopping and event definitions. Its temperature schedule is a
separate population from the original 24-source collection. The numerical audit
is reused after a hash-pinned review of two import relocations inside `audit()`;
all production functions are AST-identical to the validated implementation.
Reuse records retain both validated and executing hashes and the review checksum.
Unreviewed implementation changes still fail validation reuse.

The explicitly requested GPU search revision uses `gpu-extract` with the audited
`resident` execution and `voronoi` PTM ordering. All four detector definitions,
native source frames, ancestry roles and chronological lineage equations remain
unchanged. GPU PTM uses float64 geometry; bond-order arithmetic is float32. This
is a distinct numerical producer: the published audit found exact PTM agreement
on its sampled frames/windows but small bond-order differences, and does not
establish exact agreement on every future source. GPU caches and birth exports
use separate locations and bind their actual code, audit, inputs and chunk
checksums. Extraction identities include the GPU backend. CPU lineage consumes
verified GPU labels and never reruns CPU structure extraction. Raw MD precision
and native-crystallinity stopping diagnostics retain their original definitions.

An execution-only optimized resume pins the existing `technical/plans.json`
checksum and verifies both plan identities, the campaign recipe and extraction/
lineage settings. Completed chunks and tracking tasks retain their original
identities and checksums. New descriptor chunks use the selective detector layout;
both layouts are read without relabelling. Per-job/per-source execution receipts
record the executing implementation and which chunks/tasks were reused. Final
exports bind the current implementation; original plans and execution receipts
remain preserved. A resume cannot add sources to the frozen population.


Chunk scheduling changes execution granularity only. Bounded round-robin tasks
apply the same extractor to the same 16-frame chunks; each detector's tracking
starts after all chunks of that source pass verification. The operational worker
count is recorded separately from the immutable scientific campaign recipe.
For a streaming campaign, only completed, checksum-verified native trajectories
enter immutable per-source plans. Active/partial simulations are never read.
The final population manifest references those per-source execution plans and
is sealed after the declared producers finish (or all planned sources finish).
Failed and missing sources remain explicit coverage records. No partial export
is labelled a complete campaign or used as a negative birth label.


The user-requested October 8 duration revision caps the current 36-source Ni
collection at 400 ps at target temperature, 420 ps including cooling. Earlier
crystallinity stopping still applies. Original completed raw histories and the
original 500-ps recipe are preserved. A receipt-owned controller publishes the
exact saved 0–420 ps prefix and native checkpoint before stopping an active
MD child. Native execution overshoot is recorded separately; it is not included
in the capped observations. Positions, velocities and force arithmetic retain
the original precision.

The superseded capped GPU analysis revision admitted every exact native 0.10-ps observation up
to min(raw endpoint, 420 ps), including a capped view of older completed sources
in this same population. Source identities, roles and original array checksums
are retained; the frame list and observation-window metadata bind new plan
identities. Sources are not selected using crystallinity or detected births.
The previous historical 24-source collection is not changed.

`observation_outcomes.csv` has one row per completed source and detector, even
when event or established-birth counts are zero. `established_liquid_births`
counts events with origin `emergence_from_liquid` and a non-null establishment
confirmation, matching the existing event rule. The isothermal count additionally
requires onset at or after 20 ps. Observed hold duration is max(0, endpoint−20 ps);
`reached_hold_cap` means the endpoint is exactly 420 ps. Raw endpoint/termination
remain separately recorded. Zero means no established liquid-origin birth was
observed within that window, not that embryos or crystals were absent forever.
Incomplete end-of-window persistence stays unconfirmed; no later frames are
used to establish it. Failed/missing sources remain coverage records, not zero
birth outcomes. No-birth waiting times are right-censored at the saved endpoint;
crystallinity-triggered earlier termination is informative censoring, so these
counts alone do not constitute an unbiased nucleation-rate estimate.

The subsequent user clarification removes the analysis cap. The
`full_saved_trajectory` revision includes every verified saved native frame,
including complete raw histories beyond 420 ps. For a controller-stopped source,
any retained converted chunks beyond its published endpoint are checksum-verified,
copied to STORE without re-quantization and appended by exact native step and
atom identity. The original capped publication and stopping certificate remain
unchanged; saved tails are additional frames of the same source, not new samples.
Source plans bind all part manifests, raw-array checksums and the saved-tail
receipt. This retention does not restart or extend MD. The 400-ps simulation hold
limit remains distinct from the analysis extent.

In this full-history revision `analysis_capped` is false and `reached_hold_cap`
is undefined because there is no analysis cap. Observed endpoints and hold
durations use the full saved history. Establishment and persistence may use
frames after 420 ps when they exist; censoring is at the actual saved endpoint.
Zero-count source/detector outcomes, source roles and the original birth rules
are otherwise unchanged. Differing observation durations remain explicit.


The RTX6000 handoff preserves the current campaign source identities and saved
frames. Its active Ni hold resumes from the native float64 LAMMPS restart, with
matching `fix integrate` NPT state and the unchanged GPU EAM force implementation.
Restart initialization verifies unchanged native Q/V, cell and step. Original
checkpoint, converted prefix and pre-handoff logs remain archived; the retained
restart boundary appears once in the final trajectory. The last unstarted source
uses its original independent seed and protocol. Both relocated workers enforce
the 400-ps hold limit directly. GPU type/job and handoff implementation are audit
metadata. The move introduces no new independent source or new numerical audit.
The full-history search preserves its source/cache identities; an explicit
producer handoff receipt replaces allocation dependencies without waiting on the
old allocation, which hosts the active user session and no longer runs MD.

October 12 validation reuse adds the reviewed repository trajectory-reader
simplification. The existing two-frame, 32,000-atom publication assay loads
identical arrays and checksums with the validated and current readers; all
top-level trajectory writers and composition functions are AST-identical.
`ni_validation_reader_review_20261012.json` binds the exact old/new hashes and
the review. Physical integration, observation quantization, detector definitions
and lineage calculations are unchanged. New exports record this provenance;
historical exported contracts retain their original files.

New GPU-search bundles copy every file named by the Ni metric contract, in
addition to the selected numerical audit. This includes the original GPU audit
outside `docs/metrics/`, which must remain available when its inherited evidence
is reused. This packaging correction changes no descriptor, lineage or metric
calculation. An October 12 analysis recovery uses an unchanged copy of its
original source bundle plus the missing file at the originally recorded hash;
the failed bundle and log remain preserved.
