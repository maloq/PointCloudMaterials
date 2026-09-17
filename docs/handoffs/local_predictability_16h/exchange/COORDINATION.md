# Execution handshake (2026-09-17)

H100 execution is authorized and active preparation is being implemented. Shared
start: 2026-09-17T12:53:56Z. Actual local allocation 995957/node53 ends
2026-09-18T03:59:30Z. Training cutoff: 2026-09-18T02:59:30Z; final hour for export.
H200 should respect this same shared deadline, or any earlier provider deadline.

Exchange directory in the H100 repository:
`docs/handoffs/local_predictability_16h/exchange/`.
Data output: `output/local_predictability/h100-20260917/technical/`.
Dense cache: `${storage:cache}/local-predictability-20260917-broad`.

H100 owns `src/research/local_predictability/data.py`, descriptor baselines,
assay/metrics, and local execution. H200 owns native model/loader/trainer and their
tests. Do not edit each other's files. A cohort.json containing exact source,
center and anchor identities will be published before target computation. Source
IDs remain the inherited integer IDs. Native anchors are frame indices
64,104,...,664 (times 48,78,...,498 ps). 16 core centers per source, from a
64-center outcome-independent pool, exact legacy selections reused when hashes
match. H200 may begin raw-index work as soon as cohort.json is available.

Proposed release v1: each source has a NumPy NPZ with `packet` float32[16,801,128],
`labels` uint8[16,801] PTM structure codes, `atom_ids` int64[16],
`times_ps` float64[801], `order` float32[16,801,8], `shell` float32[16,801,12].
All arrays describe fixed tracked centers over the complete timeline. Center/order
and shell features are diagnostics, not changed physical targets. `release.json`
will contain source metadata, relative shard filenames and SHA256, normalizer
mean/scale, exact native/dense anchors and horizon frame offsets. Normalizer uses
train-only current and six future packets on the 16 native anchors, scale floor
1e-4, matching the existing fit_scaler definition. Physical training uses all
native rows; event masks are external to the physical objective. H200 must not
use labels to select a physical checkpoint. H100 will publish explicit per-row
onset/censor/eligibility labels in a separate native_rows.npz for supervised fits.
