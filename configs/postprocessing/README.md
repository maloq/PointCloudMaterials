# Post-simulation execution contracts

The implementation and command interface are documented in
[postprocessing](../../docs/postprocessing.md). Scientific recipes remain in
their existing analysis/data/simulation collections.

`producer_equivalence.json` contains reviewed PTM classifier hashes that may reuse
the same verified cached labels. Update an equivalence only after a real-data
output comparison. It is frozen with new history workers. This declaration does
not authorize changes to scientific thresholds, input sources or labels.

[`ni_lowT_gpu_search_20261008.json`](../analysis/ni_lowT_gpu_search_20261008.json)
pins the low-temperature Ni campaign and GPU audit, requests one A100/H100 for
`gpu-extract` and twelve CPUs for subsequent lineage tracking. It supersedes
only that campaign's pending CPU descriptor search, with separate GPU provenance;
[launch](../../docs/simulations/ni_births_lowT_010_20261008.md#gpu-nucleus-search-october-8).

[`ni_lowT_gpu_search_400ps_20261008.json`](../analysis/ni_lowT_gpu_search_400ps_20261008.json)
supersedes that pending GPU revision after the user's duration change. All 36
sources use exact native observations through at most 420 ps including cooling;
every completed source/detector has an outcome row, including zero births.
Extraction waits for the MD producers and successful hard-stop controller.

The current [`ni_lowT_gpu_search_full_20261008.json`](../analysis/ni_lowT_gpu_search_full_20261008.json)
supersedes the capped analysis at the user's clarification: use all saved native
frames, including full histories longer than 420 ps and verified retained tails.
It preserves the 400-ps simulation hold limit and the explicit zero-birth table.

[`ni_lowT_gpu_search_handoff_20261008.json`](../analysis/ni_lowT_gpu_search_handoff_20261008.json)
uses the same full-history analysis after the last two sources move off RTX6000.
A pinned handoff receipt replaces the old allocation dependency with the two
new MD jobs and their completion gate; source plans and scientific settings remain.
