# Full-cell birth search in Al shooting data

October3 recovery:823/824 futures had completed. A second mutable-registry read
failure in lane4 left one source unstarted, blocking collector1018609. Job1019799
resumed that frozen lane using a fixed registry snapshot, reused102 completed
sources and processed the remaining source. The collector was reconnected to
this successful repair. See `technical/registry-retry-20261003.json`; the earlier
lane2 recovery and all original receipts remain preserved.

The final collector completed October3:824/824 futures,77 new isolated
establishments and30 brief isolated verified-dissolution candidates. No sustained
failed candidates passed the declared criterion. Counts are branch-local,
correlated harvest candidates; strict crystal-free prediction-history eligibility
is a separate step. The final tables are in `analyses/births-fates-v1/`.

```bash
python -m src.research.crystallization_origin.shooting prepare --config configs/analysis/shooting_births_20261002.json
python -m src.research.crystallization_origin.shooting submit --config configs/analysis/shooting_births_20261002.json
```

Use conda `pointnet-torch214`. The first command binds registered campaign roots,
available complete branches, canonical parent/source ancestry, actual 0.30 ps
timelines and nested continuation identities. Missing outcomes are inventoried.
The second freezes code and submits eight CPU lanes, eight source workers each,
then a dependent collector. No GPU, new simulation, relaxation or W&B fit is needed.

Each branch independently resumes verified full-cell PTM chunks, constructs its
ancestry graph and writes birth/fate labels. A checksum failure stops that lane
with the offending source. Use the original frozen `technical/code/config.json`
and `worker --index INDEX` to resume; never submit a second complete queue on top.

Output: `${storage:training_storage}/nucleus_fates/al-shooting-20261002`.
`technical/inventory.json` records coverage; `technical/launch.json` records Slurm
IDs. `technical/lane-N.json`, per-source `progress.json`, `stage.json` and
`complete.json` expose progress and failures. Final scientific tables are in
`analyses/births-fates-v1/tables/`, with frozen metric definitions/hashes.

The search covers the original nested pilot plus its fixed24 continuation once,
the large 48 ps shooting campaigns and completed 15/48 ps top-ups. The two
small smoke campaigns were deleted by a separate cleanup during preparation;
their exclusion is bound to that cleanup receipt in the configuration.
Parent-present embryo growth is separate from a newly observed birth. Historical
Al64 candidates and the running appearance-prediction comparison are unchanged.
Predictor-ready history/control extraction requires a later strict crystal-free
eligibility pass; current results are a harvest catalogue.

[Scientific definitions](metrics/shooting_births.md) ·
[Original Al64 fate audit](nucleus_fates.md)

## Submitted October 2, 2026

Verified inventory: **824 futures, 70 unique parent configurations, 29 ancestral
source runs and 106,557 observations**. The 480 main 48 ps futures, 40 completed
48 ps top-ups, 160 CSLD 15 ps futures and 144 nested futures are included. Original
nested segments and their fixed24 continuations are searched once. There are
440 declared branches without completed outcomes across the top-up/preparation
records; three deleted smoke/preparation campaign records are excluded separately.

Full real-source checks passed for one 48 ps Langevin future, one 15 ps CSLD
future and one nested24 continuation. Their production outputs are reused.
The 48 ps check exercised establishment and parent-present-embryo attribution.

Slurm array **1018608_[0-7]** runs eight processes per lane (64 CPUs total), with
103 futures per lane. Collector **1018609** waits for all lanes to finish.
All eight lanes were observed running, with all 64 first-wave trajectory input
checksums verified. Each lane has a 12-hour allocation limit; this is a limit,
not a completion estimate. Scientific definitions and source code are frozen in
the output's `technical/code` directory.
