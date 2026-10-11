# Full-cell Ta birth and dissolution search

The historical analysis covers 30 trajectories. On 2026-10-03 the user deleted
29 raw trajectories, retaining the earlier million-atom baseline and original
Ta snapshots. Derived birth/fate tables, crystal-content summaries and crops
remain available; full-cell recomputation now depends on the removed inputs.
See [the cleanup receipt](storage/ta_simulation_removal_20261003/receipt.json).

```bash
python -m src.research.crystallization_origin.ta prepare --config configs/analysis/ta_births_20261002.json
python -m src.research.crystallization_origin.ta submit --config configs/analysis/ta_births_20261002.json
```

Use conda `pointnet-torch214`. Covers six earlier Ta runs plus24 completed shooting
futures; six parent configurations, one conservative preparation-ancestry group.
Five parents have10,000,422 atoms and one has1,024,000. The exact0.5ps grid and Ta
cutoffs match the earlier external-material audit. Its six full-cell PTM/graph
artifacts are verified and reused;24 shooting futures receive a new full-cell
search. No training, simulation or relaxation is launched.

Output: `${storage:training_storage}/nucleus_fates/ta-20261002`.
The immutable source plan is `technical/plan.json`; scheduler receipts are
`technical/launch.json`. Four CPU lanes each use two independent source workers,
eight PTM threads per source:64 CPUs in total,64GiB per lane. Every lane has a
24-hour allocation limit. This is a limit, not a runtime estimate. Large-frame
PTM is saved after each frame; the result collector waits for all four lanes.

Per-source `progress.json`, `graph-progress.json`, `stage.json`, and
`complete.json` expose progress, failures and completion. Each stage fails with
the affected source on checksum or reconstruction mismatches. Resume a failed
lane from `technical/code` with its frozen configuration and `worker --index N`;
do not submit a second complete queue on top.

Results go to `analyses/births-fates-v1/`, with grouped metric CSVs and immutable
definitions. Newly observed births, parent-present embryo growth, front-associated
events, failed episodes, merging and censoring remain separate. Strict pre-birth
history eligibility and predictor fitting are later stages.

[Definitions](metrics/ta_births.md) · [Ta simulations](simulations/ta_shooting_20260926.md)

## Completed search and crystal content

All30 sources completed. A separate read-only summary is available at
`${storage:training_storage}/nucleus_fates/ta-20261002/analyses/crystal-content-v1/README.md`,
with full fraction trajectories, endpoint tables and ancestry sensitivity.
Reproduce it with `python -m src.research.crystallization_origin.ta_content --config configs/analysis/ta_births_20261002.json`.

The24 new shooting branches show growth from existing crystalline structures.
For example, the3.0ns parent's PTM crystal fraction increases from12.457% to a
four-shot mean16.705% after24ps. All six starting configurations contain crystal.
The shooting search classifies193 establishments as already present at the start,
54 as connected to starting embryos, and one as interface-associated.

Of those54 parent-connected establishments,46 retain a directed strong-overlap
path to the starting frame. Eight rely on weaker overlap links and are therefore
ambiguous under a less conservative ancestry interpretation. Their saved cases
are listed in `tables/parent-ancestry-sensitivity.csv`; this sensitivity summary
does not relabel them or establish eight new physical births. Zero accepted
isolated births is the result of the frozen conservative rule.

The previously reported four brief dissolved candidates include three shooting
episodes and one historical episode. The24 shooting branches contain28 verified
dissolutions,17 with isolated origins, of which only three retain the required
strong-connected lineage. None meets the sustained size criterion.
[Additional metric definitions](metrics/ta_crystal_content.md) distinguish
crystal fractions, event counts, weak ancestry and label limitations.

## Submitted October 2, 2026

All30 trajectories passed source identity, schema, potential and timeline binding.
The inventory contains1470 observation frames:294 historical PTM frames reused
and1176 frames in the24 new full-cell searches. Five futures have1,024,000 atoms;
the remaining25 have10,000,422 atoms.

The complete historical million-atom source passed fate derivation. Recomputed
PTM on an identical new million-atom parent frame matched every historical label
(zero mismatches); material/potential fields were also checked. These production
artifacts are retained for reuse. The one-frame check used four PTM threads;
the submitted queue uses eight, recorded in `technical/real-source-verification.json`.

Slurm array **1018665_[0-3]** runs four16-CPU lanes (two source workers each),
with collector **1018666** dependent on successful completion. All four lanes
were observed running on nodecpu04 and nodecpu06. Source/config snapshots are in
`technical/code`; the Al queue uses its own already frozen producer.
