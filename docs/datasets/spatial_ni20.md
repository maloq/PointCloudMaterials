# Spatial training with 20% Ni

This is the prepared parent release. The current dataset registry entry now
points to the [crystal/interface enrichment with merged evaluation](spatial_ni_enriched.md).
The parent values and cache below remain unchanged.

Dataset ID: `spatial-training-ni20`. Recipe:
[`spatial_ni20.json`](../../configs/data/spatial_ni20.json).
**Complete, October 8:** all inputs and targets are prepared and verified.
This extends the actual 1,056,768-row spatial fitting population, rather than
the larger candidate pool. All previous training observations remain in their
original order; Ni is appended. This is spatial pretraining data, not a nucleus
prediction cohort or a set of new independent simulation runs.

## Population

| Material | Training neighborhoods |
| --- | ---: |
| Al | 593,617 |
| Mg | 44,369 |
| Ti | 57,284 |
| Ta | 361,498 |
| Ni | 264,192 |
| **Total** | **1,320,960** |

Ni is **20% of the expanded training set**, equivalent to adding 25% of the
previous row count. All prior Al selection (192,960), calibration (16,848) and
test (45,291) rows remain held out. Ni contributes separate **4,096 selection**
and **4,096 test** neighborhoods, with no calibration subset.

The frozen Ni inventory contains **53 completed production trajectories** from
`ni-births-010-20261007`, `ni-births-010-20261007-extension01`, and
`ni-births-lowT-010-20261008`. Original roles are retained: 19 training, 17
selection and 17 test melt ancestors. Seven unfinished sources are excluded
and named in the preparation plan. The potential is Mendelev 2012 Ni EAM;
manifest, potential, completion and metadata hashes are recorded per source.
Engineering checks and nucleus-recovery experiments are not selected.

Each 32,000-atom Ni trajectory supplies 320 deterministic tracked centers
(1% of atoms). Actual saved timestamps select frames at least 3 ps apart from
native 0.10-ps data. Within each role, rows are sampled uniformly without
replacement from the frame/center candidate pool. Longer available histories
therefore contribute more candidates. No crystallinity, birth or other outcome
label chooses the rows. The simulations' own stopping rules still affect which
observations are available. Rows from the same source or atom are correlated.

## Inputs and targets

Inputs are unrelaxed, center-relative nearest-80 periodic neighborhoods, including
the center. Stored positions are float32 derived from the verified saved Ni
float16 positions. There are no species, material, scale, velocity, temperature
or time features. Previous observations and material normalizations are unchanged.

Ni uses the existing fixed-material normalization rule, calibrated on **training
sources only**: 4,000 points across eight sources, saved frame 2, nearest-160
distance quantile 0.995 multiplied by 1.02. Its scale is **8.2279559213 Å**;
coordinates are multiplied by **1.1085850759**, using the unchanged Al reference
of 9.1213891395 Å. Exact calibration sources and atom IDs are saved in `plan.json`.

Targets have **198 columns: 40 bond-order, 45 CNA and 113 TDA**. The pinned
[`spatial_active_cna_targets.json`](../../configs/data/spatial_active_cna_targets.json)
schema comes from the previous training-only variation mask. It is retained for
this extension; previously excluded TDA columns are not recalculated or reselected
using Ni or held-out errors. Descriptor support is radius <8 in normalized
coordinates. See [target preparation](../target_preparation.md) for definitions.

Previous materials' target values are copied from verified prepared caches;
Ni values are calculated with the safe native alpha-persistence backend.
Target means, scales and active masks are refitted on the **combined training
rows only**. The spatial objective assigns one third to each family, with TDA
split equally over its 18 groups and their active coordinates. Held-out targets
receive the training transform. New fits must rebuild validation packets using
this transform; old standardized packets are not interchangeable.

## Consumption and preparation

External cache: `${storage:cache}/spatial-pretraining/ni20-20261008-r2`.
Plan identity: `e830847334cf10d2772fab37697a2390c441aaaa0339dfdb2637ee46125dad2c`.
Receipts: `output/spatial-dataset/ni20-20261008-r2/technical/`.

`complete.json` is the final readiness marker. `consumer.json` contains the
cache paths, prepared identity, row count, target schema and normalization for
new fitting configurations. Merge these data fields into a new training recipe;
they are directly consumed by `spatial_data()` / `PackedPatches`, while
`CachedPatches` supplies original-role held-out observations. Training remains a
separate action; this workflow does not start or change encoder fits.

`packed/` stores contiguous training positions and raw targets. `shards/` retains
each source's positions, targets and `row_ids.npy`: Ni IDs are `(saved frame,
center atom ID)`, old training IDs point to their original packed row, and old
held-out IDs point to their original role-specific row. The parent packed release
retains the mapping back to its original candidate pool. All files are checksummed.

```bash
conda activate pointnet-torch214
python -m src.postprocessing spatial-dataset plan --config configs/data/spatial_ni20.json
python -m src.postprocessing spatial-dataset run --config configs/data/spatial_ni20.json
```

Preparation is source-parallel, CPU-only, and resumable. Submission freezes code
and requires the matching successful Ni reference audit recorded in
`technical/preflight.json`. Job **1028722** completed with exit code 0 in
**3 minutes 2 seconds**, using 16 CPU workers on `nodecpu04`. Resume an interrupted
execution using its frozen code and configuration from `technical/launch.json`,
preserving the plan identity. The initial attempt 1028692 failed
during checksum verification and was cancelled; its receipts are retained.

The successful preflight calculated 82 real Ni neighborhoods, verified 16 against
the full reference descriptor implementation (maximum stored-value difference
**0**), and verified completed-shard reuse. Publication checks all shard hashes,
training/held-out ancestry separation, packed-reader shapes and target transforms.
The final audit checked 159 early/middle/late neighborhoods from **all 53 Ni
sources**, again with zero stored-value difference against the reference. It
confirmed that all 1,056,768 previous training rows and their retained raw targets
are bitwise unchanged, all original roles have disjoint ancestry, and all 198
retained outputs remain active. See `technical/final-audit.json`.
Population counts are exported with frozen metric definitions in
`output/spatial-dataset/ni20-20261008-r2/tables/`.
The packed release identity is
`994d6e8e7e4225509cc9460ef8b01c940b9560fe3306f18b8c0fad8165e890fb`.

To admit newly completed Ni simulations later, use a new cache/output recipe and
freeze a fresh plan; do not silently append to an identity already used for fitting.
The registry's stable dataset ID can then point to that completed release.
