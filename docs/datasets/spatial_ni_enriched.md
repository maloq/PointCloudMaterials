# Ni crystal interiors and interfaces in spatial training

**Complete and verified, October 8.** CPU job **1028846** completed in
3 minutes 2 seconds. All new targets and the merged evaluation set are ready.

This adds **52,838 Ni training neighborhoods**, 20% of the previous 264,192 Ni
rows, to the [completed spatial fitting set](spatial_ni20.md). The additions are
**26,419 crystal interiors and 26,419 crystal/disordered interfaces**. The
expanded set contains **1,373,798 training rows**, including **317,030 Ni (23.08%)**.
All previous raw observations and target values are retained. Validation and
test are merged into **one pretraining evaluation set**. Their 8,192 Ni rows
also receive 20% more data: **1,638 additions**, split into 819 crystal interiors
and 819 interfaces. Evaluation totals **248,081 rows**, including **9,830 Ni**.
The separate 16,848 Al calibration rows remain preserved outside this merge.

Recipe: [`spatial_ni_enriched.json`](../../configs/data/spatial_ni_enriched.json).
External cache: `${storage:cache}/spatial-pretraining/ni-enriched-eval-20261008`.
Output: `output/spatial-dataset/ni-enriched-eval-20261008/`.
The existing registry ID `spatial-training-ni20` points to this completed
extension; its name reflects the initial 20%-Ni mixture, not the enriched share.
The immutable parent remains at its recorded cache location.

## What counts as a crystal or interface

Full-cell periodic PTM is evaluated on saved **unrelaxed** Ni coordinates with
RMSD cutoff 0.1. FCC, HCP and BCC count as crystalline. The local crystalline
fraction uses each center's nearest 80 atoms, including the center, within
normalized radius 8, matching the encoder patch support.

| Added category | Explicit sampling rule |
| --- | --- |
| Crystal interior | Central atom is PTM-crystalline and local crystalline fraction is at least 80%. |
| Interface | Local crystalline fraction is 20–80% (upper bound excluded), with both a crystal-interior center and a disordered center within normalized distance 8. |

A disordered reference center has at most 5% crystalline atoms in its local
support. Periodic nearest-reference distances are used. The two-sided condition
avoids treating every thermally disordered crystal neighborhood as an interface.
This remains an operational structural classification: it does not independently
establish thermodynamic liquid phase, growth rate, event origin or nucleus fate.

The source pool is fixed to the parent's 19 training and 34 held-out Ni ancestors. Native
structural monitoring identifies existing 3-ps-spaced frames with at least 10%
global crystallinity. Twelve training sources supply **182 frames**; 21 held-out
sources supply another **314 frames**. Candidates are classified separately.
At most 16 evenly spaced eligible-frame
indices per source span the observed growth regimes. Every atom can supply a
center in those frames. Exact source/frame/atom pairs already in the original
role are excluded, then a seeded uniform sample without replacement is drawn
separately from each category within training and evaluation.
Sampling is intentionally phase-enriched, not representative of MD occupancy.

## Targets, ancestry and consumption

All previous material scales remain fixed, including Ni's 8.2279559213 Å scale
and 1.1085850759 coordinate multiplier. Encoder inputs remain geometry-only;
PTM and enrichment categories are audit/sampling metadata, not features or new
targets. No minimization, new simulation or encoder training is part of this
preparation.

The same **198 targets** are used: 40 bond-order, 45 CNA and 113 TDA. Targets for
new rows use the audited native backend; parent targets are copied unchanged.
Target normalization is recalculated on combined training rows only. The merged
evaluation rows retain original-role metadata and ancestry, and receive the new
training transform. Rebuild standardized evaluation packets for fits using this release.

`complete.json` marks readiness. The generated `consumer.json` supplies the
cache, packed loader, row count, target schema and normalization fields for a
new training configuration. `inputs/*-category.npy` records additions as
1=crystal interior and 2=interface, aligned with each selected input and its
source shard. `phase/` stores full-frame PTM labels, local fractions, distances,
candidate pools and checksums for reproducing the selection.
The merged role is named `evaluation`; read it with
`CachedPatches(config, config['data']['evaluation_role'])`. It has no source
ancestry in common with training and is not a separate validation/test pair.

Prepared identity:
`0325195ca0b4e060f6a41390823265b58f65267a3ae130148ffff28c4b688162`.
Packed identity:
`6a86e9102ceda0409b527a9058e1a0073f1f1e799ce2ccd3652fa05f34ee65b0`.

All 198 columns were calculated. **197 receive training weight** after refitting
the declared SD >=1e-4 mask: `tda/n32_h2_image20` falls just below the threshold
in the expanded population (its parent SD was 0.0001001792). The retained schema
and numerical target values remain available; this column has zero loss weight.

The final audit verified every added row's phase category, exclusion of parent
frame/atom pairs, disjoint training/evaluation ancestry, and exact preservation
of all 1,320,960 parent fitting rows and 263,291 parent held-out rows. All 66
reference target checks, covering both categories in every participating source,
matched the native targets exactly. Receipts are in `technical/final-audit.json`.
The initial training-only job 1028842 was superseded at the user's request;
its matching completed frame classifications were reused.

```bash
conda activate pointnet-torch214
python -m src.postprocessing spatial-enrich scan-plan --config configs/data/spatial_ni_enriched.json
python -m src.postprocessing spatial-enrich audit --config configs/data/spatial_ni_enriched.json
python -m src.postprocessing spatial-enrich submit --config configs/data/spatial_ni_enriched.json
```

Preparation uses 16 CPU workers. Submission freezes code and checks the successful
real-data phase/target audit. Resume from the frozen execution named in
`technical/launch.json`; completed phase frames and target shards are reused.
Do not create concurrent writers for the same cache. Population CSVs use the
[enrichment metric contract](../metrics/spatial_ni_enrichment.md).
