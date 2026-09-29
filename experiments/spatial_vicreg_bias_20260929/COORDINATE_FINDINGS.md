# Archived GeoFormer epoch 34: what the native coordinates show

The [completed coordinate analysis](../../output/spatial_vicreg_bias/archived-geof34-coordinates-20260929/analyses/coordinate-profiles-v2/README.md)
uses raw encoder outputs and the once-applied projector. Coordinate identities
were selected on the fitting half of 174 ps, then frozen for opposite-region
measurements at 166/174/177 ps. These are archived training snapshots; the spatial
holdout separates coordinate selection but does not establish source generalization.

## Findings

- At 177 ps, the three leading selected raw coordinates have absolute distance
  Spearman correlations 0.743–0.778 across the mixed population. In supports
  containing no PTM-detected crystal, their absolute correlations are <=0.031.
  The selected projector coordinates show the same pattern: 0.685–0.764 versus
  <=0.030. At 174 ps, mixed-population correlations are around 0.40–0.43; the
  mostly liquid 166 ps snapshot has weak correlations for these coordinates.
- The training-defined bulk phase direction correlates with distance at about
  0.80 for 177 ps, about 0.40–0.42 for 174 ps and 0.04–0.06 for 166 ps.
  Distance is a crystal-core/bulk-liquid distance contrast, not an independently
  reconstructed signed interface distance.
- Individual atom-centered transects are not uniformly smooth monotone ramps.
  Several projector coordinates dip strongly in mixed/interfacial patches and
  return toward a liquid plateau; other paths fluctuate or cross another local
  region. A high pooled correlation therefore does not establish smooth local
  variation. These dips could reflect interface geometry, nonlinear response to
  phase mixing, encoder sensitivity, or a combination. This assay cannot choose
  among those explanations.
- The strict-clear result concerns the selected **phase-associated coordinates**.
  It does not show that the entire 128-dimensional representation lacks liquid
  structural information. The source-separated descriptor readouts will test that.

## Consequence for the experiment

Retain crystal and interface inputs in the main analysis. Separately report
liquid centers with crystal visible in their crop, and fully PTM-clear crops.
Track both raw z and projector y, native variance/rank, phase-axis profiles,
per-pair contraction and physical descriptor retention. Do not use attractive
cluster shells or broad pooled correlations as proof of an intermediate state.

The checkpoint's saved recipe has `vicreg_neighbor_view=false` and FactorVAE
on. It is a descriptive reference, not the S1 treatment. The nine newly submitted
plain-VICReg fits isolate the pair relation using identical four-view inputs.
Future-prediction claims require their own source/ancestry and crystal-visibility
controls; none is inferred from these static plots.
