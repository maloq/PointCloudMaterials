# Rich structural partitions around the interface

User correction: compare independently formed clusters of rich TDA, bond-order
and CNA vectors with neural embedding clusters, prioritizing the interface and
its spatial surroundings. The previous family-averaged descriptor prediction
scores answer a different question. They neither implement this comparison nor
establish that all intermediate clusters are smoothing artifacts.

Questions:

1. Do neural clusters near the interface correspond to reproducible groups in
   rich structural feature space, and which signatures distinguish them?
2. Does correspondence remain among atoms on the same side and at comparable
   interface distance/crystal fraction, beyond simply partitioning phase bands?
3. How do these correspondences change from epochs 0/1/4/8/12/18/24, with
   same-center, half-neighbor and full-neighbor VICReg, in encoder and projector?

The frozen Al uniform structural cohort, existing neural cluster assignments and
exact 80-atom descriptors are reused. New independent cluster fits use only train
ancestors. Both all-phase and interface-focused fitting are retained; K=3/6/7/10
and three fixed descriptor seeds provide sensitivity and initialization stability.
TDA, bond order and CNA each get their own partition, plus a family-balanced joint
partition. Primary comparison is the 12-Å neighborhood of a static crystal-side
boundary layer; crystal and disordered-side shells and small defect pockets are
reported separately. See the [complete metric definition](../../docs/metrics/interface_cluster_correspondence.md).

Evidence will consist of bidirectional contingency heatmaps, full multivariate
feature signatures, ARI/AMI and entropies within physical shells, source-bootstrap
intervals, and matched spatial shuffles. Scalar crystal-fraction/diffusion/q6
controls receive the same comparisons. A positive coarse match alone is not
sufficient to call a cluster a distinct interface state. Stable descriptor groups
with correspondence beyond distance/fraction provide stronger support, but do
not prove metastability or predictive precursor structure.

No encoder training, future labels, explicit time/temperature inputs or new
simulation is introduced. This diagnostic does not change the preceding results.
The original 64 sampled centers per snapshot only support sparse spatial
projections; dense maps require a separately declared extraction.

Reproduction (submit once):

```bash
python -m src.research.spatial_vicreg_bias.correspondence submit \
  --config configs/analysis/interface_cluster_correspondence.json
```

[Execution record](../../docs/spatial_vicreg_bias.md) ·
[Published analysis location](../../output/spatial_vicreg_bias/matched-al64-20260929/analyses/interface-correspondence-v1/README.md).

## Requested PaCMAP visualization

[2D and interactive 3D PaCMAP](../../output/spatial_vicreg_bias/matched-al64-20260929/analyses/interface-pacmap-v1/README.md) uses identical held-out atoms across raw encoder/projector and independent TDA, bond-order, CNA and joint spaces. Views cover epochs 4/12/24 for all nine models, with all-test and interface-within-20-Å populations. Neural weights remain frozen; only forward inference is regenerated if needed. High-dimensional cluster assignments are used for coloring, never inferred from the PaCMAP layout. [Frozen definitions](../../docs/metrics/interface_pacmap.md).
