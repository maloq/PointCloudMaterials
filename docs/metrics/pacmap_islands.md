# Exploratory PaCMAP island diagnostics

Implementation: `src/research/spatial_vicreg_bias/island_audit.py`.
This is a screenshot-selected descriptive audit, not a held-out predictor score.
All 24,960 saved test rows are retained, with six rectangular selections fixed
before descriptor inspection. Definitions are also exported in `technical/audit.json`.

- **Neighbor membership:** average fraction of the exact Euclidean 20 nearest
  other displayed observations in the same selected region. Exclude self by row
  identity, not distance. Compute separately for frozen joint features, each
  family, and joint features with training z values clipped to ±5 before family
  balancing. Feature ties can affect which neighbors are returned.
- **Outside/inside distance ratio:** each row's nearest distance outside its
  selected region divided by its nearest other distance inside. Report 10th,
  50th and 90th percentiles. Inside distances are float64; outside candidate
  identities come from exact FAISS search and distances are recomputed float64.
- **Descriptor attribution:** match each selected row to its nearest point in
  the population excluding all six regions. Sum squared differences per feature
  and divide by total squared difference; sum feature shares by family. Also
  report quantiles of per-row family shares to expose domination of pooled
  contributions by rare extreme observations. This attributes the declared
  Euclidean metric, not physical causation.
- **Physical summaries:** PTM counts; 10/50/90 percentiles of the consumed
  patch's crystal fraction, unsigned accepted-interface distance and selected
  raw descriptors. Exclude infinite distances from quantiles and report their
  count. Source/frame counts and distinct source-atom identity counts describe
  repeated observations, not independent statistical replicates.

Means, standard deviations, active columns and family balance come unchanged
from the train-interface-fitted descriptor model used by the saved projection.
No model, clusterer or projection is refitted. No phase, precursor, free-energy
or metastability claim follows from these statistics. Region sizes differ;
neighbor membership is a within-region sensitivity comparison, not a quality
ranking across regions. See the [findings](../../experiments/spatial_vicreg_bias_20260929/PACMAP_ISLANDS.md).
