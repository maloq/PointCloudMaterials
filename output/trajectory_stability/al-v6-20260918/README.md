# Trajectory stability of MACE, GATr and descriptors

Matched exploratory audit of 10 training/selection-held-out Al MEAM sources at 400, 450, 500, 510 and 520 K. Four seeded centers per source, all 801 frames from 0 to 600 ps: 32,040 observations and 32,000 adjacent pairs. Five separate training sources supply 420 normalization observations. No model is fitted or updated.

MACE selected update 1465; GATr selected update 1216. Weights and source selection were frozen at the beginning of this audit. Native compiled mixed precision with FP32 geometry; raw encoder z128, no projector or prediction head.

| Representation | RMS jump at 0.75 ps (95% source interval) | Roughness | Effective rank |
|---|---:|---:|---:|
| MACE | 0.723 (0.696–0.749) | 1.480 | 1.87 |
| GATr | 0.680 (0.659–0.701) | 1.473 | 6.28 |
| TDA (80 atoms) | 0.371 (0.369–0.372) | 1.486 | 1.20 |
| SOAP (7 Å) | 0.587 (0.571–0.601) | 1.474 | 2.76 |
| Bond order | 0.278 (0.262–0.293) | 1.446 | 1.16 |
| Radial (32) | 0.661 (0.644–0.678) | 1.478 | 2.92 |
| Angular (16) | 0.513 (0.506–0.520) | 1.480 | 1.70 |

[Normalized RMS jump](../../../docs/research_glossary.md#normalized-rms-jump) divides movement by the RMS independent-pair distance of the separate training reference. A value of 1 therefore means movement as large as that reference distance. Roughness measures second differences relative to adjacent increments: 0 for a linear path, about 1.5 for independent frames, and 2 for exact alternation. Intervals resample complete sources within temperature; they condition on the fixed training reference.

![Comparison](plots/comparison.png)

![Trajectories](plots/trajectories.png)

![All tracked atoms](plots/all-tracks.png)

[Explore all 40 trajectories interactively](explore.html).

## Interpretation limits

Finite-lag jumps contain real atomic motion, neighborhood replacement and stored-coordinate quantization. This measures observed temporal variability, not a pure numerical-noise estimate. No temporal smoothing is applied. The native saved cadence cannot resolve sub-0.75-ps jitter. The sources use float16 full-box positions; the audit does not infer a full-precision noise floor.

Descriptors have different observation support: the encoders use the trained 16.87 Å neighborhood, TDA uses the nearest 80 atoms including the center, SOAP uses a 7 Å nominal Gaussian-density cutoff, radial/angular features use the 5–7 Å smooth support, and bond order uses 12-neighbor local and neighbor-averaged invariants. This compares the deployed representations, not an isolated architecture effect. The raw covariance-trace normalization is basis invariant; coordinate-standardized results are also exported as a sensitivity check. A small jump or low effective rank alone does not establish usefulness or collapse.

PTM-unclassified is not synonymous with liquid. Phase-conditioned tables require the same PTM class at both endpoints and do not establish metastability. These test sources were explored in earlier research; they are held out from these encoder fits/selection, not a newly untouched confirmatory test. Only one trained seed per encoder is evaluated.

[Exact metric definitions](tables/METRICS.md) · [Scores](tables/summary.csv) · [Paired comparisons](tables/paired-comparisons.csv) · [Per-source values](tables/per-source.csv) · [Phase conditions](tables/phase-conditioned.csv) · [Neighbor turnover](tables/neighbor-turnover.csv)

Reproduce with `conda run -n pointnet-torch214 python -m src.research.trajectory_stability --config configs/analysis/trajectory_stability.json`. Completed reports are preserved; change the output directory for another audit.
