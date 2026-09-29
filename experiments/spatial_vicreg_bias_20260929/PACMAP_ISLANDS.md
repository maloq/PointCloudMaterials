# What are the detached islands in the joint-descriptor PaCMAP?

The screenshot matches `descriptors-joint-all_test`, not a neural embedding.
It shows 24,960 held-out atom-time observations, using training-interface-fitted
standardization and family balancing of TDA, bond order and CNA. Six regions
were selected from the screenshot before inspecting their descriptors. These
are exploratory selections, not new physical ground-truth classes.

[Annotated plot](../../output/spatial_vicreg_bias/matched-al64-20260929/analyses/pacmap-islands-v1/plots/annotated-islands.png)
· [Measurements and definitions](../../output/spatial_vicreg_bias/matched-al64-20260929/analyses/pacmap-islands-v1/technical/audit.json)
· [Exact observation identities](../../output/spatial_vicreg_bias/matched-al64-20260929/analyses/pacmap-islands-v1/data/selected-rows.npz).

## Observed identities

| Region | Observations | Physical reference and descriptor signature |
|---|---:|---|
| A: purple upper-left | 765 | 756/765 (98.8%) centers are PTM HCP. Median CNA fractions are 0.5 for 421 and 0.5 for 422; median q4=0.116, q6=0.466. Median input crystal fraction 93.75%. |
| B: orange upper-right | 409 | All centers are PTM Other, but median input crystal fraction is 75%. Median finite interface distance 2.88 Å (12 observations lack an accepted interface). Median q6=0.479. The n80 H2 maximum log-lifetime median is 0.669 versus 0.293 for A, consistent with a different cavity-persistence signature. |
| C: thin far-left | 451 | All centers PTM Other; median input crystal fraction zero. Differences from nearest main-population points are dominated by rare fixed-3.2-Å CNA 555/544 signatures. |
| D: bottom-left | 602 | 599/602 centers PTM Other; median input crystal fraction zero. Enriched CNA 666; median bond fraction about 0.077 at 3.6 Å and with adaptive12 cutoff. |
| E: bottom-right | 859 | All centers PTM Other; median input crystal fraction zero. Enriched CNA 444; adaptive12 median fraction 1/12. |
| F: small central bridge | 242 | 225/242 (93.0%) centers PTM FCC. Median input crystal fraction 86.25%; median interface distance 2.73 Å. CNA fixed36 fractions are 2/3 for 421 and 1/6 each for 544/433. |

Counts are atom-time observations, not independent atoms or nuclei. The regions
cover respectively 28, 28, 30, 30, 30 and 27 source trajectories. Repeated
observations are retained. A/B have more repeated source-atom identities than
the liquid regions. Source/frame metadata are audited, never model inputs.

## Do they exist outside the projection?

For every selected point, find its 20 exact Euclidean nearest other points
among all 24,960 displayed observations. The table gives the percentage of
those neighbors belonging to the same screenshot region. Each family uses
its frozen active columns and training standardization. The clipping control
caps standardized coordinates at ±5 before the original family balancing.
No model, clustering or projection is refitted.

| Region | Joint | Bond order only | CNA only | TDA only | Joint, clipped |
|---|---:|---:|---:|---:|---:|
| A | 99.2% | 92.9% | 96.5% | 8.9% | 99.3% |
| B | 95.4% | 85.9% | 65.6% | 90.6% | 93.4% |
| C | 99.3% | 11.9% | 98.8% | 4.1% | 79.1% |
| D | 95.7% | 5.3% | 96.9% | 4.9% | 69.3% |
| E | 94.4% | 7.7% | 94.2% | 6.4% | 87.9% |
| F | 92.9% | 5.7% | 95.3% | 4.6% | 93.3% |

Thus the original feature metric already separates these populations locally;
PaCMAP did not create that local separation from nothing. However, their
separation is often specific to the descriptor family. In particular, C/D/E/F
mix extensively with other observations in bond order and TDA. Neighbor purity
is not a predictive or thermodynamic validation, and depends on region size and
sample density; this is a within-region sensitivity comparison, not a ranking
of which region is most physical. Discrete-feature distance ties also make the
exact CNA-only purity dependent on neighbor tie ordering.

## Why the metric deserves scrutiny

The mean-square distance uses `(feature - training_mean)/training_sd`, then
divides each family by the square root of its active dimension. This balances
average family scale but does not bound rare individual coordinates.

For C, `cna/fixed32_555` has training standard deviation 0.00324. One qualifying
bond can change its fraction by roughly 0.1, about 31 standard deviations before
family balancing. This produces a large metric change from a discrete bond
event. Clipping reduces C/D neighbor purity substantially, while leaving A/B
largely intact. Hard CNA distance cutoffs can also change a bond signature when
a distance crosses the threshold; we have not measured perturbation stability
here. The actual producer is `cna_packet`, with fixed cutoffs 3.2/3.6 Å and its
recorded adaptive12 cutoff, not a whole-atom OVITO CNA phase label.

For B, TDA accounts for 99.7% of **pooled squared distance** to nearest main-
population observations. Extreme samples dominate that pooled statistic:
the median **per-observation** TDA fraction is 79.7%. `n32_h2_betti3.25` and
`n80_h2_image26`, with training standard deviations about 8.0e-7 and 2.1e-7,
account for 76.5% and 19.4% of pooled distance. These are smooth descriptor
values, not integer counts of cavities. Nevertheless, B remains coherent in
bond order and after clipping, so it cannot be dismissed as a numerical-tail
artifact. Its longer H2 persistence warrants inspecting actual local geometry
and finite-patch boundaries before assigning a defect type.

## Interpretation and limits

A is the clearest independently supported structural distinction: HCP-like
stacking. In FCC materials, stacking faults and coherent twin boundaries can
appear as HCP layers, but HCP labels alone do not identify a specific planar
fault. That requires spatial arrangement and orientation information; see
[OVITO's planar-fault method](https://www.ovito.org/manual/reference/pipelines/modifiers/identify_fcc_planar_faults.html).
B is a candidate defective/irregular environment in crystal-rich surroundings,
not established bulk liquid. F is mostly FCC-like, with an altered bond graph.
C/D/E are predominantly crystal-poor, disordered local environments with
specific bond motifs. A 555, 666 or 444 bond does not establish a complete
icosahedron, BCC phase or distinct liquid state.

The large empty gaps and thin shapes are visualization geometry, not free-energy
barriers, physical distances or evidence of separate phases. PaCMAP optimizes
neighbor, mid-near and far pairs ([primary implementation](https://github.com/YingfanWang/PaCMAP));
the location of an island does not specify its physical relation to crystal.
We did not vary projection seed or neighborhood count in this audit.

For the interface research question, prioritize A/B/F for direct MD-space and
local-geometry inspection. C/D/E need cutoff/jitter and feature-scaling controls
before becoming liquid subclasses. This audit supports treating classical
descriptor clusters as a representation with its own biases, not automatically
as ground truth for neural clusters. It makes no precursor, future-fate,
metastability or causal VICReg claim.

## Reproduction

Implementation: `src/research/spatial_vicreg_bias/island_audit.py`.
Slurm CPU jobs 1013992 and 1013994 completed successfully; the latter adds
feature-family and clipping controls and is the reported final artifact.
Use conda `pointnet-torch214` and run within a Slurm CPU allocation:

```bash
python -m src.research.spatial_vicreg_bias.island_audit \
  --config configs/analysis/interface_pacmap.json \
  --output OUTPUT_DIRECTORY
```

Outputs are an annotated saved layout, exact row identities, and JSON statistics
with definitions, feature-model hashes, projection hash and implementation hash.
No neural inference/training or W&B run; original plots, clusterers and metrics
remain unchanged.
