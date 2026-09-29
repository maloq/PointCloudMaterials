# Saved-cluster local order — version 1

This diagnostic reuses the exact frozen cluster assignments and representative
sample indices from a completed static analysis. Source-coordinate and assignment
hashes are checked before calculation. It does not fit clusters, run an encoder,
select checkpoints or introduce encoder/predictor inputs.

PTM runs on every atom of each original nonperiodic source snapshot before any
display crop. All saved analysis centers are mapped to exact source atoms. The
existing source protocol excludes boundary centers. FCC, HCP, BCC and ICO templates
are explicitly enabled; all other templates are disabled. OVITO version is saved.
PTM runs with RMSD cutoff zero (disabled) to retain best-template assignments and
residuals. Final type is Other when no template matched or residual exceeds 0.10.
The recorded sensitivity thresholds are 0.05, 0.10 and 0.15. A zero residual with
best type Other is an unavailable match, not a perfect structure.

| Column | Definition |
| --- | --- |
| PTM count / fraction | Number of saved centers of a type, divided by all centers in the stated frame/cluster or pooled cluster. Empty strata have zero count and undefined fraction. |
| Crystalline fraction | FCC + HCP + BCC fraction at the declared RMSD cutoff. ICO is reported separately as local fivefold order. |
| q4, q6 | Steinhardt magnitude of the mean spherical harmonics of the focal atom's 12 nearest-neighbor bond directions, using `bond_order`. |
| w4, w6 | Normalized third-order invariant using the existing Wigner-3j contraction. |
| qbar6 | Magnitude after averaging q6m over the center and its 12 neighbors, with each neighbor's own 12-neighbor environment. |
| mean_q6_coherence | Mean real normalized q6m inner product between center and its 12 neighbors. |
| coherent_bonds_065/070/075 | Number of those neighbors whose q6m inner product exceeds 0.65/0.70/0.75, respectively. |
| density_r12 | 12 divided by the sphere volume at the 12th-neighbor distance, in inverse cubic Angstrom. |
| smooth_coordination | Sum over the same 12 neighbors of exp(-(r/3.7 Angstrom)^8). This is a restricted descriptor, not all-neighbor coordination. |

Continuous diagnostics use up to 64 centers per frame and cluster, uniformly
sampled without replacement with seed 20260926. Strata smaller than 64 use all
members. Sample indices and source atom rows are exported. This stratified sample
is a descriptive comparison; its unweighted pooled distribution does not estimate
the original population mixture. PTM population fractions use every saved center,
not this subsample. Representatives are retained original samples, not selected
after looking at their order scores.

Related frames and overlapping neighborhoods are not independent observations.
No binomial confidence interval or held-out generalization claim is made. PTM
Other does not establish a liquid phase. Highlighted atoms indicate local template
matches; they do not establish a single crystallite or shared orientation.

Display edges are a sparse subset of cutoff-valid connections and do not enter
any metric. The focal first shell retains its cutoff-valid graph; outer context
uses mutual two-nearest connections within the same recorded cutoff. Colors use
distance from the actual focal atom. No hull is rendered. Cutoff and display
settings are preserved in the rendering receipt separately from calculations.

PTM reference: [Larsen et al., 2016](https://doi.org/10.1088/0965-0393/24/5/055007).
[OVITO method and RMSD threshold](https://www.ovito.org/manual/reference/pipelines/modifiers/polyhedral_template_matching.html).
