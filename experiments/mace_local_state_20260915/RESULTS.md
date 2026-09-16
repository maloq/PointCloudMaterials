# Frozen local-state comparison: results

The run completed successfully. Learning a distance from local-group physics
improved the organization of the frozen features, but did not produce the desired
coherent liquid states. The physical map loses substantial instantaneous topology;
the tested short-time canonical maps do not give a consistent improvement over
the inner-feature control. There is no justification yet for replacing the encoder
with either transformation.

MACE and the existing VICReg projector stayed frozen. These results concern affine
maps of their features, source-separated physical readouts, and density discovery.
No forecast objective or velocity input was used. The source split is 18 training,
6 validation and 6 test trajectories; this is an existing encoder-development
cohort, not a newly untouched final test set. All nine representations and both
two-frame/snapshot variants completed. The static analysis covered 684,723 centers
across six Al frames.

## Snapshot comparison

Values average the six held-out sources (balanced context counts). Lower errors
and normalized changes are better; higher neighbor recall is better.

| Representation | Change over 0.75 ps | Group-target error | Other group error | Instantaneous TDA error | Relaxed TDA error | Physical-neighbor recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Inner PCA, 16 dimensions | 0.5676 | 0.2054 | 0.1888 | 0.0547 | 0.0341 | 35.02% |
| Inner temporal, 8 dimensions | 0.5772 | 0.1696 | 0.1753 | 0.1390 | 0.0369 | 35.00% |
| Inner temporal, 16 dimensions | 1.0466 | 0.1538 | 0.1611 | 0.1097 | 0.0320 | 31.14% |
| Inner physical, 10 dimensions | 0.1712 | 0.1060 | 0.1372 | 0.1397 | 0.0396 | 45.87% |
| Existing projector PCA, 16 dimensions | 0.1562 | 0.0995 | 0.1477 | 0.1261 | 0.0350 | 38.92% |
| Dual PCA, 16 dimensions | 0.9873 | 0.2828 | 0.2363 | 0.0649 | 0.0452 | 27.59% |
| Dual temporal, 8 dimensions | 0.5080 | 0.1859 | 0.1848 | 0.1422 | 0.0388 | 34.44% |
| Dual temporal, 16 dimensions | 0.9588 | 0.1683 | 0.1765 | 0.1299 | 0.0355 | 31.04% |
| Dual physical, 10 dimensions | 0.2023 | 0.1123 | 0.1441 | 0.1405 | 0.0412 | 45.39% |

“Group-target error” averages ten mean/spread targets used to learn the physical
map. “Other group error” averages mean and spread of w4, w6 and smooth coordination:
six group observables excluded from fitting the physical map. The latter are
additional validation quantities, although correlated with the training targets.
The reported errors come from separately fitted current-structure readouts.
They are normalized MSEs, not percentages of classification mistakes.

Physical-neighbor recall measures overlap of top-eight similar groups, using
calculated group physics as the reference, separately within each source/frame
context. It is not atom-ID retention or agreement of spatial cluster labels.
Chance recall for 64 samples is 8/63, approximately 12.7%.

Temporal change is squared mapped-feature displacement divided by total
within-context TRAIN variance. It measures change relative to local variability,
not raw embedding displacement. Earlier raw-encoder tables used different
normalizations and should not be compared numerically with this column.

Relative to inner PCA16, the inner physical map gives:

- 69.83% lower normalized temporal change at 0.75 ps.
- 48.39% lower error on the ten group targets used to learn the map.
- 27.34% lower average error on the six additional group observables.
- Physical-neighbor recall rising from 35.02% to 45.87%, a gain of 10.85 percentage points.
- 2.554 times the instantaneous-TDA error and 16.11% higher relaxed-TDA error.

Both the gains and the topology losses occur in all six held-out sources. The
temporal-change ratio ranges from 0.286 to 0.314 across sources. This is not driven
by one favorable trajectory. It also does not imply uniform improvement across
every physical observable: mean smooth-coordination error increases from 0.0020
to 0.0219, while the other five additional group errors decrease.

The existing projector control is competitive: it is smoother than the new
physical map, has lower group-target and TDA errors, but somewhat higher average
error on the additional group observables and worse physical-neighbor recall.
The physical map improves a particular distance geometry; it is not a universal
improvement over all existing representations. Adding the tracked-center block
does not improve the physical map's listed snapshot metrics over the inner block.

The 16D inner temporal map is less smooth than the inner PCA control. This map
uses equal-weight canonical coordinates after covariance whitening; introducing
additional weaker directions is a plausible contributor, but was not isolated by
an ablation. This result applies to this linear construction and chosen 0.75 ps
lag, not to every method that learns from short local histories.

## Two-frame averaging and boundary sensitivity

The trailing average uses current and previous features, spanning 0.75 ps. Both
snapshot and averaged variants are evaluated against CURRENT physical targets.

| Representation | Snapshot change | Averaged change | Snapshot instantaneous-TDA error | Averaged instantaneous-TDA error |
| --- | ---: | ---: | ---: | ---: |
| Inner PCA16 | 0.5676 | 0.1515 | 0.0547 | 0.0939 |
| Inner physical10 | 0.1712 | 0.0462 | 0.1397 | 0.1451 |
| Projector PCA16 | 0.1562 | 0.0445 | 0.1261 | 0.1361 |

Averaging lowers the adjacent-time change by about 72–73%, but also removes
instantaneous information. For inner PCA, instantaneous-TDA error rises 71.72%.
For the physical map, the extra increase is 3.87%, on top of information already
lost by the snapshot physical compression. Physical-neighbor recall for the
physical map falls from 45.87% to 42.79%. Averaging is therefore not a free solution.

For the retained 1e-4 A boundary perturbations, the squared jump divided by natural
0.75 ps squared changes is 2.95e-9 for inner PCA and 7.08e-6 for the inner physical
map. The latter amplifies small perturbations more, although this particular
boundary response remains small relative to natural changes. This is not a new
complete storage-precision or velocity-sensitivity audit.

## State discovery did not solve the liquid problem

The separate static discovery fit used 3,000 PTM Other centers in the Al 166 ps
training slab, x<85 A, with x>185 A reserved for spatial testing. HDBSCAN used
minimum cluster sizes 40 and 100 and minimum samples 10.

**Every one of the nine representations found zero selected density clusters
at both settings.** Training labels themselves are all unassigned, so this is
not just a prediction or plotting failure on the test slab. The inspected saved
models confirm their 3,000-point fit arrays and empty cluster catalogs.

All static assignments are consequently rejected, soft unassigned mass is one,
and spatial agreement / physical variance explained by clusters are undefined.
There is no valid spatial-coherence score to compare with GeoFormer here. The
subsequent five frames reuse the same empty catalog; their all-unassigned outputs
are a consequence of that first fit, not five independent failed discoveries.

The protocol excludes a single all-encompassing cluster (`allow_single_cluster=False`).
Thus, zero clusters means that the tested method/settings did not resolve supported
separate groups. It does not establish that local liquid structure is absent.
Relevant patterns may be connected by transitions, described better by continuous
coordinates, or not separated at the sampled density and chosen resolution.
Only two minimum-size settings and one static discovery frame were tested.

The broader MD cohort yields two to six density clusters, but inspecting their
physical profiles shows mostly low-order/high-order separation. For example:

| MD representation / state | Held-out members | Mean group qbar6 | Mean group bond coherence |
| --- | ---: | ---: | ---: |
| Inner PCA16, state 0 | 590 | 0.1723 | 0.2801 |
| Inner PCA16, state 1 | 322 | 0.5001 | 0.9256 |
| Inner physical10, state 0 | 697 | 0.1712 | 0.2780 |
| Inner physical10, state 1 | 154 | 0.5300 | 0.9704 |
| Inner physical10, state 2 | 49 | 0.4959 | 0.9292 |

These rows use the minimum-size-40 catalogs. The extra physical-map clusters
subdivide high-order environments. Likewise, the six-cluster projector result
has one broad low-order group and several groups with higher order. These are not
evidence of six distinct liquid states. Labels are local to each fitted catalog;
their numeric IDs have no shared physical meaning across methods.

## Interpretation

The experiment identifies a useful continuous local-group distance, with gains
that transfer across the held-out MD sources. It also shows that compressing the
representation into those physical quantities discards useful instantaneous
topology. The existing projector is a strong control, and the temporal maps as
tested do not improve the desired balance.

The desired informative, smooth encoder with meaningful liquid organization is
not yet achieved. A next controlled comparison should preserve additional
structural information alongside the physical coordinates and evaluate continuous
local organization as well as uncertain state assignments. These are proposed
follow-ups, not results of this run; no additional training was launched during
the interpretation.

Sources are the run's retained `comparison.csv`, `information.csv`,
`information_summary.csv`, `neighborhoods.csv`, `smoothness.csv`, `states.csv`,
`static_states.csv`, `static_physics.csv`, saved density models and per-sample
memberships. Exact definitions remain frozen in `tables/METRICS.md`.
