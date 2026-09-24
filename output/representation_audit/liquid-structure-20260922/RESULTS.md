# Liquid structure representations: plot audit and training diagnosis

2026-09-22. **The plots support a visual difference in how liquid environments are organized. They do not yet establish that one training run progressively erased physically useful information.** The strongest next step is a matched checkpoint audit of conditional liquid physics and predictive information, followed by a small native-encoder ablation. Optimizing for more separated colored clusters would be the wrong target for a continuous transition.

[Browse the indexed plots](index.html) · [Six-run comparison](plots/shortlist-comparison.png) · [Exact inventory](technical/plot-inventory.json)

## What was inspected

The current repository's published `output/<family>/<run>/plots` directories contain105 unique UMAP/t-SNE PNGs after SHA256 deduplication. All66 UMAPs were screened in contact sheets, alongside12 t-SNEs from the archived GeoFrame and current structural-static families. The25 older MACE t-SNEs are indexed but were not individually reviewed. Runtime/code snapshots were excluded, and the entire external archive was not rescanned. No new encoder inference, training, projection fitting or physical score computation was performed for this review. Contact sheets preserve the original images.

This is a visual shortlist, not a quantitative winner ranking. Labels/colors are independently assigned. UMAP and t-SNE distort distances/density and can create apparent gaps; their connectivity is not a transition-path estimate. The UMAP authors explicitly discuss these limitations in [their clustering documentation](https://umap-learn.readthedocs.io/en/latest/clustering.html).

## Visual shortlist

| Candidate | What is visually promising | Main limit |
|---|---|---|
| GeoFrame V2, VICReg + FactorVAE, epoch034 | Clearest connected succession of differentiated color regions among the named examples; C4/C6 occupy connecting bands near the crystal-associated C7. The reanalysis t-SNE preserves a similar qualitative ordering. | Projector output; historical training includes the static data. C5/C6/C7 pooled representatives are FCC; C1–C4 are Other. C6 is not established to be a liquid precursor. |
| GeoFrame multiscale, VICReg + FactorVAE, epoch059 | Elongated differentiated regions and connecting bands; a useful additional reference. | Different architecture. C5/C6/C7 representatives are FCC; focus liquid differentiation on C1–C3 and candidate interface C4. |
| GATr–VICReg, step3072 | Best visual reference among the current GATr runs for ordered liquid color regions. | Dominant crystal-associated cluster is C1, and C2 also has an FCC representative; C3–C7 representatives are Other. Raw PC1 explains98.33% of variance; visually organized is not synonymous with rich dimensionality. |
| MACE local, step622 | Stronger secondary candidate than Epi-direct: an elongated liquid body with comparatively ordered color gradients. | C3 is the crystal-dominated group (90.71% at240ps); the other pooled representatives are Other. Still needs matched physical/prospective validation. |

Epi-direct step768, GATr local622, GATr v6 step1216, GATr temporal-backtracking, MACE v6 step1465 and the selected relaxed MACE more often show one color-segmented liquid body plus a separate crystal-associated region. The latter images can still contain useful continuous physics; mixed colors do not establish meaningless embeddings.

Older MACE `mean-blocks-seed20260910` and `atom-temporal-blocks-seed20260910` have visually interesting ordered bands. They use different history/export and held-out trajectory populations, so they are secondary inspection candidates rather than members of a matched static-Al ranking. Crystallization-transfer trajectory figures and the Zr plot are likewise outside that ranking.

**Representative-label check:** GeoFrame V2 epoch034 C6's pooled representative is FCC (PTM RMSD0.01626), while C4's is Other. C5 and C7 are FCC too. Original GATr C1/C2 are FCC representatives; C3–C7 are Other. MACE local622 has FCC C3 and Other representatives elsewhere. The multiscale059 C5/C6/C7 representatives are also FCC. These are individual representative labels, not cluster-purity estimates. A spatially intermediate environment can already have a crystalline central atom. See [GeoFrame representatives](../../factor_vae_archive/geoframe-v2-vicreg-epoch034/tables/real_md/representatives/10_cluster_representatives_structure_analysis_k7.csv) and [GATr representatives](../../structural_static/gatr-vicreg-step3072-al-20260918/technical/real_md/representatives/10_cluster_representatives_structure_analysis_k7.csv).

Direct image links:

- [GeoFrame V2 epoch034](../../factor_vae_archive/geoframe-v2-vicreg-epoch034/plots/real_md/latent/latent_projection_umap_clusters.png)
- [GeoFrame multiscale epoch059](../../factor_vae_archive/geoframe-multiscale-vicreg-epoch059/plots/real_md/latent/latent_projection_umap_clusters.png)
- [GATr step3072](../../structural_static/gatr-vicreg-step3072-al-20260918/plots/md-umap.png)
- [MACE local622](../../structural_static/mace-local-step622-al-20260919/plots/md-umap.png)

## Corrections to the training narrative

**Epoch034 and epoch159 are different scratch runs.** The saved configs identify VICReg25/25/1 at034 versus VISReg projected-distribution matching at159; both use FactorVAE gamma0.1. Both analyses export the128D MLP projector. They are not an early/late pair from one objective. Compare [034 configuration](../../factor_vae_archive/geoframe-v2-vicreg-epoch034/technical/original-training-config.yaml) and [159 configuration](../../factor_vae_archive/geoframe-v2-visreg-epoch159/technical/original-training-config.yaml).

**Epi-direct's measured rank increased during training.** Its development history goes from invariant participation rank1.809 to4.161, and within-temperature noncrystalline ranks roughly2.67–2.84 to3.72–4.13. The freshly initialized order decoder's MSE also improves, but that is not an encoder-only comparison. Thus a poor final UMAP is not evidence that this run monotonically discarded all liquid variation. [Training history](../../neighborhood_jepa/regularization-20260920/technical/runs/epi-direct-order/validation.jsonl).

**Export, scale and domain differ.** GeoFrame uses projector128; original GATr uses raw encoder128 with16.87Å support; Epi uses LayerNorm invariant128 with7.94Å support. Historical GeoFrame analyses use772,953centers versus684,723 in current static analyses. Original GATr and GeoFrame include the analyzed static files in training, whereas Epi was trained on hot dynamic Al and evaluated here on relaxed snapshots. The latest relaxed encoder has raw export and80 candidate atoms selected on static geometry, whereas training tracked hot-selected candidates across quenching. These are not isolated architecture or loss comparisons. [GATr scope](../../structural_static/gatr-vicreg-step3072-al-20260918/RESULTS.md), [relaxed static protocol](../../../experiments/relaxed_encoder_expanded_20260921/STATIC_AL.md).

## What appears to be missing

The desired object is a **structural state coordinate whose distances retain meaningful differences among liquid environments and whose evolution carries information about later rearrangement/crystallization**. High variance, smoothness, good reconstruction and attractive clustering are different requirements.

1. **Global noncollapse does not identify useful liquid variation.** By the law of total covariance,

   `Cov(z) = E_c[Cov(z|c)] + Cov_c(E[z|c])`.

   Phase, temperature, material and source variation can dominate global statistics while within-liquid distinctions are weak or nuisance-dominated. A purely binary rank-one representation does not satisfy ideal full VICReg decorrelation; the concern is finite-weight tradeoffs and what the remaining dimensions encode. VICReg's variance/covariance terms prevent collapse, not automatically preserve a particular physical distinction. This is an inference about the objective, consistent with [VICReg's formulation](https://arxiv.org/abs/2105.04906).

2. **Positive pairs specify what may be erased.** Rigid rotation and atom reindexing should preserve an invariant state. Distinct neighbors and later snapshots need not have identical states. Strong equality across changing physical views can discard exactly the transient/interfacial structure of interest. Conditional JEPA prediction is different from pairwise invariance and should not be described as the same loss.

3. **Predicting trainable targets permits co-adaptation.** The neighborhood JEPA implementation differentiates both predicted and encoded target states. It can reduce predictive error partly by simplifying targets. Existing fixed physical/TDA/order anchors constrain this, but their aggregate weighting need not secure residual liquid information. The model already has present/future prediction and eight order anchors; simply recommending “add JEPA/q6” would repeat existing work. [Objective](../../../src/training_methods/neighborhood_jepa/v2/objective.py), [treatment](../../../experiments/neighborhood_jepa_regularization_20260920/README.md).

4. **Information and geometry must be tested separately.** A powerful decoder can recover an observable from dimensions that barely affect Euclidean/cosine neighbors. In the earlier matched MACE study, joint features had spatial cluster agreement above chance only0.0039, yet a spatially held-out linear probe recovered collective qbar6 withR²0.724. Information was present but poorly reflected by the chosen metric. This was one trajectory's spatial holdout, not independent-source validation. [Diagnosis](../../../experiments/mace_context_clusters_20260915/RESULTS.md).

5. **Selection optimizes a different goal.** Epi and the relaxed selection use development physical+.25TDA, not conditional liquid forecasting or stable physical neighbor relationships. The Epi liquid order MSE is0.6186 versus0.2654 in crystal; qbar6 reconstruction is strong while q4/w6 errors remain much larger. Aggregate reconstruction can improve without improving the states that matter for precursor dynamics. These are train-standardized errors, not automatic R² values.

The expanded relaxed study already tests within-temperature direct VICReg. It raises raw rank from1.17 (cold-control) to8.45 (cold-vic-temp01), while reported MLP12psAP is0.343 versus0.299 and NLL0.16138 versus0.16417. This single-seed reused-cohort result does not establish a significant ordering, but it rejects “more rank necessarily solves the problem.” [Completed study](../../relaxed_encoder/expanded-20260921/RESULTS.md).

## Measurements that distinguish these failures

Define evaluation masks independently of the encoder's own clusters. PTM-Other is a useful disordered mask but includes defects and interfaces. Report early liquid, noncrystalline regions far from a crystal, interface and crystal separately; repeat under declared qbar6/coherence thresholds. Never define “liquid” as whichever clusters make the projection look best.

| Requirement | Measurement | Essential control |
|---|---|---|
| Useful liquid variation survives | Within-temperature/disordered covariance spectrum, trace, participation ratio `(tr C)^2/tr(C^2)`; held-out linear and small nonlinear probes for angular order/topology | Trace and physical skill alongside rank; rank alone can reward noise |
| Distance represents physics | Physical-observable error or future-distribution error among nearest neighbors in original z; cluster-explained variance of qbar6/coherence; liquid-only trustworthiness of the display | Matched density/current order/temperature, shuffled features, SOAP/smooth-density baselines; own-label separation is circular |
| Robust to nuisance perturbations | Normalized feature drift under rotations, reindexing, precision changes, small thermal perturbations and cutoff membership changes | Check sensitivity to real bond rearrangements; original GeoFrame has documented tiny-perturbation instabilities |
| Spatially coherent | Embedding variogram versus physical separation; neighbor agreement adjusted for occupancies within liquid | Match overlapping receptive fields, spatial distance, phase and local density; don't equate all adjacent atoms |
| Temporally useful | Same-atom lagged drift relative to matched unrelated states; autocorrelation and neighbor survival; prospective skill at several lags | Real atom IDs, separate steady-state/transition intervals; a constant vector must fail information tests |
| Relevant to crystallization | Incremental held-out NLL/Brier/AP and calibration for persistent onset; future order/mobility/neighbor-retention skill | Baseline includes temperature, time, density, current order, crystal-neighbor fraction/interface distance; report misses with timing error |

For prediction, report `gain=(loss_baseline-loss_baseline_plus_z)/loss_baseline` on held-out sources. Fit all normalizers/readouts on training data, select on development, and bootstrap entire sources. Split before extracting overlapping windows. Keep several observable families entirely out of the training losses to test generalization rather than reconstruction of the targets being taught.

For the proposed C4/C6 intermediates, ask whether atoms enter these regions **before** crystalline order rises, whether the states are reproducible across sources, whether their future distributions differ at matched present order and interface distance, and whether these properties remain in original128D space. A band caused by varying crystal-neighbor fraction is useful interface information, but does not by itself establish a distinct precursor. Prestructured liquid has improved a nucleation coordinate in nickel, which motivates the question without proving it for Al: [Díaz Leines & Rogal](https://arxiv.org/abs/1810.04782).

The six relaxed snapshots can support structural comparisons. Their saved transitions match nearby sample centers, not persistent atom IDs; they are not a kinetic transition matrix. Use original MD trajectories/IDs/times for temporal evaluation, with relaxation as an observation transform if desired. A local position-only state does not determine the full system's future; stochastic propensity is a more realistic target than deterministic destiny. Multiple shooting futures permit propensity estimates; one trajectory does not establish a committor.

The repo already has a useful liquid-specific precedent:3,054 initially noncoherent centers, six held-out sources and eight shooting futures. Reported conditional future-topology MSE improvements were earlier GeoFrame+0.44%, corrected MACE VICReg+0.81%, versus SOAP PCA+4.79% and smooth-density MLP+7.16%[2.82,10.54]. These are different models/tasks, but make a strong benchmark template. [Liquid SRO study](../../../docs/liquid_sro_benchmark_20260905.md).

## A concrete training formulation to test

Let `z=f(x)` be the exported structural state, `c` the available thermodynamic condition, `phi(x)` fixed continuous physical observables, and `Y_tau` independently defined future order/mobility/onset targets. Fit a baseline `b(c, basic_structure)` on training data and define residual detail targets `r=phi-b`. The baseline must not use the physical quantity it is supposed to test, and nuisance conditioning should not inadvertently remove the actual signal of interest.

A proposed ablation objective is

`L = L_existing + lambda_detail L_liquid_residual + lambda_future L_liquid_future + lambda_metric L_physical_neighborhood`.

- `L_liquid_residual`: balanced within-temperature/disordered reconstruction of residual collective/angular order from the **exported** z. Existing physical heads remain; this changes which errors matter instead of adding another globally averaged head.
- `L_liquid_future`: proper predictive loss for fixed future observables at several lags, conditioned on present state/temperature. Compare against persistence and condition-only forecasts. Use distributional targets when repeated futures are available; MSE to a single future tends toward its conditional mean and cannot describe uncertainty by itself.
- `L_physical_neighborhood`: align local neighbor probabilities or selected pair distances in z with a declared physical/future-observable metric, restricted to condition-matched liquid examples. For example, minimize `sum_i KL(p_i^physical || q_i^z)` with train-fitted scales and finite bandwidths. This is explicit supervision of geometry; preserve held-out observables and source tests to detect imposing a preferred descriptor rather than discovering useful structure.

Use existing conditional variance/covariance regularization as a supporting constraint, not as the success criterion. Don't require128 independently varying physical axes or impose seven true liquid categories. Don't add all three terms at once: keep the same parent, data, support, export, budget and seeds; evaluate baseline, each term separately, then only the supported combination. A low-weight symmetry consistency loss is sensible; unconditional long-lag equality or spatial smoothing is not.

A fixed teacher/reservoir is an ablation, not a guarantee: it limits target co-adaptation but inherits its own biases. A two-part export (persistent structural coordinate plus residual local detail) is another later option if one bottleneck cannot retain both; declare which part defines distances and prevent a large decoder from making the structural coordinate irrelevant.

Temporal operator methods are useful baselines, not a promised fix. Short-lag TCCA/TAE has predicted longer-time bond-breaking propensity in a model glass former ([Qiu et al.](https://doi.org/10.1073/pnas.2427246122)), but nonlinear time-lagged reconstruction can mix slow and high-variance modes ([Chen et al.](https://arxiv.org/abs/1906.00325)). The repo's earlier matched VAMP test was already worse than static PCA on future-law retrieval0.47130 versus0.45715, with limited independent-source support. [Existing assessment](../../../docs/predictive_atlas_current_progress_20260903.md). Crystallization is nonstationary and a local patch need not be Markovian; equilibrium slow-mode assumptions need explicit checking.

## Smallest decisive next experiment

1. Recover multiple checkpoints from each actual training run; where early checkpoints were not retained, use a new matched run that saves them. Evaluate the same held-out liquid centers, feature spaces, preprocessing and physical radii. Include encoder and projector separately where both exist; inspect raw and clustering-transformed features.
2. On frozen embeddings, compare physical probes with physical/future neighbor retrieval. If probe skill survives while distance quality deteriorates, improve the deployed metric or the native distance-learning objective. If both decline, investigate positive pairs, target co-adaptation, weighting and bottleneck capacity.
3. Plot checkpoint curves of conditional physical skill, prospective gain, perturbation robustness and conditional spread. Test the claim of progressive erasure directly. Audit per-loss gradient norms/conflicts on liquid batches as supporting mechanism evidence.
4. Run the small ablation above and choose checkpoints using a preregistered development criterion centered on liquid physical/predictive usefulness, with nuisance-stability and information-retention constraints. Keep aggregate training loss and UMAP as diagnostics.

Success is a connected, reproducible physical coordinate that resolves different liquid structures/futures while staying stable to nuisance variation. It need not form isolated blobs, and a visually cleaner projection alone is not success.
