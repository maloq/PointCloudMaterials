# Results overview and comparison limits

[Handbook](README.md) · [Search all evidence](../../output/encoder_research/catalogue/index.html) ·
[Curated numeric CSV](../../output/encoder_research/catalogue/tables/headlines.csv) ·
[Full database](database.md)

This table summarizes the trajectory of the research, **not a ranking across
rows**. Each row has its own cohort, target, budget and metric. Numerical summaries
retain reported precision; the catalogue links full original tables and their
metric definitions. “Test” below may be a historically reused test split; it
should not be read as an untouched confirmatory test.

| Study / encoder family | Measured result | What we learned / limitation |
| --- | --- | --- |
| [GeoFrame FactorVAE archive](../../output/factor_vae_archive/README.md) | Three selected models, four restored static analyses; original metrics and galleries indexed | Static organization and representation diagnostics; the reanalysis is not another trained model |
| [GeoFrame temporal VICReg/VISReg](../geoframe_spatiotemporal_vicreg_20260905.md) | Al cosine silhouette: original0.7095, VICReg best0.7646, VISReg best0.6137; Davies–Bouldin original0.7106 versus VICReg best0.7955 | Mixed static criteria; projector smoothing differs from raw encoder behavior; within-trajectory validation, one seed |
| GeoFrame continuity audit, archived September5 | Selected-switch median jump reduction with held frames:99.9971% Al,99.9976% Mg/Ta | Identified canonical-frame discontinuity; targeted switch selection, not its prevalence or a predictive gain |
| [Task-trained continuous controls](../predictive_encoder_training_20260905.md) | Future-topology MSE reductions versus nonlinear coarse-order baseline: densityMLP7.05%, referenceMACE2.08%, GeoFrame3.11%; compactTDA mobility7.15% | Three seeds; target/readout-dependent strengths; no universal encoder winner |
| Archived temporal-hypothesis study | Predictive density / MACE: topology gains6.91% /3.43%, mobility14.55% /9.21%; GeoFrame screen negative | Coarse-augmented ridge, six test sources; not the preceding direct nonlinear assay; different screen/confirmation budgets |
| [Frozen MACE topology denoising](../../experiments/mace_al_denoising_20260910/README.md) | Balanced test MSE: single-frameMLP0.03751, atom-temporal0.02554, relaxed-input ridge0.00965 | History helps this relaxed descriptor; decoder/frozen-backbone experiment, not native temporal message passing |
| [Trainable relaxed VICReg24-fit comparison](../../experiments/mace_vicreg_relaxed_20260910/RESULTS_20260911.md) | Ridge balanced TDA MSE: snapshot0.032914, mean-history0.029632, atom-history0.030542; mean versus snapshot reduction9.69%, source95% CI1.19–15.43% | Three seeds; attention did not consistently beat mean pooling; combined-loss checkpoint selection confounded native heads |
| [Complete-context/dual readouts](../../experiments/mace_context_recovery_20260914/README.md) | Nonlinear instantaneous TDA: original mean80 0.017009, smooth-inner0.041357, frozen fusion0.040124; fusion q6-change reduction20.46% | Smoother support and center information have different tradeoffs; change readout sees both endpoints and is not a forecast |
| [Coordinate/velocity MACE](../../experiments/mace_velocity_20260915/README.md) | Both eight-epoch variants completed;304-channel physical/stability/intervention tables indexed under `velocity` | Separately supervised structural/activity/flow state; no forecasting objective in that comparison |
| [Frozen feature maps](../../experiments/mace_local_state_20260915/RESULTS.md) | InnerPCA16→physical10: reported0.75ps change0.5676→0.1712, instantaneous-TDA error0.0547→0.1397 | Clear smoothness/information tradeoff; approach discarded. Consecutive-motion44-fit study passed neither retention nor joint0.10 jump gate |
| [Causal MACE short pilot](../../output/predictive_memory/research-summary-20260917/RESULTS.md) | Low-order9ps MSE: positionA0.4607, velocityC0.5734, real-historyD0.5312, repeat0.5667 | Three seeds,1000updates; history helps weak velocity baseline but position-only remains better; none meet0.10 jump target |
| Same causal study, longer H100 | Width16: snapshot0.3987, history0.4043, repeat0.3963 at9ps | One seed,5000updates; source intervals do not establish history advantage |
| H200 causal width study, **reported only** | Width32 snapshot/history/repeat0.4028/0.3946/0.4056 at9ps; width16 0.4009/0.3990/0.3963 | Three seeds,17 low-order sources; modest width32 history benefit reported; width advantage and all-source benefit unresolved |
| [Predictive-memory H100/H200](../../output/predictive_memory/research-summary-20260917/RESULTS.md) | Joint path NLL width16/32: snapshot0.9247/0.9354; H12 0.9254/0.9369; H48 0.9256/0.9302; repeat0.9236/0.9405 | Two seeds,3000updates; larger width not supported at this budget; H200 raw results unavailable locally |
| [Memory stronger-present-loss follow-up](../../output/predictive_memory/research-summary-20260917-stopped/RESULTS.md) | Seed17 snapshot/H12/H48/repeat NLL0.94213/0.91071/0.90483/0.92719; stopped10/16fits | One completed stronger-loss seed favors history; its second-seed replication was not run; native prediction still leaves packet information unused |
| [Local descriptor onset forecasts](../../output/local_predictability/research-summary-20260917/RESULTS.md) | Dense-grid9ps AP snapshot/history/repeat0.2499/0.3508/0.2580; wider context0.4586 | Useful descriptor history/context signal; native-grid resampling changes these numbers, so keep grid identity |
| Same study, physical native backbones | Mean future MSE: MACE0.79993, GATr0.85640, current-packet ridge0.72644, H12packet0.71961 | One short matched-budget screen; fixed packet remains stronger; RTX repeat agreement is numerical verification, not a new seed |
| Same study, native onset states | MACE native snapshot/history9ps AP both0.0405; fresh nonlinear0.0772/0.0777 | Heads and state learning need diagnosis; current raw-atom recognition AP0.9940 does not imply onset skill |
| Same study, H200 **validation only, reported** | Snapshot/history/repeat future MSE0.75121/0.75060/0.75114 | Tiny validation differences; no locally received final test exports or intervals in the indexed summary |
| [Shared local MACE/GATr622](../../experiments/gatr_conditional_information_20260918/LOCAL_LAST622.md) | Conditional q6 improvement beyond radial controls: MACE21.55%, GATr−0.026%; qbar6 MACE46.59% | MACE retains angular information; smoother GATr is not thereby more predictive; checkpoint-specific assay |
| [Neighborhood JEPA and regularizer studies](../../experiments/neighborhood_jepa_regularization_20260920/README.md) | In larger matched relaxed-onset test, older EpiJEPA/SIGReg/VICReg MACE MLP AP0.1534/0.1433/0.1650 | These are historical-checkpoint external controls, not an isolated regularizer effect across every protocol; v1 confounds motivated v2 |
| [Expanded relaxed-input study](../../output/relaxed_encoder/expanded-20260921/RESULTS.md) |12ps MLP AP hot0.1751, cold0.3431, cold descriptor0.3680;117 positive event windows | Relaxed observations help; rank increases are not monotonically useful; one seed |
| [Larger relaxed fixed-grid assay](../../output/relaxed_encoder/large-test-20260921/RESULTS.md) |12ps MLP AP hot0.1419, cold0.2529, cold descriptor0.3911;11256windows,338 local onsets,30sources | Strong fixed-descriptor reference; score changes from expanded assay reflect a different origin grid/cohort, not regression of the same model |
| [BCR G1 and conditioning audit](../../experiments/bcr_followup_20260922/RESULTS.md) | Fresh-decoder noise MSE3.07% lower; original decoder correct-code gain versus optimized constant0.166%; exported radial RMSE31.5% worse on melt assay | Some useful conditioning but structural retention failure; all-liquid independent roots do not establish onset skill |
| [Structural-state v1](../../output/structural_state/screen-20260922/RESULTS.md) | Distance arm raw spread fell to about4.3% of initialization; refitted probes hid the shrinkage | Optimization failure, not an accepted smooth representation; motivates actual-head and fixed-scale audits |
| [Repaired structural-state v2](../../output/structural_state/repaired-review-20260923/README.md) |12ps nonlinear AP observed0.1521, relaxed0.2675, teacher0.1552, distance0.2668; relaxed−observed source CI[0.0081,0.1846] | One seed;643 at-risk windows/18 positives; descriptor advantage unresolved; probability calibration and ordinary future-MSE improvement not established |
| [Distance/future factorial](../../output/structural_state/future-metric-20260923/RESULTS.md) | Eight completed fits: no predeclared mechanism success; distance improves linear Brier≈0.43%; mean AP unchanged; all second-seed MLPs select step0 | Two seeds, reused small cohort; distance gain is small, future residual benefit does not replicate, probe/seed sensitivity matters |

## GeoFrame through 35 passes

The [completed reproduction and review](../../output/geoframe_evolution/epoch34-review-20260923/RESULTS.md)
retains 35 full passes, initialization and the archived epoch34 reference: 37
checkpoint evaluations, both exported feature stages, nine fixed material frames,
and the 45-root Al future assay. The original 160-epoch learning-rate clock was
preserved. This is one training seed and a transductive static assay; the original
recipe includes these snapshots. Spatially separated probe regions prevent local
patch overlap but do not make the encoder's training data independent.

| Mean within-liquid order R² | Encoder: 12 → 35 passes | Projector: 12 → 35 passes |
| --- | --- | --- |
| Al | 0.180 → 0.206 | 0.191 → 0.159 |
| Ta | 0.237 → 0.287 | 0.219 → 0.176 |
| Zr | 0.173 → 0.205 | 0.175 → 0.141 |

Al projector planar-fault AP rises 0.662 → 0.768, while its liquid participation
rank falls 2.41 → 2.06. One final Al177ps cluster contains about 98% of ordinary
liquid, 98% of five-fold proxies and 89% of ordered-liquid candidates. Decodable
information and unsupervised cluster separation are different requirements.
Nonbulk distance-matched boundary AUROC remains near chance. Small-displacement
smoothness improves without preserving projector liquid fidelity.

The final encoder's conditional 12ps onset AP is 0.203, but its Brier gain beyond
current physics has a paired-root interval spanning zero. Its 9ps conditional
future-order residual MSE is worse by 0.0391 (95% source interval 0.0215–0.0558).
The projector selects a constant-risk readout. These use current-order/geometry
conditions and must not be ranked against older temperature-only hazards.
Checkpoint correlations with AP do not establish correlations with calibration
or a causal benefit from optimizing structural fidelity.

The archived epoch34 recipe is VICReg, epoch159 VISReg; both include FactorVAE.
Their difference cannot establish a training-duration effect. Matched ablation
of FactorVAE and instantaneous physical targets remains a proposed experiment.

[Dense Ta/Zr examples](../../output/geoframe_evolution/structured-liquid-regions-20260923/README.md)
show connected ordered-liquid candidates far from accepted PTM crystal, with
boundary truncation recorded. These selected regions establish neither prevalence
nor future nucleation. Static generating potentials remain unknown, and accepted
Zr dynamics are unavailable. Most sampled Al planar faults are coherent twins;
other subtypes lack sufficient held-out examples. Original image colors are not
labels without the matching atom assignments.

## What we can carry forward

Continuous geometry removes a demonstrated GeoFrame failure mechanism, but it
does not automatically preserve useful liquid structure. Explicit present-state
measurements, initial-encoder controls and fixed descriptors repeatedly expose
problems that smoothness, total loss or rank alone miss. Relaxation and broader
physical context can help prediction, but they change the observations available
to the model. History benefits are conditional on protocol and budget; they are
not a universal property of the tested native encoders.

The latest simple geometry reconstruction is easier to audit than BCR's
geometry-conditioned decoder. The repaired one-seed onset result is promising,
while the subsequent factorial warns against treating it as a robust general
solution. No collected study establishes a sufficient Markov state, universal
liquid clusters, a committor, or simultaneous fulfillment of the original0.10
jump requirement and strong information retention.

## Where are the remaining numbers?

The catalogue imports the original CSV rows, report table lines and selected
result JSON across the registered local, WORK, STORE and archived collections.
This includes individual seeds, horizons, populations, native/frozen heads,
comparisons, failed controls and diagnostics omitted from this overview. Search
by family and path, then inspect the source metric contract before querying.
The curated CSV is intentionally a smaller interpreted subset. Its source quotes
are checked on every build. Both it and the database preserve the distinction
between a local report and a remote summary.

Hardware profiles are excluded from the scientific overview; use
[hardware documentation](../hardware_benchmark.md). Training counts are not
inferred by counting artifact rows. Missing historical definitions, oversized
index-only artifacts and duplicate exports are visible in the
[coverage record](../../output/encoder_research/catalogue/technical/coverage.json).

## Parameter-search evidence cut,23September

[Review of29 completed native checkpoints](../../output/encoder_research/parameter-search-20260923/RESULTS.md): historical VISReg159 has the strongest liquid-neighbor retrieval and nonbulk Al fault AP in this cut. This does not establish calibrated crystallization benefit. Stronger regularization can increase rank while worsening physical neighbors. A separate fit-normalized topology error avoids allowing tiny evaluation variances to dominate the old mean R². The resulting28-fit two-seed [parameter search](../encoder_parameter_search.md) tests these mechanisms with fixed final budgets.

### Matched35-pass interim findings

[Two-seed training update](../../output/encoder_research/parameter-search-20260923/INTERIM-20260923-2050.md): MLP VISReg improves raw liquid-neighbor error~14% versus matched VICReg in both seeds, with improved Al/Ta/Zr order readouts and liquid-context enrichment. One seed loses0.0257 Al fault AP, so the strict structural rule is not passed. Prediction remains unimproved. Turning FactorVAE off does not consistently help; direct-head/covariance comparisons are still running.
