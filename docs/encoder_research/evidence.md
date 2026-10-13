# Evidence: what the completed studies establish

This page is the canonical synthesis of the encoder, prediction and learned-dynamics research. Its supplied review summarizes repository evidence at [commit `05e3e8c`](https://github.com/maloq/PointCloudMaterials/tree/05e3e8ca57f42ea95f63f7f11c0aa05426fa5896), reviewed on 11 October 2026. The Ni section additionally incorporates completion receipts and [new diagnostic findings](#ni-methods-diagnostics) checked during integration on the same date. Original reports remain at their stable paths; the collapsed source lists preserve findings, corrections and unsuccessful branches without making every report a separate starting point.

**The strongest recurring distinction is between recognizing present structure and predicting a useful future distribution.** Static organization, readable descriptors, event classification, autonomous rollout and physical sensitivity are separate accomplishments. Different datasets, inputs, teachers, selectors and readouts prevent a universal ranking across the studies below. “Completed” describes the recorded scientific result; a launch or checkpoint receipt is not a final result or a live scheduler check.

Use [methods](methods.md) for present contracts and proposed method changes, [the experiment plan](experiment_plan.md) for proposed tests, and [literature](literature.md) for external methods. Proposals in those pages are not additional empirical findings. Historical temperature/time inputs, AP-trained models and discontinued reconstruction treatments retain their actual labels; the [current rules](../../AGENTS.md) do not rewrite their history.

<a id="motif-and-theory"></a>

## 1. What did motif discovery and early temporal theories establish?

The early motif work supplies a controlled structural benchmark and examples of organized representations. Its synthetic single-grain patches and known motifs do not establish discovery of nucleation precursors in unrestricted liquid trajectories. The GeoFrame archive likewise demonstrates static organization and interpretable neighborhood analyses, with an important qualification: several later static evaluations reuse frames within the encoder’s original training scope. Their held-out probe split does not make the encoder training independent of those frames. [Paper scope](../../paper/main.tex) · [Consolidated early results](results.md).

The September temporal studies show that success depends on what is measured. One spatiotemporal VICReg treatment improved Al cosine silhouette from 0.7095 to 0.7646 while another cluster-quality measure worsened. A targeted continuity reanalysis removed nearly all of the jump at selected frame switches; it did not estimate switch prevalence or demonstrate a forecasting gain. These are useful geometric diagnostics, not interchangeable measures of representation quality. [Spatiotemporal comparison](../geoframe_spatiotemporal_vicreg_20260905.md) · [Continuity and evaluation qualifications](results.md).

Predictive density encoders performed well on several declared future-descriptor readouts. However, a seemingly large GeoFrame improvement over persistence—49.49%—shrank to 0.14% against a condition-shuffled future control. Teacher-specific targets and strong conditional baselines changed the interpretation. The subsequent correction withdrew the broad “best replacement” conclusion: predictive training had not preserved the older model’s static organization, and data, layer, objective, selection and budget changed together. Historical baselines also included temperature. [Temporal hypotheses](../temporal_hypotheses_review_20260906.md) · [Required correction](../static_structure_vs_prediction_20260906.md).

The later 35-pass GeoFrame audit sharpened that lesson. Some continuous liquid readouts improved in the raw representation while deteriorating in the projector. A final Al cluster collected almost all ordinary-liquid and fivefold examples despite measurable continuous descriptor information. The selected prediction gain remained uncertain, and the longer run also changed the regularizer. Training duration, cluster appearance, decodability and future usefulness therefore cannot be inferred from one another. [Evolution protocol](../../experiments/geoframe_evolution_20260923/README.md) · [Completed evolution findings](results.md).

**Retained conclusion:** invariance, continuity, spatial separation and predictive information are distinct requirements. Early theoretical reviews remain useful hypothesis history; their proposed recipes and favorable headlines require the later corrections. Linear VAMP and conditional-future-law approaches were already tested in this repository, so citing them as wholly untried solutions would recycle an existing branch. [Representation review and correction](representation_literature_20260926.md).

<details>
<summary>Source reports and historical context</summary>

- [Early result synthesis, including targeted continuity and GeoFrame evolution](results.md).
- [Predictive encoder training](../predictive_encoder_training_20260905.md), [temporal hypotheses review](../temporal_hypotheses_review_20260906.md), and [static-versus-predictive correction](../static_structure_vs_prediction_20260906.md).
- [Spatiotemporal VICReg/VISReg](../geoframe_spatiotemporal_vicreg_20260905.md) and [GeoFrame evolution](../../experiments/geoframe_evolution_20260923/README.md).
- [Representation literature review](representation_literature_20260926.md): historical proposals, prior negative kinetic results and staged follow-ups.

</details>

<a id="representations"></a>

## 2. Do smoother representations retain useful local information?

The MACE context studies found a concrete source of artificial discontinuity: changing hard nearest-neighbor membership. Complete context and smooth inner/center summaries removed the tested membership jumps. That did not preserve every target equally. In a matched eight-epoch comparison, relaxed topological-descriptor error improved by 13.01%, while instantaneous topological error was 2.42 times the control’s. The hard-neighbor target could itself jump, and a center-local readout did not represent the entire patch’s topology. [Context results](../../experiments/mace_context_20260914/RESULTS.md).

Recovery and ridge audits limited the apparent benefit of auxiliary physical supervision. Center-plus-inner fusion recovered information lost by narrower summaries, but the matched instantaneous-topology gain from physical supervision was only 0.0174%, with an interval crossing zero. An earlier comparison also mixed a 128-dimensional projector with a 256-dimensional encoder export. Fresh, consistently tuned readouts made the competing checkpoints nearly equal; random MACE features were competitive under tuned ridge regression. This establishes sensitivity to layer and probe, not universal equivalence of random and trained encoders. [Recovery](../../experiments/mace_context_recovery_20260914/RESULTS.md) · [Readout audit](../../experiments/mace_tda_ridge_audit_20260914/README.md).

Several subsequent branches failed their own declared gates. None of 40 learned local-motion map candidates passed the retention or joint stability gate. Shared MACE/GATr pretraining experienced saturation and, for GATr, near-collapse under the initial high-learning-rate recipe. Smaller-rate restarts prevented that failure without proving a useful representation. Precision error could be small in absolute terms yet large relative to the surviving signal; a refitted readout and the originally trained head could consequently disagree. These findings concern the tested optimization and export paths, with bundled changes that prevent a single-cause diagnosis. [Motion-map results](../../experiments/mace_local_motion_20260916/RESULTS.md) · [Shared-pretraining study](../../experiments/shared_pretraining_20260918/README.md).

JEPA variants repaired normalization, amplitude-shrinkage and regularization problems, but the completed larger comparison did not establish a win over older VICReg or rich descriptor controls. The historical comparator called “geometry” actually included velocity and order channels, and some readouts appended temperature/time conditions. BCR similarly exposed a readout distinction: a fresh decoder improved about 3.07% over initialization, while the trained decoder gained only 0.166% from the correct code over an optimized constant code; exported radial reconstruction was worse. A nonconstant code or successful reconstruction fit did not establish useful transferable information. [JEPA protocol](../../experiments/neighborhood_jepa_20260920/README.md) · [Completed JEPA synthesis](results.md) · [Input audit](prediction_context.md) · [BCR follow-up](../../experiments/bcr_followup_20260922/RESULTS.md).

The strongest conditional-information readouts in this historical branch came from MACE, while the corresponding GATr export added little beyond radial controls. Neither demonstrated practical additional onset information after controlling present bond/angular order and redundant radial inputs. Later expanded pretraining changed both data and optimization budget, so it did not isolate a data-scale effect. BCR’s high-temperature development roots and one fitted seed likewise did not supply independent undercooled-liquid or crystallization confirmation. Those scope limits belong alongside the retained negative results. [Conditional-information comparison](../../experiments/gatr_conditional_information_20260918/LOCAL_LAST622.md) · [Expanded MACE treatment](../shared_pretraining_mace_expanded_dual_20260919.md) · [BCR protocol](../../experiments/bcr_g1_20260921/README.md).

The structural-state and supervised-onset branches add negative evidence rather than a new preferred recipe. An amplitude-collapse repair produced a nonconstant representation, but the later factorial did not meet its declared mechanism-success criteria. On the historical AP-optimized onset task, a fixed relaxed-descriptor MLP outperformed the selected jointly trained MACE model at both reported horizons. These treatments used their recorded conditions and selection rules. Physical-reconstruction pretraining, AP optimization and explicit temperature/time inputs are now discontinued for new comparisons. [Structural-state history](../../experiments/structural_state_20260922/README.md) · [Completed factorial synthesis](results.md) · [Supervised onset](../../experiments/supervised_onset_20260924/RESULTS.md) · [Current branches](training_branches.md).

<details>
<summary>Source reports, failed branches and retirement records</summary>

- [MACE context](../../experiments/mace_context_20260914/RESULTS.md), [context recovery](../../experiments/mace_context_recovery_20260914/RESULTS.md), [TDA/ridge audit](../../experiments/mace_tda_ridge_audit_20260914/README.md), and [local-motion maps](../../experiments/mace_local_motion_20260916/RESULTS.md).
- [Shared-pretraining overview](../../experiments/shared_pretraining_20260918/README.md), [initial diagnosis](../../output/shared_pretraining/diagnosis-20260918/RESULTS.md), [restart diagnosis](../../output/shared_pretraining/restart-diagnosis-20260918/RESULTS.md), and [precision/stability audit](../../output/shared_pretraining/stability-20260918/RESULTS.md).
- [GATr conditional information](../../experiments/gatr_conditional_information_20260918/LOCAL_LAST622.md), [JEPA](../../experiments/neighborhood_jepa_20260920/README.md), [BCR geometry variant](../../experiments/bcr_g1_20260921/README.md), and [BCR follow-up](../../experiments/bcr_followup_20260922/RESULTS.md).
- [Structural state](../../experiments/structural_state_20260922/README.md), [structural future state](../../experiments/structural_state_future_20260923/README.md), and [historical supervised onset](../../experiments/supervised_onset_20260924/RESULTS.md).
- [Discarded frozen maps](../discarded_frozen_encoder_maps.md) and [architecture retirement](../architecture_retirement.md) distinguish preserved evidence from live implementations.

</details>

<a id="memory-and-context"></a>

## 3. When do history and spatial context improve forecasting?

The embedding-forecast studies established history averaging as a necessary control. Autoregressive decoding improved path error over an anchor-only baseline, but its advantage over a mean-history control was less secure. In the matched context sweep, six picoseconds of history captured about 89% of the autoregressive anchor-to-24-ps improvement, with diminishing returns around 6–12 ps. The autoregressive decoder had more parameters, and repeated use of a small held-out source set limits a universal memory-timescale claim. [Embedding forecasts](../../experiments/embedding_forecast_20260911/RESULTS.md) · [Autoregressive comparison](../../experiments/embedding_forecast_20260911/AUTOREGRESSIVE_RESULTS.md) · [History-length sweep](../../experiments/forecast_context_20260913/RESULTS.md).

Spatial context also helped on the historical local-onset task. Twelve picoseconds of history with 32 neighboring centers gave the best observed setting; expanding to 128 or 512 centers did not improve F1. Its 9-ps-window F1 was 0.5686, but only 16.12% of events were detected with timing accurate to 1.5 ps when evaluated exactly 9 ps before onset. The much higher overall state accuracy was dominated by easier already-crystalline or liquid states. The task used three consecutive crystalline center observations on the historical 0.75-ps grid; it was not a nucleus-birth assay. [Spatial-context results](../../experiments/forecast_spatial_mixture_20260913/SPATIAL_CONTEXT_RESULTS.md).

Later matched follow-ups prevent turning these results into “more context always helps.” A relaxed-reference treatment improved integrated Brier score by 13.3% over the original reference on matched windows, but both checkpoint and input domain/support changed. Extra history and a second original encoder worsened their repeat controls. Separate local-predictability work found useful history/context gains on a dense grid, while absolute values changed under the native-grid evaluation and current physical packets remained strong future-descriptor controls. [Crystallization follow-up](../../experiments/crystallization_followup_20260922/RESULTS.md) · [Local-predictability findings](../../output/local_predictability/research-summary-20260917/RESULTS.md).

The predictive-memory branch stopped after ten of sixteen fits. One completed seed favored its stronger present-information loss, but the intended second-seed confirmation was absent. That result remains suggestive. Taken together, completed studies support task-specific gains from history and nearby context, with mean-history, present-descriptor and repeat controls essential to the claim. They do not establish a universally sufficient observation window or a transferable nucleation precursor. The original 150 Al production histories are now deleted; their conclusions and ancestry survive, but they are not an available raw-data benchmark for a new queue. [Stopped branch](../../output/predictive_memory/research-summary-20260917-stopped/RESULTS.md) · [Retirement record](../storage/al075_removal_20261007/README.md).

<details>
<summary>Forecasting and memory source reports</summary>

- [Relaxed VICReg history comparison](../../experiments/mace_vicreg_relaxed_20260910/RESULTS_20260911.md).
- [Embedding forecast](../../experiments/embedding_forecast_20260911/RESULTS.md), [autoregressive results](../../experiments/embedding_forecast_20260911/AUTOREGRESSIVE_RESULTS.md), [history-length sweep](../../experiments/forecast_context_20260913/RESULTS.md), and [spatial-context sweep](../../experiments/forecast_spatial_mixture_20260913/SPATIAL_CONTEXT_RESULTS.md).
- [Local-predictability synthesis](../../output/local_predictability/research-summary-20260917/RESULTS.md), [stopped predictive-memory synthesis](../../output/predictive_memory/research-summary-20260917-stopped/RESULTS.md), and [matched crystallization follow-up](../../experiments/crystallization_followup_20260922/RESULTS.md).

</details>

<a id="alignment-and-partitions"></a>

## 4. How do alignment, readout and partition change structural usefulness?

The completed encoder-mechanism study provides comparatively strong matched evidence: 18 fits, three seeds and a common evaluation collection. Alignment improved Epi’s calibrated 6-ps predictive loss from 0.07496 to 0.06843, with a paired source-bootstrap difference of −0.006536 and interval [−0.008522, −0.004728]. Physical-neighbor and reference-boundary diagnostics worsened across the same seeds. Predictive improvement therefore coexisted with losses on different structural criteria. Aligned VICReg and aligned Epi had an uncertain calibrated-loss difference; the result does not support an unconditional winner. [Completed mechanism comparison](../../experiments/encoder_mechanisms_20260926/COMPLETED-RESULTS.md).

Frozen Epi with a trained head outperformed scratch training on that predictive assay; fine-tuning did not establish a further predictive gain. Yet fine-tuning improved boundary AUC in each seed and left liquid reconstruction error similar. “Fine-tuning erased structure” is too broad. Physical-descriptor add-back still helped some readouts, which identifies limitations of those fitted probes, not a measured mutual-information deficit. The confidence intervals resample 30 sources after averaging three fixed seeds; they do not cover a population of training seeds or every exploratory comparison. [Mechanism results and limits](../../experiments/encoder_mechanisms_20260926/COMPLETED-RESULTS.md).

The spatial VICReg study separates continuous information from a global partition. Across nine models, 99.96–100% of strictly clear liquid examples fell into one global seven-cluster cell, even though continuous readouts retained useful geometry and topology. The same-center treatment achieved 15.24% liquid topological skill against its training-mean reference, versus zero for the global seven-cluster membership. Neighbor-positive training traded some topological skill for geometric skill. All seven cells were occupied in the full population, so this was not global representation collapse. Neither a one-cluster liquid picture nor low projector rank proved information absence; changing the partition was an unrun follow-up. [Completed partition audit](../../experiments/spatial_vicreg_bias_20260929/RESULTS.md).

The interface evidence adds a scope boundary. Wide-context predictors often saw an existing crystal already inside their inputs: 468 of 480 reported alarms in the vector model did. In a stricter clear-liquid stratum with an external crystal, the completed interface model’s distance RMSE was 15.227 Å against 10.711 Å for the matched training-mean control. This supports failure of that model on that localization assay, not proof that invisible-interface information is impossible. Its dominant feature direction was strongly associated with phase in both train and test; destroying a useful direction in a frozen head does not establish spurious overfitting. Later liquid adaptation has an interim overfitting diagnosis and a missing-early-checkpoint limitation, not a completed remedy. [Visibility audit](../../experiments/spatial_distance_20260926/RESULTS.md) · [Feature/clear-liquid audit](../../experiments/crystal_interface_20260928/FEATURE_DOMINANCE.md) · [Adaptation record](../../experiments/crystal_interface_20260928/LIQUID_TRAINING_DRIFT.md).

<details>
<summary>Alignment, partition and interface reports</summary>

- [Initial quality screen](../../experiments/encoder_quality_20260926/RESULTS.md) and [completed matched mechanisms](../../experiments/encoder_mechanisms_20260926/COMPLETED-RESULTS.md).
- [Spatial VICReg results](../../experiments/spatial_vicreg_bias_20260929/RESULTS.md), [coordinate findings](../../experiments/spatial_vicreg_bias_20260929/COORDINATE_FINDINGS.md), and [interface correspondence](../../experiments/spatial_vicreg_bias_20260929/INTERFACE_CORRESPONDENCE.md).
- [Spatial approach](../../experiments/spatial_approach_20260926/RESULTS.md), [distance/visibility](../../experiments/spatial_distance_20260926/RESULTS.md), [feature dominance](../../experiments/crystal_interface_20260928/FEATURE_DOMINANCE.md), and [liquid-training drift](../../experiments/crystal_interface_20260928/LIQUID_TRAINING_DRIFT.md).

</details>

<a id="birth-and-fate"></a>

## 5. What is being predicted: arrival, birth, appearance or establishment?

These targets answer different questions and select different populations:

| Assay | Population and endpoint |
|---|---|
| Historical local onset | A tracked center becomes crystalline, commonly through arrival of existing crystal. |
| Historical screened preappearance | Retrospectively selected establishment sites with crystal-free histories versus matched persistent-liquid controls. |
| Catalogue birth localization | Future founder sites versus same-source/time random nonbirth centers at 1/3/6-ps leads; no added neighborhood-order screen. |
| Embryo establishment | Present 8–63-atom embryos reaching a persistent 64-atom interval; the recorded cohort additionally requires future-resolved eligibility. |
| Next local appearance | Presently clear, fixed-time local observations gaining any crystalline consumed atom at 0.1/0.5/1/3/6 ps; the separately fitted persistent-appearance target is 3 ps. |

[Historical protocol](../../experiments/birth_prediction_20260930/README.md) · [Catalogue/localization contract](../structural_response.md) · [Embryo metric](../metrics/embryo_establishment_cohort.md) · [Appearance metric](../metrics/motion_first_appearance.md).

The origin audit found that 98.50% of positive Al64 onset windows at both 3 and 6 ps represented arrival of established crystal; its original grid contained no regional-birth training windows. Earlier onset results cannot be relabeled as birth prediction. The later fate audit found 28 conservative failed-embryo candidates across 24 original sources. Of 95 retained original births, 87 showed the declared post-establishment growth and eight merged before confirmation; none had confirmed dissolution. Those conservative definitions and finite follow-up do not establish a critical nucleus size or indefinite survival. [Origin results](../../experiments/crystallization_origin_20260925/RESULTS.md) · [Completed fate audit](../../experiments/nucleus_fates_20261002/README.md).

Eight Al fast-v4 motion fits and their birth readouts are complete. At a 3-ps lead, a single present neighbor-averaged q6 scalar achieved AP 0.940, compared with 0.918–0.931 for the learned structural-plus-motion readouts. Zeroing motion changed AP by at most 0.0044; shuffling structure within source/frame-matched sets reduced it to 0.189–0.199. Present structure dominated these fitted readouts. At that lead, 84% of positives already contained a PTM component of at least four atoms, versus 3.4% of controls. Exact input reconstruction found no future-coordinate or train/test-ancestor overlap. High AP therefore reflects a strong present-order distinction in this retrospective, fixed-20%-prevalence population; it does not measure population nucleation probability. [Completed structural-response audit](../../experiments/structural_response_20261007/README.md).

The embryo collection is recorded complete at 34,060 observations, while the first-appearance comparison has launch/progress records without a final new-model comparison in the reviewed report. Collection is not predictive success. Future-resolved embryo eligibility, delayed persistence confirmation, appearance aliases and ancestry-dependent weighting are interpretation questions addressed in the methods and proposed analyses; the audit has not demonstrated that they actually bias a reported result. An onset-by-horizon target confirmed after persistence must also remain distinct from a confirmation-time endpoint. [Collection record](../continuous_motion.md) · [Embryo definitions](../metrics/embryo_establishment_cohort.md) · [Appearance definitions](../metrics/motion_first_appearance.md) · [Proposed discriminating analyses](experiment_plan.md).

The older nested-shooting pilot resolved 131 of 144 trials, with only two of 28 transition configurations showing mixed outcomes; a prior-probability control won its Brier comparison. Sparse replication and only four final transition parents limit conclusions about momenta or useful committors. Finite-horizon establishment probabilities and state-conditioned, kernel-specific shooting probabilities remain separate quantities. New birth work uses the pinned consolidated `010` catalogue or separately native-only `030`; pre-birth tails annotate existing events. The October 8 `010` publication records 129 source views, 110 ancestry groups and 312 established PTM records. Earlier studies keep their own fingerprints and eligible cases; these three counts do not measure the same sampling unit. No new neighborhood-order filter is authorized for the canonical catalogue protocol. [Nested pilot](../nested_committor_pilot_20260902.md) · [Catalogue and ancestry contract](../datasets/nuclei_birth.md).

<details>
<summary>Birth, fate and current cohort sources</summary>

- [Origin audit](../../experiments/crystallization_origin_20260925/RESULTS.md), [historical birth protocol](../../experiments/birth_prediction_20260930/README.md), [Al fate audit](../../experiments/nucleus_fates_20261002/README.md), [Ta extension](../../experiments/nucleus_fates_20261002/TA.md), and [shooting extension](../../experiments/nucleus_fates_20261002/SHOOTING.md).
- [Fast-v4 findings and appearance progress](../../experiments/structural_response_20261007/README.md), [embryo collection](../continuous_motion.md), [embryo metrics](../metrics/embryo_establishment_cohort.md), and [all appearance targets](../metrics/motion_first_appearance.md).
- [Nested committor pilot](../nested_committor_pilot_20260902.md), [rate-calibration protocol](../metrics/nucleus_rate_calibration.md), [catalogue contract](../datasets/nuclei_birth.md), and [Al pool coverage](../al_research_pool.md).

</details>

<a id="simulator-response"></a>

## 6. Do simulator responses improve learned futures?

There is a clear positive result in a narrow controlled setting and a negative matched result on local Al observations. The full-cell study used perturbed FCC Al256, a MACE teacher, 20/100-fs targets and three seeds. With eight value observations per configuration, adding physical response supervision reduced held-out feature MSE from 0.033864 to 0.008677 and response MSE from 0.522076 to 0.148839. Thirty-two value observations did not recover this advantage. Response training cost substantially more, and most selected checkpoints were near the training limit. This establishes a benefit under that recipe, without an equal-compute or converged-training claim. [Completed atomistic experiment](../../experiments/response_atlas_20261001/ATOMISTIC_TRAINING.md).

The local study used 135 source queries, four treatments and three seeds, with local 80-atom observations and a larger moving environment for the physical teacher. Response supervision worsened mean feature and response errors by 18.15% and 10.69%; one seed improved and two worsened. More value-training time did not resolve the gap. The teacher replaced the parent trajectories’ MEAM physics with MACE, and an environment-radius convergence check did not establish equivalence to the original full periodic system. These are substantial domain and conditioning differences from the Al256 success. [Completed local comparison](../../experiments/local_response_20261002/README.md).

Later coordinate/velocity, disturbance, precision and dense-target work clarifies what derivatives were computed and whether the numerical comparison is meaningful. Such audits are prerequisites, not evidence of improved future prediction. A full-state physical response need not equal the derivative of a conditional future law given only the learner’s local observations; short fixed-box teachers also differ from long native futures. The completed results support testing those distinctions, not claiming that a proposed response factorization or a new loss has solved them. Current GPU MEAM precision rules and the Al256-specific fast profile remain separate protocols. [Disturbance audit](../../experiments/local_response_20261002/DISTURBANCE_AUDIT.md) · [Velocity-response protocol](../../experiments/response_velocity_20261004/README.md) · [Current methods](methods.md).

<details>
<summary>Response evidence and numerical/protocol history</summary>

- [Al256 completed training](../../experiments/response_atlas_20261001/ATOMISTIC_TRAINING.md) and [local Al completed comparison](../../experiments/local_response_20261002/README.md).
- [Disturbance audit](../../experiments/local_response_20261002/DISTURBANCE_AUDIT.md), [coordinate-response protocol](../../experiments/response_geometry_20261003/README.md), [velocity-response protocol](../../experiments/response_velocity_20261004/README.md), and [structural-response findings](../../experiments/structural_response_20261007/README.md).

</details>

<a id="nickel-dynamics"></a>

## 7. Can compact Ni states support autonomous dynamics?

The matched original/relaxed Ni comparison completed its motion fits and readouts. Relaxed observations helped some event-specific comparisons; physical-response supervision was mixed, and adding motion did not consistently improve structure-only readouts. These compared whole input/checkpoint treatments. The quench audit subsequently made identical-input quenches repeatable with sorted neighbors on tested cells, while tiny input changes still reached different minima. Historical relaxed targets retain their original recipe; repeatability does not establish differentiability through relaxation. [Ni response comparison](../../experiments/ni_response_20261009/README.md) · [Quench audit](../../experiments/ni_quench_repeatability_20261009/README.md).

The first compact pilot completed 16 fits crossing code width, one/four observations and value/response supervision, with two seeds. Frozen unrelaxed Small eSEN features, actual velocity features and descriptors supported useful event-matched nucleus readouts. But one-picosecond structural MSE improved only about 6–7% over a training-mean control; six-picosecond autonomous rollouts showed severe seed-dependent failures. Physical sensitivities did not beat the zero-response reference. Good classification and likelihood near observed futures therefore did not validate the full generated rollout distribution. Empirical microstate retrieval was also not an atomic decoder or a nucleation-rate validation. [Completed compact pilot](../../experiments/ni_latent_sim_20261010/README.md).

The subsequent completed factorial compared physical-response supervision, learned-map Jacobian control and 1/6-ps training horizons. Jacobian control modestly improved extrapolation from 1-ps training; training through 6 ps gave the more consistent gain. The 6-ps-trained value control reached structural MSE 0.95384 against 1.00168 for the training mean. Every treatment-average physical-response error remained above the 0.500 zero-response baseline. The original extreme divergences disappeared in every new arm, including controls, while random within-window starts changed to fixed starts. Their disappearance cannot be attributed solely to the Jacobian penalty. [Completed stability factorial](../../experiments/ni_latent_sim_20261010/DERIVATIVE_STABILITY.md).

The sealed expanded collection contains 14,400 sequences split equally among original, persistent-liquid and advancing-front regimes. These reuse the original 60 ancestors; future-selected regimes are not natural occupancy samples, and only one inner-selection ancestor supplies front examples. Completion receipts now cover all six dense-path MLP fits and their evaluations, the two-fit transformer extension and four-model architecture comparison, and all eight overnight transformer fits and evaluations. Actual batch migrations remain part of the interpretation: the overnight 1-ps response fits used 1,000 optimizer updates and 1,024,000 examples; the two 1-ps value fits used 3,925 and 4,000 updates with the same example exposure. The 6-ps fits used 4,000 updates. Equal exposure does not make optimization equivalent. The planned 36,608-row expansion was not the launched campaign. Completion alone does not resolve physical sensitivity. [Expanded regimes](../../experiments/ni_latent_sim_20261010/REGIMES3X.md) · [Dense-path protocol](../../experiments/ni_latent_sim_20261010/DENSE_PATH.md) · [Execution and batch history](../ni_latent_sim.md).

<a id="ni-methods-diagnostics"></a>

### Completed methods diagnostics, October 11

The [first diagnostic revision](../../output/ni_latent_sim/methods-diagnostics-20261011-v1/README.md) freezes three matched MLP models: endpoint-response, path-values and path-response, each trained for 3,200 updates with seed 20261011 and selected by the original validation NLL. Their native training rollout covers 1 ps; evaluation through 6 ps includes extrapolation. This is a one-training-seed comparison with three Monte Carlo keys, not three training replications. Path evaluation uses 214 rows from all 40 held-out ancestors; the response audit re-encodes one existing physical anchor from each of 60 ancestors and scores students on the 40 held-out anchors. The history assay retains all 14,400 rows and original ancestor roles. All three jobs completed in their first attempt. [Protocol](../../experiments/ni_latent_sim_20261010/METHODS_DIAGNOSTICS.md) · [Frozen metric definitions](../metrics/ni_motion_diagnostics.md).

**Physical response remains unresolved.** At K=128 and epsilon=.5, pooled standardized response MSE divided by the zero-response MSE is:

| Frozen model | Initial-position direction | Initial-velocity direction |
| --- | ---: | ---: |
| Endpoint response | 1.079 | 1.013 |
| Path values | 1.066 | 1.020 |
| Path response | 1.069 | 1.011 |

Values below one would improve on predicting zero; none does in this pooled point comparison. The reference averages all 20 physical times, all held-out anchors and feature coordinates. Estimated two-shot mean noise accounts for 3.41% and 4.51% of Q/V response power, respectively; two shots limit that estimate. Reducing the student epsilon from .5 to .125 changes total response MSE by at most 0.43%. This does not validate physical derivatives against finite differences or prove compatibility between the full-state intervention and the permitted local observation. Per-model errors are in the run's `analyses/response-loss-*/tables/metrics.csv`; the all-time reference is calculated from [retained shot packets](../../output/ni_latent_sim/methods-diagnostics-20261011-v1/analyses/response-teachers/technical/packets/), since the exported teacher table lists only four times.

**The finite-K mean-loss penalty is small at these checkpoints.** At K=4, variance/K is 0.0206–0.0379% of Q error, 0.0045–0.0099% of V error, and 0.081–0.097% of value error across the models. The mathematical penalty exists, but its measured size does not make changing to a U-statistic the primary repair. Its causal influence during optimization was not tested. The sampled training normalizer hits its floor at the first two V times (10% of times); Q does not reach the floor. [Normalizer table](../../output/ni_latent_sim/methods-diagnostics-20261011-v1/analyses/response-normalizers/tables/metrics.csv).

**Path-response does not improve the emitted-path score.** For K=128, the ancestor-weighted path-energy ranges over three sampling keys are:

| Forecast | Full 6-ps emitted path energy, lower better |
| --- | ---: |
| Endpoint response | 0.593476–0.594038 |
| Path values | 0.600564–0.601437 |
| Path response | 0.627503–0.628526 |
| Persistence | 1.125225 |

The path-response model has a positive path scrambling gap, but post-hoc inspection of retained conditional-mean paths finds maxima of 1,604–4,229 training standard deviations; the largest observed target is 8.34. Approximately 0.88–0.99% of its particle paths exceed magnitude 100, counting paths equally across the 214 rows rather than applying ancestor weights. Almost all of its positive scrambling-gap contribution comes from rows containing such runaway paths. These are additive contribution diagnostics on retained rows, not a filtered benchmark or a preregistered threshold test. They prevent interpreting the positive gap as validated useful dynamics. All outputs were finite, showing that finiteness alone did not establish stability. Original path metrics, row scores and trajectories remain under the run's `analyses/paths-*/` directories. The same audit on retained 6-ps-trained models is needed to separate training-horizon extrapolation from the response treatment.

**History helps some readouts, with little separation between the compact codes.** At a common shifted forecast origin, ancestor-weighted smooth24 MSE at +1 ps is:

| Representation | 1 observation | 4 | 8 | 14 |
| --- | ---: | ---: | ---: | ---: |
| Raw280 | 0.711253 | 0.685262 | 0.677504 | 0.676094 |
| Path-response compact16 | 0.682817 | 0.675949 | 0.677370 | 0.681010 |

Other compact16 models are close. Unordered past summaries compete with ordered histories; fixed-model past reversal harms short-offset prediction more than the 3-ps readout. These are Gaussian ridge results with different input widths and selection-chosen penalties, not a demonstration of optimal memory or autonomous closure. Longer raw history helping where ordered compact history saturates motivates an observation/compression/readout check before scaling the transition architecture. [All history scores and weighting panels](../../output/ni_latent_sim/methods-diagnostics-20261011-v1/analyses/history-comparison/tables/metrics.csv).

### Follow-up interpretation: a failed readout is not an information bound

The [October 12 methods handoff](../../experiments/ni_latent_sim_20261010/METHODS_HANDOFF_20261012.md) records the completed transformer and response-predictability follow-up. Native 6-ps training gives a modest matched improvement over 1-ps training in one seed; adding physical derivatives has opposite effects in the two seeds. The response readouts use all 300 physical anchors (80 fitting, 20 selection, 200 held out), with a state-conditioned kernel as well as a directional linear readout.

The crucial positive control is present-response recovery. Copying the measured present geometry secant gives relative position-response error about 0.000006 at the present and 0.0234 at 5 fs. The fitted full-observation state-conditioned readout gives about 0.963 at the present and 0.9615 at 5 fs. Its input contains useful response information that this fitting procedure fails to recover. Therefore its weak future scores do not establish an information-theoretic limit, fundamental compression failure or impossibility of motion learning. Persistence becomes poor by 100 fs (about 2.77), so the successful control is not itself a future-response model. Raw280 denotes encoded observations, not raw atomic Q/V. These numbers concern the latent branch's geometry128 target and its own metric definition.

<a id="direct-measurement-followup"></a>

### Direct measurements: all six fits completed

Checked 12 October 2026 against per-fit training/evaluation receipts in [direct-20261011-v1](../../output/measurement_sim/direct-20261011-v1/README.md). All six models completed 1,600 updates and their 9,600-row evaluations; both queue workers have exited. The two-GPU queue took about 2.76 hours from launch through its final completion receipt. This timing is specific to the recorded models and hardware.

This branch evolves 244 BO/TDA/density/velocity measurements directly, without an eSEN compressor or learned observation decoder. Its state is anchored exactly at the present measurement. The table gives arithmetic means across two training seeds of the already exported ancestor-balanced scores at 6 ps; the 9,600 rows cover 40 held-out ancestors. It is a descriptive two-seed summary, not additional independent evidence or a new metric revision.

| Treatment | Endpoint MSE | Whole-path energy | Increment energy | Lag-product energy | Nominal 90% path coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Marginal | 0.822001 | 0.627142 | 0.730044 | 0.234511 | 0.816163 |
| Connected | 0.817852 | 0.625344 | 0.729008 | 0.230391 | 0.824120 |
| Connected + response | 0.817651 | 0.625070 | 0.729296 | 0.231209 | 0.826814 |

Connected objectives give small improvements over marginal-only training in both seeds. Their increment and lag-product scores still do not beat the recorded diagonal-OU baseline (approximately 0.7254 and 0.2205). The derivative treatment gives no consistent added endpoint advantage and slightly worsens increment/lag-product scores in both seeds. Roughly matching overall path variance does not establish the correct temporal dependence; coverage remains below its nominal level.

At 100 fs, the arithmetic mean of the eight exported group/direction relative response errors is 1.0434/1.0444 for connected and 1.0570/1.0507 for connected+response across the two seeds. Zero response scores 1 in these cells. This aggregation is explicitly an average of group/direction ratios, unlike the latent audit's pooled error-to-power ratio. Source-paired differences are retained in [paired sensitivity](../../output/measurement_sim/direct-20261011-v1/analyses/comparison/tables/paired_sensitivity.csv). Some velocity-direction groups improve, while position-direction groups worsen; an overall successful sensitivity learner is not demonstrated.

These results weaken a claim that the learned spatial encoder or 16-dimensional bottleneck is the sole explanation of response failure. They do not prove that fixed measurements form a closed dynamical state, that the teacher matches the native conditional forecast, or that the response is unlearnable. Keep the target and normalization differences explicit: direct244 scores cannot rank these models against latent280 models. See [metric definitions](../metrics/measurement_sim.md) and the [October 12 continuation](experiment_plan.md#ni-followup-20261012).

<a id="measurement-overnight-20261012"></a>

### Overnight memory and matched physical response: all 15 fits completed

Checked 12 October at11:39 CEST against all15 training/evaluation completion receipts
in [overnight-20261012-v1](../../output/measurement_sim/overnight-20261012-v1).
The final evaluation completed at08:15 CEST, 7.16 hours after launch. The queued
continuation1034820 ran successfully; no work from this queue remains running or
pending. CPU job1034819 also completed78 response readouts and three history readouts.
This is completion of the declared overnight queue, not all proposed E0–E9 work.

The following values are arithmetic means of the existing exported scores across
three training seeds, without redefining the frozen metrics or creating additional
independent observations. Native evaluation uses9,600 rows from40 ancestors; all
three memory treatments share the shifted forecast origin and5-ps future. Physical
evaluation uses200 anchors from40 ancestors, with two real shots per anchor and
three neural noise keys. See [metric contract](../metrics/measurement_followup.md).

| Native treatment at5 ps | Endpoint MSE | Marginal NLL | Whole-path energy | Nominal90% particle coverage |
| --- | ---: | ---: | ---: | ---: |
| Ordered4 observations | 0.817299 | 1.368404 | 0.623436 | 0.820177 |
| Ordered14 observations | 0.809372 | 1.361526 | 0.620745 | 0.811547 |
| Present + past13 mean/std | 0.806173 | 1.349414 | 0.619763 | 0.811717 |

The summary treatment improves endpoint MSE by1.36% over history4 and improves
MSE/NLL/path score in allthree paired seeds. Ordered history14 is less consistent:
its endpoint MSE improves in two seeds and worsens in the third. The summary
treatment's advantage suggests useful averaging/history statistics; it does not
establish successful learning of ordered long-time dynamics. Source-paired
intervals are in [paired predictions](../../output/measurement_sim/overnight-20261012-v1/analyses/comparison/tables/paired_predictions.csv).

Native temporal dependence remains a problem. For the summary treatment,
independently shuffling particle identities between times preserves every marginal
ensemble but improves increment energy from0.726086 to0.715539 and lag-product
energy from0.238146 to0.199090. Whole-path energy barely changes (0.619763 to0.619854).
Its individual-path temporal power is0.708469 versus MD0.712690; matching this
overall variance has not produced correct temporal correlations. The empirical
90% particle bands cover about81% of observations. Mean trajectories having much
less variance is not by itself evidence of collapse: conditional means of a
stochastic process can be smooth.

The separate Gaussian ridge history readout reaches5-ps MSE0.781340 with past
mean/std, compared with0.787123 for ordered4 and0.788200 for ordered14. These are
point predictions, while neural MSE uses a finite32-particle estimate of its mean;
do not interpret the difference as a Monte Carlo-corrected comparison of exact
conditional means. [Readout table](../../output/measurement_sim/overnight-20261012-v1/analyses/history-readouts/tables/history.csv).

| Matched physical task at100 fs | Endpoint MSE | Marginal NLL | Whole-path energy | Response relative MSE |
| --- | ---: | ---: | ---: | ---: |
| Physical values | 1.148852 | 1.570256 | 0.631005 | 1.388881 |
| Physical values + derivatives | 1.141877 | 1.594747 | 0.630523 | 1.367736 |

Response values average the eight group/direction ratios and three neural keys,
then the three training seeds. Zero response is1 at100 fs. Derivatives reduce
response error by1.52%, but both treatments remain worse than zero; predictive NLL
worsens with derivatives. The largest failure is density response to a position
perturbation (2.51216 with derivatives); density response to a velocity perturbation
is essentially zero-reference performance (1.00074). No100-fs family/direction
average beats zero in the derivative treatment. The100-fs macro error changes to
1.378568 at half epsilon, so this is not rescued by that secant refinement.
[Paired sensitivity](../../output/measurement_sim/overnight-20261012-v1/analyses/comparison/tables/paired_sensitivity.csv).

The training curves show a generalization failure: between updates200 and4000,
the derivative model's fixed training-panel response loss decreases from about1.21
to0.47 while selection response loss increases from about1.58 to2.83. Allsix
physical fits select update200, the earliest checkpoint considered, by selection
NLL. The reported held-out results use those selected checkpoints, not the final
overfit checkpoint. There are only80 fitting anchors from16 fitting ancestors.

The diagnostic given the known present response finds some short-time signal:
full80-anchor linear transport has mean Q/V-group relative error about0.704 at20 fs,
0.962 at50 fs and1.071 at100 fs (zero=1 at these offsets). State-conditioned
residual readouts help most near5 fs but do not repair100-fs generalization. Their
20–100-fs mean error improves as the fitting subset grows, yet remains above zero
at the full80 anchors. This readout receives H and known R0; its information
contract differs from ordinary trajectory inference. It neither proves response
unlearnability nor demonstrates a useful long-time simulator.
[Response readouts and data curves](../../output/measurement_sim/overnight-20261012-v1/analyses/response-readouts/tables/response.csv).

The supported next questions are response generalization and temporal dependence.
Extra ordered history and matching the short teacher protocol have not, on their
own, made motion/derivative training useful for long native futures. No new
nucleus-birth prediction evaluation was part of this queue.

<a id="density-rates-20261013"></a>

### Density rates, more physical states and smaller steps: all 36 fits completed

The [October 13 campaign](../../output/measurement_sim/density-rates-20261013-v1/README.md)
completed all 36 fits at the fixed final 1,600-update checkpoint, all evaluations,
18 fixed readouts and 751 physical anchor packets in about 6.01 hours. There are
only train and test roles: the pooled test contains 310 anchors from 62 sources,
including former selection sources. No test-based selection or inner split was
used. Results below average three training seeds and three neural noise keys.
These are matched 100-fs physical continuations, not long native futures.

| Training states / sources | Rate inputs | Derivatives | Neural step fs | Endpoint MSE | Marginal NLL | Response macro RME |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 80 / 16 | No | Yes | 5 | 1.0588 | 2.0761 | 2.5335 |
| 320 / 16 | No | Yes | 5 | 0.9843 | 1.6551 | 1.8589 |
| 320 / 28 | No | No | 5 | 1.0238 | 1.6849 | 2.6338 |
| 320 / 28 | No | Yes | 5 | 0.9973 | 1.6219 | 1.9559 |
| 320 / 28 | Yes | Yes | 5 | 1.0003 | 1.6485 | 1.8205 |
| 320 / 28 | Yes | Yes | 1 | 1.0024 | 1.6561 | 1.7985 |

Response macro RME averages eight group/direction error-to-target-power ratios;
zero response scores 1. Lower is better in every score column. Definitions remain
those of the frozen [metric contract](../metrics/measurement_rates.md).

Increasing states from 80 to 320 within the original 16 sources reduces derivative
model endpoint MSE by 7.0%, consistently across seeds. Increasing source diversity
at fixed 320 states has mixed effects: better likelihood but worse endpoint MSE.
On diverse320, adding derivative supervision improves endpoint MSE by 2.6%, NLL
by 3.7%, and response error by 25.7%, in all three paired seeds.

Adding density rates to that derivative model lowers response error by 6.9%, but
worsens NLL in every seed. Gains concern position perturbations; all four
velocity-direction group means worsen. The smaller 1-fs step gives no convincing
benefit over 5 fs. All eight neural family/direction means at 100 fs remain worse
than zero response, and nominal 90% intervals cover only about 65% of outcomes.
Thus improved forecasting is demonstrated, but successful physical sensitivity
learning is not. The fixed linear transport readout, which additionally receives
the known initial response, reaches macro RME 0.9825; this small improvement uses
a different information contract. No new nucleus-birth evaluation was included.

<details>
<summary>Ni completed results and recorded follow-ups</summary>

- [Original/relaxed response comparison](../../experiments/ni_response_20261009/README.md) and [quench repeatability](../../experiments/ni_quench_repeatability_20261009/README.md).
- [Compact pilot](../../experiments/ni_latent_sim_20261010/README.md) and [completed derivative/stability factorial](../../experiments/ni_latent_sim_20261010/DERIVATIVE_STABILITY.md).
- [Regime expansion](../../experiments/ni_latent_sim_20261010/REGIMES3X.md), [dense-path design](../../experiments/ni_latent_sim_20261010/DENSE_PATH.md), [transformer extension](../../experiments/ni_latent_sim_20261010/TRANSFORMER.md), [October 11 comparison](../../experiments/ni_latent_sim_20261010/AUTONOMOUS_OVERNIGHT.md), and [execution/batch history](../ni_latent_sim.md).

</details>

## How this evidence changes the next decision

The record supports retaining strong present-information controls, comparing actual input and target contracts, and checking a representation through more than one readout. It also contains reasons to stop particular branches: failed stability gates, uninformative trained codes, unsuccessful matched response supervision and discontinued objectives. The [methods](methods.md) and [experiment plan](experiment_plan.md) turn these findings into conditional next steps. They must preserve old results, ancestry, frozen metric definitions and canonical cohort roles; an untested explanation remains a hypothesis until the proposed discriminating analysis is complete.
