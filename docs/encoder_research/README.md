# Encoder research handbook

[Rich-MACE checkpoint interface comparison](../spatial_vicreg_bias.md#current-rich-mace-checkpoint-in-the-interface-viewer) adds the active multimaterial MACE256 correlation3/angular3 state to the matched and relaxed static Al viewers; descriptor supervision is explicitly recorded.

[GeoFormer versus descriptors: checkpoint explorer](../../output/spatial_vicreg_bias/matched-al64-20260929/analyses/interface-pacmap-v1/index.html)
compares the saved 3D PaCMAP layouts and full-snapshot MD clusters. Choose epochs
4, 12 or 24; pairing variants and training repeats are under advanced controls.
Optimal cluster colors are accompanied by a membership matrix and per-pair IoU.
The default is a fixed spatial-neighbor VICReg example, not a selected best model.

[Six relaxed Al snapshots, 166–240 ps](../../output/spatial_vicreg_bias/static-al-six-20260929/analyses/interface-v1/index.html):
frozen GeoFormer and rich descriptor clusters, PaCMAP and two dense MD panels
with consistent palettes. [Protocol and execution](../spatial_vicreg_bias.md)
distinguish this descriptive static transfer from the held-out MD study.

**Interface-focused correction:** [rich-descriptor cluster correspondence](../../experiments/spatial_vicreg_bias_20260929/INTERFACE_CORRESPONDENCE.md) compares independent TDA/bond-order/CNA partitions with neural clusters around the crystal-side boundary. Earlier scalar family-level prediction summaries are a different diagnostic.

[Completed spatial VICReg mechanism study](../../experiments/spatial_vicreg_bias_20260929/RESULTS.md):
nine 24-pass GeoFormers. Neighbor alignment contracts spatial pair distances,
but global liquid-cluster resolution disappears even without it. Continuous
liquid TDA readout improves while global K=7 membership loses predictive skill.
Simple crystal-fraction clusters reproduce much of the all-phase association.
Raw encoder and projector differ substantially; no precursor claim is established.
[Plots and tables](../../output/spatial_vicreg_bias/matched-al64-20260929/analyses/completed-review-v1/README.md) ·
[Archived-coordinate reference](../../experiments/spatial_vicreg_bias_20260929/COORDINATE_FINDINGS.md).

[Rich descriptor control](../../experiments/liquid_predictability_20260928/DESCRIPTORS.md) compares fixed topology/order/CNA features with learned encoders using matched liquid-only observations.


[Completed encoder mechanisms experiment](../../experiments/encoder_mechanisms_20260926/COMPLETED-RESULTS.md):
18 completed scientific fits; alignment trades physical retention for prediction,
frozen Epi beats scratch, and this fine-tuning recipe adds no predictive benefit.
All three seeds and fixed training trajectories are reported with paired-source
uncertainty; precursor detection remains unsupported by the current birth cohort.

[Liquid structure predictability](../../experiments/liquid_predictability_20260928/README.md)
separates descriptor signal, frozen representation information, joint adaptation,
source diversity and actual observation clearance.

[LCD-MACE128-VC liquid-only localization](../../experiments/crystal_interface_20260928/LIQUID_DISTANCE.md)
targets distance to an existing external crystal without established crystal in
the input; it replaces the unstarted interface-only exclusion fit.

[Feature dominance in the interface encoder](../../experiments/crystal_interface_20260928/FEATURE_DOMINANCE.md)
distinguishes predictive concentration from train-only feature overfitting using
PCA readouts and interventions on the frozen context predictor.

[Interface visibility/overfitting audit](../../experiments/crystal_interface_20260928/OVERFITTING.md)
checks all six localization fits; [dense invisible-context adaptation](../../experiments/crystal_interface_20260928/UNSEEN.md)
retains VCReg and the frozen original benchmark.

[CIV-MACE128 interface study](../../experiments/crystal_interface_20260928/README.md)
learns distance/direction to an interface atom layer, including crystal interiors;
it is a separate target family from crystal-set CDV-MACE128.
[Completed comparison](../../experiments/crystal_interface_20260928/RESULTS.md)
covers all six fits and their held-out localization/embedding diagnostics.

[CDV-MACE128](../../experiments/crystal_vector_20260928/README.md) jointly trains
equivariant patch exports and vector-message spatial context for current crystal
distance/direction, with independent random batches and a matched VCReg treatment.

[CD-MACE128 and CD-MACE128-VC](../../experiments/distance_encoder_20260926/VCREG.md)
cover joint crystal-distance training, local-only prediction and the supervised
variance–covariance regularization comparison. These are distinct from paired-view
VICReg pretraining. Exact input/metric definitions and stable checkpoint identities
are linked in that protocol.

[Joint distance-encoder experiment](../../experiments/distance_encoder_20260926/README.md)
trains MACE on 12.75M dynamic multi-material patches, with greater likelihood
weight on 20–32 Å proximity and subsequent vector/harmonic context evaluation.

[Confidence and visibility results](../../experiments/spatial_distance_20260926/CONFIDENCE_RESULTS.md)
and [continuous distance training](../../experiments/spatial_distance_20260926/README.md)
extend the fixed-snapshot approach study.

[Spatial warning-distance study](../../experiments/spatial_approach_20260926/README.md) ·
[Results and interpretation](../../experiments/spatial_approach_20260926/RESULTS.md)
compares local and surrounding observations while moving through a fixed
snapshot toward a crystal, with explicit input-visibility and false-alarm controls.

A working reference for understanding **what each encoder was trained to retain,
what its exported state contains, and what the measurements actually establish**.
Scope: GeoFrame through the September23 distance/future factorial and the
completed 35-pass GeoFrameV2 reproduction,
including archived studies, local H100/RTX/A100 work and the H200 results reported
to this repository. This is an evidence snapshot, not a live run monitor.

| I want to… | Start here |
| --- | --- |
| Decide the next experiments from what we already tried | [Evidence audit and staged encoder-mechanism experiments](../../experiments/encoder_mechanisms_20260926/README.md) |
| Find prior work on continuous, physical and predictive representations | [Broad literature review: atomic dynamics, predictive states, feature suppression and competing liquid order](representation_literature_20260926.md) |
| Assess Ta force-field uncertainty before interpreting encoder results | [Potential comparison and validation priorities](../simulations/ta_potential_review_20260926.md), [24-shot pilot](../simulations/ta_shooting_20260926.md) |
| Compare our spatial warning experiment with prior work | [Al interface GNNs, structural scores and precursor studies](spatial_warning_literature_20260926.md) |
| Read the completed latest-MACE comparison | [Eight checkpoints: liquid information, spatial limits and prediction](../../experiments/encoder_quality_20260926/RESULTS.md) |
| Evaluate the latest efficient MACE on structure and predictive information | [Eight frozen checkpoints and matched controls](../../experiments/encoder_quality_20260926/README.md), [execution](../encoder_quality.md) |
| Define a good encoder and the evidence needed to establish it | [Quality criteria, controls and falsifiable training hypotheses](quality_criteria.md) |
| Review MACE/Epi training curves and the parameter queue's actual completion | [26 September evidence review](../../output/encoder_research/mace-vicreg-epi-20260923/RESULTS-20260926.md) |
| Separate local establishment from crystal arrival | [Cluster-ancestry availability audit](crystallization_origins.md) |
| Assess nucleus and prestructured-liquid definitions | [Literature, current implementation and unresolved ambiguities](nuclei_and_prestructured_liquid_20260926.md) |
| Revisit nucleation literature and assess possible novelty | [Primary sources, falsifiable hypotheses and evidence checklist](nucleation_literature_and_open_questions.md) |
| Compare encoder training with vector/harmonic predictors | [Twelve-epoch minimum protocol](../../experiments/encoder_context_epochs_20260925/README.md), [execution and full evaluation](../encoder_context.md) |
| Compare models on identical fixed data | [Al64 benchmark and 1.16M structural training neighborhoods](../datasets/fixed_al64.md) |
| Review orientation-preserving context prediction | [Completed Al16 results](/work/PERSO/vmorozov/analysis/equivariant_context/node59-b512-v2-20260925/RESULTS.md), [four-model protocol](../../experiments/equivariant_context_20260925/README.md), [operations](../equivariant_context.md) |
| Use the implemented MACE acceleration | [Runtime changes, parity and measured speed](runtime_refactor_20260925.md) |
| Compare MACE widths 128 and 130 | [Matched speed and memory measurements](width128_vs130_20260925.md) |
| Find pipeline inefficiencies and tested fixes | [Current MACE/context efficiency audit](pipeline_efficiency_audit_20260925.md) |
| Use faster typed-field extraction | [Prefetch configuration and timings](../equivariant_context.md#batched-feature-extraction), [measured 5.3–5.8× speedup](pipeline_efficiency_audit_20260925.md#feature-extraction-implementation-and-measurement) |
| Plan shared computation across focal neighborhoods | [Implementation design and trade-offs](shared_spatial_implementation.md) |
| Speed up the encoder pipeline and understand the trade-offs | [Measured runtime audit and refactor priorities](performance_refactor_20260925.md) |
| Understand symmetric spatial context and its next design | [Actual geometry inputs, hierarchy and literature](spatial_context_20260925.md) |
| Follow the likelihood-trained 500k/1M/2M comparison | [Protocol](../../experiments/supervised_information_20260925/README.md), [operations](../supervised_capacity.md) |
| Know exactly what each prediction could observe | [Input policy and275-row context ledger](prediction_context.md) |
| Understand symmetric spatial context and design its successor | [Code audit, literature and proposed comparisons](spatial_context.md) |
| Compare our MACE depth and size with the MLIP | [Measured layers and parameter counts](mace_sizes.md) |
| Separate task-supervised and self-supervised encoder research | [Training branches](training_branches.md) |
| Review the completed larger supervised AP3/AP6 study | [Results](../../output/encoder_supervised/ap36-large-20260924/RESULTS.md), [six-arm protocol](../../experiments/supervised_onset_20260924/README.md) |
| Understand the models and training losses | [Encoder family guide](encoders.md) |
| Understand a metric or compare two results correctly | [Evaluation guide](evaluation.md) |
| See the main findings without reading every run | [Results overview](results.md) |
| See the completed 24 September onset, spatial-context and MACE/Epi screens | [Latest results, uncertainty and stability](results_20260924.md) |
| Compare 3 ps primary, 6 ps and 12 ps AP from all saved native predictions | [Horizon review](../../output/encoder_research/onset-horizons-20260924/RESULTS.md), [all 246 comparisons](../../output/encoder_research/onset-horizons-20260924/tables/all-models.csv) |
| Search every indexed report, table and gallery | [Interactive catalogue](../../output/encoder_research/catalogue/index.html) |
| Query/export the underlying evidence | [Database guide](database.md), [SQLite](../../output/encoder_research/catalogue/technical/results.sqlite), [curated CSV](../../output/encoder_research/catalogue/tables/headlines.csv) |
| Follow the broad native snapshot comparison | [Queue and evaluation protocol](../encoder_screen.md), [current results](../../output/encoder_research/screen-20260923/index.html) |
| Find plots or repeat an analysis | [Analysis and visualization guide](analysis.md) |
| Improve how we organize and interpret future research | [Workflow recommendations](improvements.md) |
| Study information useful for crystallization | [Likelihood-based protocol](../../experiments/supervised_information_20260925/README.md), [standing research policy](training_branches.md) |
| Inspect the historical predictive/robustness experiments | [Eight-arm literature-led protocol](../../experiments/robust_onset_20260924/README.md), [Slurm operations](../robust_onset.md) |
| Inspect the retired spatial hierarchy experiments | [Spatial hierarchy protocol](../../experiments/spatial_hierarchy_20260924/README.md), [queue guide](../spatial_hierarchy.md) |
| Measure state and temporal-movement dimensions | [Matched 0.75 ps table with noise response](../../output/encoder_research/noise-lag075-20260924/RESULTS.md), [guide](embedding_dynamics.md), [definitions](../metrics/embedding_dynamics.md) |
| Measure response to input noise | [Combined noise/trajectory table](../../output/encoder_research/noise-lag075-20260924/RESULTS.md), [guide](input_noise.md), [definitions](../metrics/embedding_noise.md) |
| Find a dated scientific protocol | [Study index](studies.md), [active experiments](../../experiments/README.md) |

In discussion and new result tables we call the model **Geoformer**, following
the user's terminology. Its implementation classes remain `GeoFrameTransformer`
and `GeoFrameTransformerV2`; older files and archived reports use GeoFrame.
Those internal names do not denote a different model in these comparisons.

Our recurring lesson is that four questions need separate evidence: does an
embedding preserve present structure, organize useful neighbors, respond
appropriately to dynamics, and improve future prediction? Smoothness, embedding
rank, attractive clusters and decoder accuracy can move in different directions.
The database deliberately has **no universal encoder leaderboard**.

For new onset experiments, **3 ps is the main horizon**, 6 ps is secondary, and 12 ps
is retained for context. AP is diagnostic only: no AP loss or AP-driven selection.
See the [current policy](training_branches.md).
The completed models retain their historical checkpoint/readout selectors;
reporting AP3 does not mean they were trained or selected for AP3. The small
development cohort has only three positive windows at3 ps, so use its rankings
as exploratory evidence. Trajectory stability still uses0.75 ps differences.

The [35-pass GeoFrame trajectory](../../output/geoframe_evolution/epoch34-review-20260923/RESULTS.md)
now separates encoder and projector behavior: liquid-order readout improves in
the encoder while declining in the projector across Al, Ta and Zr. Improved
fault/interface-proxy readout does not imply separate liquid clusters or better
conditional crystallization forecasts. The historical epoch34 VICReg and epoch159
VISReg pictures used different objectives; they are not an unchanged training
trajectory. See the [new evidence and limits](results.md#geoframe-through-35-passes).

A compact history is: canonical-frame GeoFrame → continuous density/MACE controls
→ pretrained MACE topology and context studies → native geometry/velocity/history
models → shared structural MACE/GATr and neighborhood prediction → relaxed-input
studies → BCR conditioning audits → simpler fixed-geometry encoder training and
its distance/future additions. This is a research sequence, not a claim that every
later model outperformed its predecessor.

The catalogue retains negative findings, historical revisions, smoke outputs and
reported-only results. Their presence is not an endorsement or a completed
scientific comparison. The human guide selects meaningful comparisons; the raw
database preserves the supporting evidence without silently normalizing unlike
metrics. H200 summaries are explicitly distinguished from locally available raw
results. Archive copies are not new fits, and the same checkpoint can occur in
several analyses.

To refresh after new results, activate `pointnet-torch214`, register their study
and evidence collection in [catalogue.json](catalogue.json), add checked findings
in [highlights.json](highlights.json), and run:

```bash
python scripts/experiment_registry.py encoders
```

This reads existing reports/tables and writes the catalogue. It does not run
training, inference, simulation, scheduler queries or hardware benchmarks.
Machine storage roots come from `machine.local.yaml`; the archive must be mounted.
See [coverage and refresh instructions](database.md). Source scientific protocols
remain in `experiments/`; this folder is their cross-study guide.

The [historical parameter search](../encoder_parameter_search.md) recorded28 matched fits after the [29-checkpoint review](../../output/encoder_research/parameter-search-20260923/RESULTS.md). Liquid-neighbor measurements favor the late VISReg raw export despite its two-blob visualization; structure and calibrated prediction remain separate selection criteria.

The [paired MACE + Epi study](../../experiments/mace_paired_epi_20260923/README.md) compares direct temporal alignment with VICReg versus geometric Epi regularization (with/without a variance floor), using two scratch seeds and24 complete passes. All six fits and24 checkpoint evaluations are complete; see the [results](results_20260924.md#mace-paired-alignment-epi-versus-vicreg). This is separate from the conditional-JEPA checkpoint screen.

The [matched execution workflow](../encoder_mechanisms.md) implements the alignment, readout and adaptation contrasts and held-out birth-coverage audit.
The [28 September interim results](../../experiments/encoder_mechanisms_20260926/RESULTS-20260928.md)
show an early structure/prediction trade-off: all nine encoders trained24 epochs,
but later milestone evaluation needs recovery. Descriptor add-back survives a
dimension-matched redundant-feature control; strict held-out birth coverage is
insufficient for a precursor claim.
The [later epoch-24 update](../../experiments/encoder_mechanisms_20260926/PROGRESS-20260928.md)
confirms the alignment trade-off in all three Epi seeds; four quality assays and
the nine adaptation fits remain outstanding at its cutoff.
