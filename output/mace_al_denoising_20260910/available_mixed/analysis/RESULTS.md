# Al temporal denoising — September 10, 2026

Question: do several thermal observations improve prediction of the relaxed-anchor topology?

All backbone weights are frozen. Whole-source splits are declared in the configuration before fitting. Target and feature scaling use training data only; checkpoints use validation only. All reported models and seeds were configured before test evaluation.

| Predictor | Test balanced MSE | Within-frame mean block R² |
| --- | ---: | ---: |
| anchor_ridge | 0.89152 | 0.4517 |
| mean_ridge | 0.69642 | 0.5716 |
| relaxed_ridge | 0.06980 | 0.9570 |
| hot_tda_ridge | 464.60477 | -285.5436 |
| train_mean | 5.10764 | -2.1649 |
| hot_tda_direct | 8.40253 | -4.1235 |
| anchor_pca | 0.91710 ± 0.10302 | 0.4363 |
| mean_pca | 0.56031 ± 0.05258 | 0.6554 |
| transformer_pca_regularized | 0.44245 ± 0.06469 | 0.7263 |
| transformer_pca_tda | 0.26928 ± 0.02240 | 0.8338 |
| transformer_blocks | 0.23485 ± 0.01815 | 0.8547 |
| anchor_blocks | 0.68754 ± 0.02853 | 0.5768 |
| mean_blocks | 0.40728 ± 0.01147 | 0.7490 |
| anchor_continue_blocks | 0.55231 ± 0.01264 | 0.6611 |
| residual_blocks | 0.25991 ± 0.01546 | 0.8397 |
| atom_anchor_blocks | 0.23151 ± 0.01268 | 0.8567 |
| atom_temporal_blocks | 0.22570 ± 0.01224 | 0.8605 |
| relaxed_blocks | 0.12463 ± 0.00326 | 0.9233 |

Within-frame R² uses variation among neighborhoods of the same source frame as its denominator; it penalizes prediction errors including frame-mean bias. It tests more than recognizing temperature or phase.

## Matched comparisons

| Candidate versus reference | Relative MSE reduction | Source bootstrap 95% interval |
| --- | ---: | ---: |
| transformer_pca_tda_versus_transformer_pca_regularized | 38.37% | [34.79%, 42.47%] |
| transformer_blocks_versus_transformer_pca_tda | 11.50% | [5.97%, 18.70%] |
| residual_blocks_versus_anchor_continue_blocks | 49.49% | [23.82%, 63.68%] |
| residual_blocks_versus_mean_blocks | 34.00% | [21.72%, 44.16%] |
| atom_temporal_blocks_versus_atom_anchor_blocks | 3.07% | [-0.09%, 5.42%] |

Bootstrap resamples whole test trajectories after averaging training seeds. 2 held-out trajectories limit confidence. The continued anchor control has the same anchor-head warm start and additional training budget as residual fusion. Atom-anchor repeats the current atom features in every time slot, preserving temporal-module capacity.

## History interventions

| Model | Real history MSE | Repeated anchor | Reversed past |
| --- | ---: | ---: | ---: |
| transformer_pca_regularized | 0.44245 | 0.48035 | 0.45095 |
| transformer_pca_tda | 0.26928 | 0.28743 | 0.26883 |
| transformer_blocks | 0.23485 | 0.26035 | 0.23558 |
| residual_blocks | 0.25991 | 0.28166 | 0.26076 |
| atom_anchor_blocks | 0.23151 | 0.23151 | 0.23151 |
| atom_temporal_blocks | 0.22570 | 0.26773 | 0.22632 |

## Potential-specific test errors

| Predictor | Al1_EAM_FS | Lee2003_MEAM |
| --- | ---: | ---: |
| anchor_ridge | 0.36352 | 1.24352 |
| mean_ridge | 0.31993 | 0.94740 |
| relaxed_ridge | 0.03218 | 0.09488 |
| hot_tda_ridge | 0.50157 | 774.00691 |
| train_mean | 5.84749 | 4.61440 |
| hot_tda_direct | 13.83877 | 4.77838 |
| anchor_pca | 0.43809 | 1.23645 |
| mean_pca | 0.36473 | 0.69070 |
| transformer_pca_regularized | 0.47966 | 0.41765 |
| transformer_pca_tda | 0.31278 | 0.24027 |
| transformer_blocks | 0.29412 | 0.19534 |
| anchor_blocks | 0.41024 | 0.87241 |
| mean_blocks | 0.36192 | 0.43753 |
| anchor_continue_blocks | 0.37188 | 0.67259 |
| residual_blocks | 0.28331 | 0.24430 |
| atom_anchor_blocks | 0.27328 | 0.20367 |
| atom_temporal_blocks | 0.25847 | 0.20386 |
| relaxed_blocks | 0.04484 | 0.17782 |

Available Al pilot: existing Al1 EAM/FS and Lee2003 MEAM relaxed labels, with CG and FIRE targets retained separately by provenance. Five actual snapshot times are supplied per sample.

Exploratory whole-continuation holdout, not independent-campaign evidence: EAM continuations share an ancestral trajectory. MEAM sources 000/001/002 are assigned train/val/test before this fit. EAM 166/170/174 train, 175 validation, 177 test. The EAM test continuation was validation data in earlier experiments. One held-out trajectory per potential; potential is confounded with temperature (650 versus 400 K), history spacing (0.1 versus 0.75 ps), and source campaign. Between-potential score differences cannot establish a causal potential effect. Original CG is preferred over duplicate FIRE labels; no input is duplicated or its target averaged.

Full per-temperature and H0/H1/H2 scores, training exposure counts, encoder exports and numerical checks are in [metrics.json](metrics.json). Raw predictions are in predictions.npz; [comparison plot](comparison.png).

All compared models share the same data and receive the actual per-history snapshot times. Relaxed-input results are diagnostic references; inference still receives thermal observations.
