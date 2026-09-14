# Al temporal denoising — September 10, 2026

Question: do several thermal observations improve prediction of the relaxed-anchor topology?

All backbone weights are frozen. Whole-source splits are declared in the configuration before fitting. Target and feature scaling use training data only; checkpoints use validation only. All reported models and seeds were configured before test evaluation.

| Predictor | Test balanced MSE | Within-frame mean block R² |
| --- | ---: | ---: |
| anchor_ridge | 0.03765 | 0.8272 |
| mean_ridge | 0.02991 | 0.8631 |
| relaxed_ridge | 0.00965 | 0.9556 |
| hot_tda_ridge | 0.04922 | 0.7738 |
| train_mean | 0.83982 | -2.8556 |
| hot_tda_direct | 1.46298 | -5.8099 |
| anchor_pca | 0.03823 ± 0.00022 | 0.8243 |
| mean_pca | 0.03293 ± 0.00018 | 0.8490 |
| transformer_pca_regularized | 0.06325 ± 0.00072 | 0.7095 |
| transformer_pca_tda | 0.03190 ± 0.00056 | 0.8537 |
| transformer_blocks | 0.02738 ± 0.00012 | 0.8746 |
| anchor_blocks | 0.03751 ± 0.00005 | 0.8277 |
| mean_blocks | 0.03299 ± 0.00003 | 0.8488 |
| anchor_continue_blocks | 0.03499 ± 0.00009 | 0.8394 |
| residual_blocks | 0.02699 ± 0.00018 | 0.8764 |
| atom_anchor_blocks | 0.03321 ± 0.00032 | 0.8476 |
| atom_temporal_blocks | 0.02554 ± 0.00011 | 0.8831 |
| relaxed_blocks | 0.01395 ± 0.00007 | 0.9362 |

Within-frame R² uses variation among neighborhoods of the same source frame as its denominator; it penalizes prediction errors including frame-mean bias. It tests more than recognizing temperature or phase.

## Matched comparisons

| Candidate versus reference | Relative MSE reduction | Source bootstrap 95% interval |
| --- | ---: | ---: |
| transformer_pca_tda_versus_transformer_pca_regularized | 49.56% | [47.43%, 51.56%] |
| transformer_blocks_versus_transformer_pca_tda | 14.17% | [11.88%, 17.44%] |
| residual_blocks_versus_anchor_continue_blocks | 22.85% | [20.48%, 25.29%] |
| residual_blocks_versus_mean_blocks | 18.19% | [11.67%, 25.71%] |
| atom_temporal_blocks_versus_atom_anchor_blocks | 23.08% | [20.52%, 26.46%] |

Bootstrap resamples whole test trajectories after averaging training seeds. 6 held-out trajectories limit confidence. The continued anchor control has the same anchor-head warm start and additional training budget as residual fusion. Atom-anchor repeats the current atom features in every time slot, preserving temporal-module capacity.

## History interventions

| Model | Real history MSE | Repeated anchor | Reversed past |
| --- | ---: | ---: | ---: |
| transformer_pca_regularized | 0.06325 | 0.07501 | 0.06332 |
| transformer_pca_tda | 0.03190 | 0.04803 | 0.03256 |
| transformer_blocks | 0.02738 | 0.04030 | 0.02792 |
| residual_blocks | 0.02699 | 0.03908 | 0.02742 |
| atom_anchor_blocks | 0.03321 | 0.03321 | 0.03321 |
| atom_temporal_blocks | 0.02554 | 0.24126 | 0.02590 |

## Potential-specific test errors

| Predictor | Lee2003_MEAM |
| --- | ---: |
| anchor_ridge | 0.03765 |
| mean_ridge | 0.02991 |
| relaxed_ridge | 0.00965 |
| hot_tda_ridge | 0.04922 |
| train_mean | 0.83982 |
| hot_tda_direct | 1.46298 |
| anchor_pca | 0.03823 |
| mean_pca | 0.03293 |
| transformer_pca_regularized | 0.06325 |
| transformer_pca_tda | 0.03190 |
| transformer_blocks | 0.02738 |
| anchor_blocks | 0.03751 |
| mean_blocks | 0.03299 |
| anchor_continue_blocks | 0.03499 |
| residual_blocks | 0.02699 |
| atom_anchor_blocks | 0.03321 |
| atom_temporal_blocks | 0.02554 |
| relaxed_blocks | 0.01395 |

Five snapshots at 0.75 ps cadence.

Few independent test sources limit precision.

Full per-temperature and H0/H1/H2 scores, training exposure counts, encoder exports and numerical checks are in [metrics.json](metrics.json). Raw predictions are in predictions.npz; [comparison plot](comparison.png).

All compared models share the same data and receive the actual per-history snapshot times. Relaxed-input results are diagnostic references; inference still receives thermal observations.
