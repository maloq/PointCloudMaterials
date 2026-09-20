# Crystallization transfer: completed results, 19 September 2026

Both queues completed: **102 trained models plus one no-transition baseline**, with no runtime failures. The initial queue finished at 05:26 Paris time; the scaling queue at 08:15. Learning quality differs sharply: frozen-feature models are useful, while the trainable-encoder results show an optimization/normalization problem.

The common test population is 44,385 eligible liquid windows from 30 independent Al trajectories. Scaling uses up to 109,838 eligible training windows from 90 trajectories, separate from the 15 selection and 15 alarm-calibration sources. One training seed was used. The initial 2,048-update sampler and subsequent shuffled-epoch sampler are separate protocols.

## Frozen encoder: controlled scaling

All rows below use spatial attention and 12 ps history. Lower event-time negative log-likelihood (NLL) is better; average precision (AP) is higher-is-better. The 9 ps positive prevalence is 0.0312.

| Comparison | Setting | Test NLL | 9 ps AP |
| --- | --- | ---: | ---: |
| Radius, three epochs | 0 Å | 1.1118 | 0.3308 |
| Radius, three epochs | 6 Å | 1.1045 | 0.3638 |
| Radius, three epochs | 12 Å | 1.0675 | 0.4589 |
| Radius, three epochs | 18 Å | 1.0384 | 0.5229 |
| Radius, three epochs | 25 Å | 0.9928 | 0.5391 |
| Duration, 25 Å | 1 epochs | 0.9883 | 0.5094 |
| Duration, 25 Å | 3 epochs | 0.9928 | 0.5391 |
| Duration, 25 Å | 6 epochs | 0.9637 | 0.5275 |
| Data, 5,151 updates | 30 sources | 1.0480 | 0.5183 |
| Data, 5,151 updates | 60 sources | 1.0206 | 0.5367 |
| Data, 5,151 updates | 90 sources | 0.9928 | 0.5391 |

![Frozen-encoder scaling curves](plots/scaling.png)

At three epochs, 25 Å context lowers NLL by 10.7% versus center-only input and 7.0% versus 12 Å. Attention beats the matched weighted-mean context by 7.5%. These are sparse contextual-center radii; the local MACE support stays fixed. Larger support also admits more of the seven cached candidate centers.

Six epochs improves NLL by 2.9% versus three epochs, with the selected checkpoint at update 7,168 (about 4.18 passes), not the final update 10,302. Improvement is not monotonic: three epochs does not clearly beat one, and 9 ps AP is slightly higher at three epochs despite better overall event NLL at six.

Using 90 versus 30 independent training sources improves NLL by 5.3% at the same update budget. The 90-versus-60 interval includes zero. Increasing windows from 25% to 100% within all 90 sources did not demonstrate benefit: NLL 0.9814 versus 0.9928 respectively. More independent trajectories and denser sampling of the same trajectories have different effects.

## History and tensor context

In the initial matched 2,048-update comparison, frozen attention gives NLL 1.0121 for a current snapshot, 0.9778 for 12 ps history, and 0.9776 for 48 ps history. The separately trained repeated-current 48 ps control gives 1.0120. Thus real history helps by about 3.4%; 48 ps does not clearly improve on 12 ps.

Tensor-context gains are small and inconsistent. At three epochs, scalar/tensor NLL is 0.9928/0.9924; at six epochs it is 0.9637/0.9659. Both paired intervals include zero. This tests the implemented tensor contractions and alignment attention, not every possible equivariant context architecture.

## Paired source uncertainty

Negative differences favor the first option. Intervals use 5,000 paired, temperature-stratified whole-source bootstrap draws. They condition on one training seed and subset realization, without multiplicity correction.

| Comparison, A minus B | NLL difference | 95% source interval |
| --- | ---: | --- |
| radius25 vs0 | -0.1190 | [-0.1380, -0.1001] |
| radius25 vs12 | -0.0747 | [-0.0876, -0.0618] |
| attention vs mean | -0.0810 | [-0.0989, -0.0648] |
| epochs6 vs3 | -0.0292 | [-0.0471, -0.0109] |
| epochs6 vs1 | -0.0246 | [-0.0335, -0.0158] |
| epochs3 vs1 | +0.0046 | [-0.0116, +0.0204] |
| sources90 vs30 | -0.0552 | [-0.0990, -0.0102] |
| sources90 vs60 | -0.0278 | [-0.0651, +0.0098] |
| windows100 vs25 | +0.0115 | [-0.0055, +0.0274] |
| tensor vs scalar E3 | -0.0005 | [-0.0017, +0.0008] |
| tensor vs scalar E6 | +0.0022 | [-0.0015, +0.0060] |
| history12 vs snapshot | -0.0344 | [-0.0403, -0.0288] |
| history48 vs12 | -0.0001 | [-0.0024, +0.0022] |
| real48 vs repeated48 | -0.0344 | [-0.0414, -0.0279] |

## Prediction quality of the validation-selected recipe

The lowest selection NLL across the completed recipes is frozen scalar MACE, 12 ps history, 25 Å attention, six-epoch budget. Its held-out event NLL is **0.9637**. This recipe was ranked by selection NLL, not by test score.

| Horizon | AP | AUROC | Window recall | Precision | Actual test FPR | Detected-window timing MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 ps | 0.527 | 0.954 | 72.6% | 38.9% | 3.67% | 1.87 ps |
| 24 ps | 0.614 | 0.886 | 62.2% | 60.7% | 4.04% | 4.49 ps |
| 96 ps | 0.633 | 0.709 | 31.6% | 75.9% | 5.82% | 15.52 ps |

Thresholds target at most 5% FPR on separate calibration sources; the actual test FPR need not be 5%. Timing MAE conditions on detected positive windows and must not be read without misses. At 96 ps, 8,336 of 12,127 positive windows are missed; source-weighted window recall is 31.6%.

Repeated monitoring detects an alarm episode for 395/409 observable event centers at the 96 ps horizon, with mean detected lead 44.0 ps and 5.11 false alarm episodes per sampled center-ns. That 96.6% event coverage is **not** 96.6% accuracy at a fixed 96 ps lead. At 9 ps, episode detection is 163/378 centers, mean lead 5.55 ps, and 3.57 false episodes per sampled center-ns. The episode rule counts a new alarm onset, so continuously positive earlier alarms do not repeatedly trigger near the event.

Spatial diagnostics remain sparse: 16 tracked centers per trajectory. Predicted transforming-fraction MAE is 0.0414 at 9 ps, 0.0909 at 24 ps and 0.2722 at 96 ps; nearby-pair difference RMSE is 0.1350, 0.2306 and 0.3801. These do not measure dense front localization or propagation speed.

## Why the fine-tuned and scratch results are not decisive architecture comparisons

At three epochs with 12 ps history / 25 Å attention, frozen scalar, fine-tuned scalar and scratch scalar NLL are 0.9928, 1.1349 and 1.1346. The trainable models have almost no short-horizon geometry skill. Six epochs only partly helps scalar fine-tuning (1.1189); scratch remains poor (1.1381).

A read-only GPU audit used 128 training-only current-center graphs. It found:

- Parent cached and freshly computed compiled features match closely (RMSE 1.82e-7). A cache/online feature mismatch is unlikely to explain the gap.
- Feature means and scales are fixed before optimization in the downstream runtime, while encoder parameters continue changing. The selected fine-tuned encoders shift the mean by 45.2 and 84.3 training normalization units for three- and six-epoch fits.
- Within-input embedding variance is nearly unchanged (~2.2e-6), so this is not evidence of total encoder representation collapse. Fixed-condition hazard logits vary only about 0.0006–0.0007 across the audited inputs, indicating that the readout largely ignores the remaining structural variation.
- All of the first 128 fine-tuning updates are gradient-clipped: median pre-clipping norms are 147 and 323 for three/six-epoch runs, versus the clipping threshold 5. The initial norm is about 2,042. Scratch shows a related large-offset, heavily-clipped regime.

These observations support a stale-normalization/optimization explanation, but a controlled corrective rerun is needed to establish causality. The current results do not establish that fine-tuning or scratch learning is intrinsically inferior. The original smoke tests established finite gradients, decreasing short-run loss and valid checkpoints; they did not catch this longer-run loss of input sensitivity.

The next corrective comparison should first train the prediction head with the encoder fixed, then unfreeze with a normalization strategy that follows the changing encoder using training data only, a smaller encoder learning rate, and explicit geometry-sensitivity/feature-drift checks. No new training was launched during this results review.

## Artifacts

- [All 103 evaluations](tables/fits.csv)
- [All horizon, timing and spatial metrics](tables/horizons.csv)
- [Paired comparisons](tables/comparisons.csv) and [metric definitions](tables/METRICS.md)
- [Training-only encoder audit](technical/encoder-audit.json)
- [Scaling plot](plots/scaling.png) / [PDF](plots/scaling.pdf)
- Frozen implementation contracts and exact metric/prediction input hashes are retained in `technical/`.
