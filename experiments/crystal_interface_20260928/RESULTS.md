# Crystal-set and interface localization: completed results, 28 September 2026

All six scientific fits completed 16 epochs / 3,968 updates, and all associated evaluations completed. Each uses one seed and random batch/microbatch 256. W&B training records received final evaluation summaries. The figures below are retained held-out evaluation metrics, not the final training minibatch or last validation epoch. All exported table hashes were checked against their completion receipts when preparing this report.

## Interface task: CIV-MACE128

The new target is distance to the crystal-side interface atom layer, including positive interior distances. The layer and its limitations are defined in [the protocol](README.md). Fixed source roles remain 90/15/15/30. The new release contains 204,857 contexts, including 18,224 uniform test centers from 30 sources. There are 4,669 interior test centers in 27 sources and 1,407 test centers on the interface layer. Interior scores below contain no censored distances. These are correlated observations, not independent trajectories.

| Treatment | Uniform-test distance NLL ↓ | Uniform RMSE (Å) ↓ | Interior NLL ↓ | Interior RMSE (Å) ↓ | Interior median-predictor MAE (Å) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distance only | 2.068 | 10.693 | 2.312 | 3.181 | 2.249 |
| Distance + direction | 2.048 | 10.786 | 2.247 | 3.037 | 2.129 |
| Distance + direction + VCReg | 2.053 | 10.768 | 2.258 | 3.036 | 2.122 |

Direction supervision lowers interior RMSE by approximately 4.5% relative to distance-only. Its distance likelihood also improves inside and across uniform test centers, but overall point RMSE does not improve. VCReg adds little predictive improvement at this strength/budget: its interior RMSE differs from the directional control by only 0.002 Å and its NLL is slightly worse. These are single-seed point estimates; no uncertainty interval or seed-robust ranking is established.

The task remains imperfect. The true distance on the interface layer is zero, yet the directional+VCReg model has 3.54 Å RMSE there. Uniform exterior RMSE is 12.82 Å. Overall uniform-test point errors include 32.6% source-weighted censoring at 64 Å and should not be described as uncensored physical-distance accuracy.

For the primary directional+VCReg model, mean interior direction error is 60.2° at 0–8 Å, 55.8° at 8–16 Å and 71.5° at 16–32 Å; no interior samples exceed 32 Å in this test subset. On the fixed exterior benchmark, mean direction error is 32.8° at 8–16 Å, 69.6° at 16–32 Å and 89.9° at 32–64 Å. Near-interface atomwise nearest-point directions are relatively difficult; distant direction is close to the 90° isotropic reference.

## Interface warning distance

Primary predeclared distance+direction+VCReg model; probability means P(interface distance <=20 Å). A warning requires two consecutive spatial observations strictly above the threshold, before entering crystal. Median warning distance is conditional on detection. All-path denominators retain misses.

| Probability threshold | Detected / 495 paths | Misses | Median warning distance (Å) | Recall while >=12 Å away | Recall while >=20 Å away |
| --- | ---: | ---: | ---: | ---: | ---: |
| >0.5 | 474/495 | 21 | 11.07 | 38.0% | 2.8% |
| >0.75 | 446/495 | 49 | 8.94 | 12.3% | 0.6% |
| >0.95 | 331/495 | 164 | 5.61 | 0.4% | 0.0% |

At threshold >0.5, 466/474 detected paths already have interface atoms visible somewhere in the 32-Å input context at alarm. The remaining eight alarms alone do not establish robust non-visible-interface inference. At >0.75 and >0.95 the visible counts are 443/446 and 331/331. This is primarily local geometric localization in the present snapshot; it is not evidence of long-lead crystallization prediction. On 292 historical away paths, alarm counts are 7, 1, and 0 respectively. Under the changed target these are away-path alarm rates, not automatically proven false positives.

## Original crystal-set task: CDV-MACE128

45,291 fixed-at-risk test queries from 30 sources; all query centers are outside confirmed crystal. Metrics below share the original crystal-set target and must not be compared numerically to interface NLL as if the targets were identical.

| Treatment | Distance NLL ↓ | Capped-mean RMSE (Å) ↓ | Capped-median MAE (Å) ↓ | Brier within 20 Å ↓ |
| --- | ---: | ---: | ---: | ---: |
| Distance only | 2.050 | 13.179 | 8.332 | 0.03230 |
| Distance + direction | 2.128 | 13.261 | 8.514 | 0.03227 |
| Distance + direction + VCReg | 2.126 | 13.273 | 8.537 | 0.03229 |

Distance-only has the best distance likelihood and point errors on this fixed test track. Directional supervision supplies direction estimates but does not improve its distance accuracy. The primary VCReg arm warns at median distances 10.80 / 8.71 / 5.38 Å for thresholds >0.5 / >0.75 / >0.95 on P(d<=20 Å), detecting 487 / 454 / 361 of 495 approach paths.

## Exported embedding behavior

Values below use CIV-MACE128 on the fixed held-out liquid-at-risk track, with 512 matched observations for response diagnostics. Normalized response is RMS embedding displacement divided by sqrt(2 trace(Cov(z))) on that track. The noise amplitude is three-dimensional coordinate RMS equal to 1% of the query neighborhood mean nearest-neighbor distance. Temporal lag is exactly 0.75 ps between independent snapshot encodings; it is not an encoder history input.

| Treatment | Local embedding d95 | Local effective rank | Movement d95 at 0.75 ps | Normalized temporal RMS ↓ | Normalized 1%-noise response ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distance only | 14 | 11.30 | 15 | 0.735 | 0.096 |
| Distance + direction | 14 | 11.38 | 15 | 0.734 | 0.096 |
| Distance + direction + VCReg | 14 | 11.43 | 15 | 0.742 | 0.088 |

VCReg slightly reduces the measured 1%-noise response but does not improve temporal stability in this run. The local embedding remains far above the earlier 0.10 temporal-jump aspiration. For the primary model, the context state has d95=2 on fixed test and uniform test populations, compared with local d95=14 and 11 respectively. This indicates concentrated variance, not proof that the representations are sufficient for other physical tasks or forecasts. Context temporal response is 0.145 and its 1%-noise response is 0.118. No new 3/6-ps AP measurement is supplied by these snapshot-localization experiments.

## Interpretation

The interface target is operational and its interior distances are learnable to roughly 3 Å RMSE on this held-out population. Direction supervision helps this interior task modestly. The present VCReg strength does not produce a clear gain in predictive skill, temporal stability, or representation breadth. Useful directions are localized to the observed neighborhood; long-range warnings mostly coincide with a boundary already entering spatial context. A next comparison should keep a common target/readout and separately test larger context or the current nearest-atom direction definition, rather than rank encoders across different target spaces. No further experiments were launched during this results review.

## Retained evidence

- crystal_interface, Distance only: [tables](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_only/analyses/localization-v1/tables/distance.csv), [frozen metric definitions](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_only/analyses/localization-v1/tables/METRICS.md), [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/08fb1ad0f91a33d85a74).
- crystal_interface, Distance + direction: [tables](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_direction/analyses/localization-v1/tables/distance.csv), [frozen metric definitions](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_direction/analyses/localization-v1/tables/METRICS.md), [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/d50391c45b0ffd6cc617).
- crystal_interface, Distance + direction + VCReg: [tables](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_direction_vcreg/analyses/localization-v1/tables/distance.csv), [frozen metric definitions](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_direction_vcreg/analyses/localization-v1/tables/METRICS.md), [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/3934db07a5e9b90f7eab).
- crystal_vector, Distance only: [tables](/work/PERSO/vmorozov/analysis/crystal_vector/al64-random-20260928/distance_only/analyses/localization-v1/tables/distance.csv), [frozen metric definitions](/work/PERSO/vmorozov/analysis/crystal_vector/al64-random-20260928/distance_only/analyses/localization-v1/tables/METRICS.md), [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/0fbf2a868839b05a95c0).
- crystal_vector, Distance + direction: [tables](/work/PERSO/vmorozov/analysis/crystal_vector/al64-random-20260928/distance_direction/analyses/localization-v1/tables/distance.csv), [frozen metric definitions](/work/PERSO/vmorozov/analysis/crystal_vector/al64-random-20260928/distance_direction/analyses/localization-v1/tables/METRICS.md), [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/75059c28c8a4b35d79fd).
- crystal_vector, Distance + direction + VCReg: [tables](/work/PERSO/vmorozov/analysis/crystal_vector/al64-random-20260928/distance_direction_vcreg/analyses/localization-v1/tables/distance.csv), [frozen metric definitions](/work/PERSO/vmorozov/analysis/crystal_vector/al64-random-20260928/distance_direction_vcreg/analyses/localization-v1/tables/METRICS.md), [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/8b10b5efda78a4882335).
