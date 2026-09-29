# Continuous distance: completed results, 26 September 2026

All five readouts completed 16 epochs and their fixed-center/scan evaluations.
The detached queue took about 23 minutes, including feature reconstruction and
exports (09:53:30–10:16:43 UTC). All 25 checkpoint/prediction/index/metric files
listed in completion receipts were checksum-verified for this report. Online
W&B receipts exist for every fit. No further training was launched for the review.

These are new continuous-distance readouts of the same **frozen** observed MACE
encoder. Training received 19,728 additional uniform centers and selection 8,784,
from 1,782 original observation frames; original evaluation rows are unchanged.
Batch and microbatch were 256. One training seed. Selection used the declared
mixture/source-weighted censored likelihood after epoch 12, not test metrics.

## Distance accuracy

Source-weighted results. MAE evaluates the predicted median of min(distance,64 Å);
RMSE evaluates its predicted mean. Empty/far-reference cells are right-censored
in the training likelihood and capped at 64 Å only for these point-error metrics.
The fixed test has 45,291 observations from 30 sources; scans have 15,084 positions
from 28 sources. Scan positions are correlated and sampled by controlled paths.

| Readout | Fixed test NLL ↓ | Fixed MAE (Å) ↓ | Fixed RMSE (Å) ↓ | Scan NLL ↓ | Scan MAE (Å) ↓ | Scan RMSE (Å) ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Local MACE | 2.5866 | 15.89 | 20.21 | 4.5644 | 20.81 | 19.77 |
| Visibility-only label control | 2.2728 | 7.74 | 11.83 | 3.9584 | 15.40 | 13.69 |
| Vector messages | 2.0106 | 8.42 | 13.42 | 4.0617 | 13.51 | 14.36 |
| Harmonic hierarchy | 2.0220 | 8.62 | 13.63 | 4.1608 | 13.65 | 14.46 |
| Symmetric invariant | 2.0266 | 8.41 | 13.51 | 4.1667 | 15.47 | 15.95 |

Vector has the lowest fixed-center likelihood loss among these fits and the
lowest scan MAE point estimate. Symmetric and vector fixed MAE are essentially
tied. The visibility-only control beats all learned geometric treatments on
fixed MAE/RMSE and scan NLL/RMSE, while vector and harmonic beat it on scan MAE.
Consequently, neither geometric context nor visibility is uniformly superior
across proper and point-error scores. No seed/source uncertainty comparison has
been computed for this status report, so small differences are not established.

44.7% of equal-source fixed-test mass is censored at 64 Å, versus 3.0% on scans.
Fixed-center and scan errors therefore describe substantially different
populations. The continuous-density NLL is **not numerically comparable** with
the original categorical distance NLL.

## Confidence alarms

Event: distance ≤20 Å; strict probability exceedance at two consecutive points.
Median distance excludes misses; counts include all 495 test approaches and 292
away paths. All other radii and instantaneous alarms remain in the full exports.

| Readout | Threshold | Median detection distance (Å) | Misses / 495 | False alarms / 292 |
|---|---:|---:|---:|---:|
| Vector | >0.50 | 12.76 | 15 | 17 |
| Vector | >0.75 | 11.93 | 36 | 5 |
| Vector | >0.95 | 8.97 | 62 | 1 |
| Harmonic | >0.50 | 12.62 | 16 | 19 |
| Harmonic | >0.75 | 11.45 | 37 | 8 |
| Harmonic | >0.95 | 7.57 | 70 | 2 |

The earlier categorical vector model at >0.95 had median 8.84 Å, 96 misses and
zero away alarms. The new fit misses fewer approaches at similar warning distance,
with one away alarm. Both the target distribution and training population changed,
so this comparison does not isolate the benefit of continuous regression.

## Interpretation of spatial context

At vector >0.50, reference crystal is already visible in the input at 468/480
alarms (97.5%); at >0.95, it is visible at 432/433 (99.8%). This remains primarily
evidence for recognizing/localizing an existing crystal within the surrounding
patches, not for detecting an unseen nucleus precursor.

The learned visibility control consumes only two reference-membership flags
(local/context). At >0.50 and >0.75 it alarms at median 19.12 Å with no misses or
away false alarms; at >0.95 it waits until median 3.01 Å. Those flags require
full-cell reference segmentation, so this is a label-assisted information control,
not a deployable matched-input competitor. Its strong scores show that this
coarse input retains substantial distance information, but do not prove that
visibility causally explains every learned feature or all likelihood gains.

## Artifacts

[Run and logs](/work/PERSO/vmorozov/analysis/spatial_distance/al64-uniform-20260926/README.md).
Each model directory contains `analyses/distance-v1/tables/distance.csv` and
`analyses/confidence-v1/tables/`, with frozen definitions and implementation hashes.
Original checkpoints and predictions remain under the model's `technical/`.

- [Vector W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/dd21c3d8596c9c944bbf)
- [Harmonic W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/ff1616e5a6e954c0b994)
- [Symmetric W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/b0cf87e0d828de768bcc)
- [Local W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/d47b2853270f6f30ac2d)
- [Visibility control W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/3f53deb5ed99e14a7e6f)
