# How much birth information remains when only the oldest frame is observed?

Requested 30 September 2026. Extend the same matched, crystal-free histories
and the same eleven descriptor/frozen-MACE readouts from eight down to one
observation. Preserve the existing source roles and completed fits.

| Newest frames removed | Frames kept | Span (ps) | Lead before appearance (ps) |
| ---: | ---: | ---: | ---: |
| 0 | 8 | 5.25 | 0.75 |
| 1 | 7 | 4.50 | 1.50 |
| 2 | 6 | 3.75 | 2.25 |
| 3 | 5 | 3.00 | 3.00 |
| 4 | 4 | 2.25 | 3.75 |
| 5 | 3 | 1.50 | 4.50 |
| 6 | 2 | 0.75 | 5.25 |
| 7 | 1 | 0.00 | 6.00 |

All inputs are prefixes of the same original history. At one frame, only its
oldest observation remains. Lead and available history change together; this
does not isolate either independently. Labels are future isolated establishments
and controls remain liquid over the original matched follow-up. First PTM-atom
appearance and established 64-atom clusters are separate reference times.

Primary analysis: 88 treatments on the same 200 held-out histories from 15
births and 11 sources, reusing the original 44 fits and adding 44 fits. Secondary
analysis: five-fold source/ancestry-grouped readout cross-validation inside the
original train population, with 440 additional fits. Use the original selection
and calibration populations separately in every fold. Never fit to or score
original test rows in CV. All arms/endpoints share the same folds and rows.

Frozen MACE pretraining used the original training sources. The CV experiment
therefore measures readout variability conditional on these features, rather
than full encoder transfer. The primary held-out test preserves the stronger
source separation. Report the two populations separately.

Fit by likelihood and select by validation NLL. Measure raw/calibrated NLL,
Brier, AP, AUROC, recall and achieved false-positive rate. Plot all eight leads
with 95% whole-source bootstrap bands, descriptor-control panels and calibration
curves. Pool OOF scores with event weights; show individual fold scores and
fold variability. Bootstrap intervals condition on fitted models and the one
split/fit seed; fold SD is descriptive, not an independent standard error.

[Recipe](../../configs/birth_prediction/drop_to_one_cv_20260930.json) ·
[Definitions](../../docs/metrics/birth_prediction_extension.md) ·
[Execution](../../docs/birth_prediction.md#truncation-to-one-frame-and-readout-cross-validation).

```bash
python -m src.research.birth_prediction.extension prepare --config configs/birth_prediction/drop_to_one_cv_20260930.json
python -m src.research.birth_prediction.extension submit --config configs/birth_prediction/drop_to_one_cv_20260930.json
```
