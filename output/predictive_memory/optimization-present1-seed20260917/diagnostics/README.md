# State-use and physical-readout diagnostics

Replacing the embedding with its training mean keeps the learned head and temperature fixed. Positive NLL increase means removing the state hurts. This intervention is not a retrained condition-only model or a mutual-information estimate.

| Model | Validation NLL increase | Test NLL increase | Test embedding-ridge future MSE |
|---|---:|---:|---:|
| xv-H0 | -0.00115 | 0.00281 | 0.78532 |
| xv-H12 | 0.00325 | 0.00660 | 0.80891 |
| xv-H48 | 0.00992 | 0.03653 | 0.78595 |
| xv-H48-repeat | -0.00185 | -0.00107 | 0.92555 |

| Physical baseline | Validation future MSE | Test future MSE |
|---|---:|---:|
| temperature_ridge | 0.82486 | 0.92770 |
| current_packet_ridge | 0.68995 | 0.74204 |

Ridge penalties use only the middle validation anchor; scaling and coefficients use only training sources. The current-packet baseline includes measured velocities and is observation-matched only to xv, not x. No history-packet baseline is fitted here. All test sources were previously examined; this is exploratory. Source intervals and per-block/per-lag errors are retained in tables and JSON. The saved linear readouts are diagnostic predictors, not replacement encoders.
