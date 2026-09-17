# State-use and physical-readout diagnostics

Replacing the embedding with its training mean keeps the learned head and temperature fixed. Positive NLL increase means removing the state hurts. This intervention is not a retrained condition-only model or a mutual-information estimate.

| Model | Validation NLL increase | Test NLL increase | Test embedding-ridge future MSE |
|---|---:|---:|---:|
| xv-H0 | -0.00095 | -0.00040 | 0.92699 |
| xv-H12 | -0.00052 | -0.00063 | 0.90238 |
| xv-H48 | -0.00083 | -0.00070 | 0.85650 |
| xv-H48-repeat | -0.00318 | -0.00200 | 0.90821 |

| Physical baseline | Validation future MSE | Test future MSE |
|---|---:|---:|
| temperature_ridge | 0.82486 | 0.92770 |
| current_packet_ridge | 0.68995 | 0.74204 |

Ridge penalties use only the middle validation anchor; scaling and coefficients use only training sources. The current-packet baseline includes measured velocities and is observation-matched only to xv, not x. No history-packet baseline is fitted here. All test sources were previously examined; this is exploratory. Source intervals and per-block/per-lag errors are retained in tables and JSON. The saved linear readouts are diagnostic predictors, not replacement encoders.
