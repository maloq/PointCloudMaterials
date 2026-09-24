# Literature-guided frozen-MACE follow-up

One seed; matched archived origins; selection uses integrated Brier through 12 ps. All models retain 96 ps physical/embedding forecasts. Historical test sources are reused: this is an exploratory follow-up, not a fresh confirmatory test.

| Experiment | State | AP 12 ps ↑ | Brier 0.75–12 ps ↓ | Timing MAE, detected (ps) ↓ | Missed / positive |
|---|---|---:|---:|---:|---:|
| cold-control | complete | 0.6457 | 0.01257 | 2.413 | 75 / 226 |
| original-control | complete | 0.5738 | 0.01449 | 2.344 | 92 / 226 |
| cold-rates | complete | 0.6393 | 0.01249 | 2.408 | 73 / 226 |
| cold-dense-repeat | complete | 0.6659 | 0.01245 | 2.356 | 73 / 226 |
| cold-dense-real | complete | 0.6227 | 0.01313 | 2.384 | 82 / 226 |
| cold-dual-repeat | complete | 0.6565 | 0.01266 | 2.527 | 72 / 226 |
| cold-dual-original | complete | 0.6270 | 0.01319 | 2.520 | 74 / 226 |
| cold-quench-difference | complete | 0.6647 | 0.01242 | 2.419 | 71 / 226 |
| cold-dense-rates | complete | 0.6118 | 0.01327 | 2.431 | 79 / 226 |
| dual-dense-rates | complete | 0.6176 | 0.01323 | 2.486 | 69 / 226 |
| dual-dense-rates-short | complete | 0.6113 | 0.01313 | 2.299 | 70 / 226 |
| dual-dense-rates-no-clock | complete | 0.6169 | 0.01322 | 2.483 | 70 / 226 |
