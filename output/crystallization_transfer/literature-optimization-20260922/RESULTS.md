# Literature-guided frozen-MACE follow-up

One seed; matched archived origins; selection uses integrated Brier through 12 ps. All models retain 96 ps physical/embedding forecasts. Historical test sources are reused: this is an exploratory follow-up, not a fresh confirmatory test.

| Experiment | State | AP 12 ps ↑ | Brier 0.75–12 ps ↓ | Timing MAE, detected (ps) ↓ | Missed / positive |
|---|---|---:|---:|---:|---:|
| cold-E48 | complete | 0.6660 | 0.01227 | 2.410 | 71 / 226 |
| quench-E48 | complete | 0.6630 | 0.01233 | 2.406 | 73 / 226 |
| rates-quench-E48 | complete | 0.6459 | 0.01237 | 2.425 | 65 / 226 |
| current-E48 | complete | 0.6716 | 0.01237 | 2.360 | 76 / 226 |
| quench-lr3e5 | complete | 0.6510 | 0.01280 | 2.451 | 75 / 226 |
| cold-lr3e5 | complete | 0.6423 | 0.01294 | 2.468 | 83 / 226 |
| rates-quench-lr3e5 | complete | 0.6459 | 0.01276 | 2.464 | 73 / 226 |
| quench-short025 | complete | 0.6403 | 0.01267 | 2.350 | 81 / 226 |
| quench-short1 | complete | 0.6453 | 0.01224 | 2.315 | 71 / 226 |
| cold-short1 | complete | 0.6475 | 0.01222 | 2.312 | 70 / 226 |
| quench-free | complete | 0.6653 | 0.01233 | 2.334 | 75 / 226 |
| cold-free | complete | 0.6607 | 0.01237 | 2.312 | 70 / 226 |
| quench-wd1e3 | complete | 0.6631 | 0.01233 | 2.406 | 73 / 226 |
| quench-wd1e2 | complete | 0.6630 | 0.01233 | 2.406 | 73 / 226 |
| quench-width64 | complete | 0.6709 | 0.01218 | 2.420 | 71 / 226 |
| quench-lr3e5-E96 | complete | 0.6627 | 0.01242 | 2.403 | 72 / 226 |
