# Literature-guided frozen-MACE follow-up

One seed; matched archived origins; selection uses integrated Brier through 12 ps. All models retain 96 ps physical/embedding forecasts. Historical test sources are reused: this is an exploratory follow-up, not a fresh confirmatory test.

| Experiment | State | AP 12 ps ↑ | Brier 0.75–12 ps ↓ | Timing MAE, detected (ps) ↓ | Missed / positive |
|---|---|---:|---:|---:|---:|
| front-control | complete | 0.6660 | 0.01227 | 2.410 | 71 / 226 |
| front-quench-control | complete | 0.6630 | 0.01233 | 2.406 | 73 / 226 |
| front-r12 | complete | 0.6839 | 0.01196 | 2.355 | 60 / 226 |
| front-r20 | complete | 0.6686 | 0.01226 | 2.348 | 63 / 226 |
| front-r25 | complete | 0.6850 | 0.01208 | 2.349 | 58 / 226 |
| quench-front25 | complete | 0.6771 | 0.01228 | 2.402 | 59 / 226 |
| front25-repeat | complete | 0.6787 | 0.01222 | 2.351 | 58 / 226 |
| front25-rates | complete | 0.6942 | 0.01195 | 2.346 | 53 / 226 |
| quench-front25-rates | complete | 0.6808 | 0.01214 | 2.396 | 65 / 226 |
| front25-coherence | complete | 0.6645 | 0.01233 | 2.357 | 67 / 226 |
| front25-geometry | complete | 0.6668 | 0.01228 | 2.412 | 68 / 226 |
| quench-front-multiscale | complete | 0.6736 | 0.01223 | 2.378 | 58 / 226 |
