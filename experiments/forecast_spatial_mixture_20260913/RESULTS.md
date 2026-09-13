# Initial results — 2026-09-13, 17:50 UTC

Only the first seed (20260913) of the two deterministic controls is complete.
All metrics below are newly computed on common forecast origins and the same held-out
sources/atoms. Probabilistic and spatial fits remain active or queued.

| History | Embedding MSE | Transition precision | Recall | F1 | Timing MAE, detected events | All-event recall timed within 1.5 ps |
| --- | --- | --- | --- | --- | --- | --- |
| 6 ps | 0.208395 | 49.24% | 51.71% | 50.45% | 2.264 ps | 26.18% |
| 12 ps | 0.206928 | 47.76% | 54.44% | 50.88% | 2.237 ps | 28.05% |

The physical target is a first local episode of three consecutive crystalline frames,
within the next 9 ps. F1-optimal thresholds were selected on validation sources
separately for each fit. Higher recall is accompanied by lower precision. This first
seed suggests a small gain, with almost unchanged conditional timing error. It does
not establish a robust improvement; the second seed and paired source comparison
are pending. Timing MAE excludes missed transitions; the final column includes them.

Full values, source-bootstrap intervals, state scores, 3/6/9 ps horizons and nine-frame
persistence sensitivity are under:

- [6 ps local results](../../output/embedding_forecast/context-space-mixture-20260913/technical/local/history6_deterministic-seed20260913/results.json)
- [12 ps local results](../../output/embedding_forecast/context-space-mixture-20260913/technical/local/history12_deterministic-seed20260913/results.json)

Reproduce these using the maintained physical-evaluation commands in [README](README.md).
No older metric export was modified. Final machine-readable comparison tables and
frozen definitions are produced only after all 12 fits/assays complete.
