# Does predicting structural evolution improve local crystallization forecasts?

Compare direct onset hazards against explicit structural-state trajectory models,
using the same 150 independent Al sources and 90/15/15/30 split, 109,838 training
windows, 44,385 test windows and held-out calibration thresholds. One seed. This
extension initially freezes the pretrained MACE encoder to isolate the forecasting
method; it is not an additional fine-tuning or scratch encoder comparison.

The companion queue freezes the best **completed scalar frozen-encoder** settings
by selection NLL at submission, recording all provenance in reference-selection.json.
At implementation this is H12/R25, factorized spatial/temporal attention, width128,
four heads, one context block per axis, LR5e-4, weight decay1e-4 and batch64. These
are a supported starting point, not established optimal generative-model settings.
No settings are chosen using test timing or test NLL. Only the context settings
are transferred; forecast heads train from scratch. All use one warmup epoch and
cosine LR decay to 5% of peak.

Run each method for 12 and 24 epochs, with independent schedules:

| Method | Forecast |
|---|---|
| Direct | One complete deterministic structural path plus dense onset hazards |
| Autoregressive MSE | GRU feeds its predicted structural state into the next step |
| Autoregressive Gaussian | GRU generates alternative futures with diagonal conditional noise |
| Whole-path mixture | Four components for complete paths and onset distributions |
| Conditional diffusion | Jointly denoise complete structural/onset paths |

All forecast 32 future structural states through96ps; no rollout gets real future
frames. Autoregressive training gradually removes teacher forcing. Dense 0.75ps
onset supervision uses the same sustained-event labels as the original assay,
including complete follow-up for confirmation. Future context geometry is not
predicted or supplied. We forecast the tracked center's state, not atomic coordinates.
Physical targets anchor comparison across trajectory methods and include motion even
though this initial MACE input is position-only. No additional TDA or simulations.

The diffusion pilot follows conditional noise-prediction training from
[DDPM](https://arxiv.org/abs/2006.11239), with
[DDIM](https://arxiv.org/abs/2010.02502) sampling. A lightweight future-time transformer
conditions on the observed spatial/temporal context. Its learned absorbing onset
channel is generated alongside the physical/embedding trajectory; it is not an
independent direct hazard head. A useful result must improve held-out timing,
miss-aware event scores and physical prediction, not simply produce smooth paths.

Primary selection: dense event integrated Brier score on selection sources. Report
original six-horizon classification, event NLL, misses, timing, alarm and spatial
metrics; additionally dense-grid timing, restricted-mean-time error including misses,
standardized physical/embedding/bond-order errors and marginal CRPS. Keep the original
coarse timing calculation alongside finer timing to avoid attributing a discretization
change to improved predictive information. Compare to physical persistence and the
existing frozen direct-hazard reference. Report one-seed limitations and distinguish
natural window prevalence from independent event counts.

See [metric definitions](../../docs/metrics/crystallization_paths.md) and
[execution](../../docs/crystallization_paths_20260919.md).
