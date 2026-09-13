# Matched observed-history forecast comparison — 2026-09-13

This exporter compares the configured 0, 1.5, 3, 6, 12 and 24 ps histories on
identical 0.75–9 ps future paths. All fits share the same frozen embedding cache,
training normalization, independent source splits, atom IDs, forecast anchor
frames, augmentation rates, optimizer schedule and update budget. Parameters are
constant across history lengths within each architecture; direct and AR models
have different parameter counts. History length is the elapsed span; the anchor
is included, giving 1, 3, 5, 9, 17 and 33 observed embeddings respectively.

The underlying errors are defined in `forecast.md`: standardized path MSE, raw
MSE, per-frame error, separate (0,3], (3,6], (6,9] ps bin errors, increment error,
and persistence/history-mean baseline errors. All model evaluation uses predicted
rollouts, with no observed future inputs. This exporter averages each retained
per-window error within source using float64 accumulation, then equally weights
sources and fitted seeds. Time coordinates and bins are not pooled before their
individual scores are exported. `mse_0p75ps` and `mse_9ps` are the first and last
future-frame errors. `raw_mse` uses squared encoder-output units.

`seed_std` is the population standard deviation (`ddof=0`) of each fitted seed's
source-mean path MSE. Plot error bars show this spread, not confidence intervals.
It is undefined for a single seed. Counts, history frames, parameter counts and
selected epochs are metadata; selected epochs are zero-based in provenance.

`gain_vs_anchor` is one minus the ratio of the candidate path MSE to the trained
0 ps history model's path MSE, within the same architecture. Average errors across
fitted seeds before drawing 2,000 paired whole-source bootstrap replicates with
seed 20260913. `gain_vs_anchor_ci95_lower/upper` are the 2.5/97.5 percentiles.
`gain_vs_history_mean` uses the observed mean baseline for that context; its paired
interval is retained in `technical/comparison.json`. Gains are fractions in CSV
and percentages in plots/Markdown. Undefined ratios or intervals remain null.

`reverse_past_mse` and `repeat_anchor_mse` equally average the existing
intervention's source MSEs across sources and seeds. Reversal keeps the anchor
fixed; repetition replaces all history with the anchor. These measure sensitivity
to changed inputs, not the effect of separately retraining on those inputs.

The 27 independent test sources have been examined previously. These comparisons
and bootstrap intervals are exploratory and conditional on two fitted seeds.
Atoms/windows are not independent replicates. Improvement over a trained anchor
includes history averaging/denoising, and does not by itself prove that temporal
ordering is useful. Longer-context conclusions are conditional on this compact
model, common 24–576 ps anchors and 16-epoch training budget.

Each export verifies exact fitted configurations, row identities, cache/source
implementation hashes, normalization and equal parameter counts across histories
within architecture. Per-fit checkpoint hashes are retained in the JSON report.
Historical forecast tables and their exported definitions are preserved.

The 2026-09-13 spatial/mixture extension of the shared forecast evaluator leaves this
historical context comparison's deterministic point-error and source-pairing formulas
unchanged. Original exported contracts stay frozen. New probabilistic mixture scores
are defined separately in `forecast.md` and are not retrospective scores of this sweep.

Host-resident validation is an execution option for newer spatial fits; the historical
context pilot retains its frozen device-resident implementation and metric definitions.
