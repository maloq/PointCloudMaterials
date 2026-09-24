# Improve local-onset ranking while retaining useful encoder dynamics

Status: proposed training experiments; completed analysis of existing exports,
followed by a matched 0.75 ps re-export of Geoformer and current MACE checkpoints.
No new simulation or model training is part of this study.

**Question:** can AP-aligned readout selection/loss, explicit precursor inputs,
and then onset-supervised encoder fine-tuning improve 12 ps local crystallization
AP beyond the current present-geometry encoder?

The [full experiment sequence](../../docs/encoder_research/ap_experiments.md)
specifies the frozen-head screen, ranking objective controls, history/context
and relaxed-input comparisons, then encoder fine-tuning. It fixes natural-risk
populations, ancestry/source splits, AP-based selection, calibrated secondary
metrics and reuse limitations. Start with one seed and cheap cached readouts.

The [completed dimension/stability supplement](../../output/encoder_research/dynamics-20260924/RESULTS.md)
adds raw-state, within-track and temporal-movement spectra for eight current
checkpoints (12 encoder/projector exports) and seven historical dense-trajectory
representations. [Metric definitions](../../docs/metrics/embedding_dynamics.md),
[reproduction guide](../../docs/encoder_research/embedding_dynamics.md),
[configuration](../../configs/analysis/encoder_dynamics_20260924.json).

Interpret AP and stability together: a quiet or low-rank state can discard
precursors, and a responsive predictive state can move during real transitions.
The current four-snapshot dataset is inadequate for short-time jitter/response
measurement; rank ceilings and physical-time gaps must remain visible.

The [0.75 ps update](../../output/encoder_research/dynamics-lag075-20260924/RESULTS.md)
addresses that limitation by applying the same eight recent checkpoints to the
existing dense observed-MD cohort. All models now use identical trajectory rows
and a single 0.75 ps lag; relaxed-trained MACE on observed inputs is identified
as an input-domain transfer. Geoformer encoder/projector states remain separate.

The [controlled input-noise supplement](../../output/encoder_research/noise-lag075-20260924/RESULTS.md)
adds source-balanced responses to Gaussian coordinate perturbations at four
amplitudes, with identical per-atom draws across models. Its primary 0.01 Angstrom
response is compared with natural 0.75 ps movement at matched sampled origins.
Repeated-input and cached-input replay differences are reported separately;
[definitions](../../docs/metrics/embedding_noise.md).
