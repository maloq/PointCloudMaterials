# Smooth spatial and temporal encoder pilot — September 5, 2026

Research experiment record. Shared representation implementation is in
[`smooth_density.py`](../../src/models/encoders/smooth_density.py); learned causal
state models are in [`smooth_state.py`](../../src/temporal_vamp/smooth_state.py).
These recipes are specific to the verified Al/Mg/Ta source manifest and this
pilot. They are not new maintained command-line tools.

Question: does removing discrete geometry recover continuity while preserving
structure, and does motion-transported memory improve the smoothness/prediction
tradeoff beyond ordinary history?

Motivation and literature:
[`geoframe_v3_literature_20260905.md`](../../docs/geoframe_v3_literature_20260905.md).
Configuration: [`config.json`](config.json). Outputs, including checkpoints,
arrays, plots, logs, and the final report, are physically inside
[`output/smooth_temporal_encoder_20260905`](../../output/smooth_temporal_encoder_20260905/).

Completed results: [report and figures](../../output/smooth_temporal_encoder_20260905/RESULTS.md).
Follow-up: [why the MACE-product pilot transfers poorly](../../output/mace_diagnosis_20260905/RESULTS.md),
including matched objective and readout controls.
Six spatial and twelve temporal runs finished. Smooth geometry resolves the
selected canonicalization jumps; ordinary GRU memory gives the strongest
smoothness/structure tradeoff in this pilot. Transport does not improve prediction
over otherwise matched gated memory. Static-Al transfer of the learned models
remains weaker than GFv2, motivating mixed static/MD training before replacement.

## Protocol

- Six Al, six Mg, and one Ta continuation; Ta has six times as many centers per
  branch for equal material counts. Zr is excluded.
- Training: 2.0–14.0 ps, 139,392 observations. Validation: 15.0–18.0 ps, 8,928
  observations. Test: 20.0–24.0 ps, 23,616 observations. Center IDs are disjoint
  between splits. Sources are shared; this is not independent-source validation.
- Every cloud includes all contributing atoms within 0.75 times the original
  per-material normalization radius. The 192-neighbor storage capacity has a
  checked positive margin outside support, including the bounded jitter margin.
  No nonzero radius contribution is truncated. Central atoms are excluded from
  the density and retained explicitly when reconstructing the GFv2 reference.
- Eight Gaussian radial channels, angular orders 0–6, a C2 cutoff envelope, and
  fixed density normalization. The untrained power spectrum is SOAP-like; it is
  not a DScribe SOAP computation.
- Learned power-spectrum MLP and central MACE product-basis model: 40 epochs,
  batch 4096, three seeds, VICReg directly on the exported 128D output. Fixed
  bounded coordinate-jitter views and spatial-neighbor views have weights 1 and
  0.25. No temporal attraction or PTM labels enter spatial representation training.
  The MACE model uses the reference symmetric contraction, correlation order 3,
  and angular orders through 4, plus the full power-spectrum branch. It is a
  strictly local model, not a complete multi-layer MACE message-passing network.
- Temporal comparisons observe five frames spanning 0.4 ps. Direct forecasts at
  0.1/0.5/1/2 ps share one fixed target: the first 32 training-fit density-power
  PCA coordinates, standardized using training data. All horizons use the same
  origins and remain inside their split. A state never reads future positions.
- Fixed exponential memory before/after invariant compression, with no motion,
  Kabsch motion, or smooth tensor transport. Time constants and transport
  regularization are chosen using validation. A five-frame initialization limits
  available memory even when the nominal decay time is longer.
- Ordinary GRU, learned gated density memory without transport, with Kabsch, and
  with smooth transport: three seeds each. Common recurrent/state/forecast heads;
  a 32D exported state and additional 64D recurrent memory. Future prediction,
  reconstruction of eight current target coordinates, state variance/covariance,
  and a within-window state-update penalty define training. This penalty is not
  presented as a proof of smoothness between independently initialized windows.
- This pilot evaluates direct forecasts; it does not establish autonomous
  multi-step latent simulation. The forecast head also uses the recurrent memory,
  not only the 32D exported structural state.

## Independent checks

Replays the existing selected GFv2 switch boundaries, including float32 Ta.
Independent crystal labels use OVITO PTM at RMSD cutoff 0.15. A best local affine
fit produces a weighted nonaffine residual in Å². These assays do not enter the
representation loss. Readout probes fit training labels and select their ridge
coefficient using validation labels. They measure agreement with a structural
assay, not comprehensive physical accuracy.

The initial PTM call incorrectly used `only_selected=True`, which excludes the
unselected neighbors from analysis. An ideal FCC control exposed this. The assay
was corrected and recomputed for every observation; model training was unaffected.
The correction and original logs are retained. A regression check now covers it.

The CUDA batched eigensolver rejected the 46,464-matrix Ta training shard. Motion
solves are explicitly batched in groups of 4096. The original failure status is
retained, and the affected history/forecast stages were rerun.

Full static Al evaluation rebuilds radius neighborhoods at all 772,953 previously
saved analysis centers. All rows are encoded and used to fit seven spherical
clusters; the repository's sampled clustering diagnostics are reported alongside
full-data PTM readouts. GFv2 is recomputed as a raw encoder on matched nearest-80
clouds. Earlier reports used projector outputs and per-snapshot normalization;
their numbers must not be treated as identical-protocol comparisons.

Al/Mg MD positions retain global float16 quantization. The static Al snapshots
and Ta continuation provide different precision/protocol checks. The new learned
spatial models start from scratch on the MD pilot; GFv2 already had substantial
static pretraining. Full-static transfer scores therefore compare these actual
models, not equal-data architecture ceilings.

## Reproduction

Use `pointnet`, from the repository root. Choose a fresh `output` in the config
for a complete rerun; preparation and training refuse to replace prior runs.

```bash
python experiments/smooth_temporal_encoder_20260905/prepare.py --config experiments/smooth_temporal_encoder_20260905/config.json
python experiments/smooth_temporal_encoder_20260905/run.py --config experiments/smooth_temporal_encoder_20260905/config.json --stage spatial
python experiments/smooth_temporal_encoder_20260905/prepare.py --config experiments/smooth_temporal_encoder_20260905/config.json --labels-only
python experiments/smooth_temporal_encoder_20260905/evaluate.py --config experiments/smooth_temporal_encoder_20260905/config.json --stage all
python experiments/smooth_temporal_encoder_20260905/full_static.py --config experiments/smooth_temporal_encoder_20260905/config.json
python experiments/smooth_temporal_encoder_20260905/train_temporal.py --config experiments/smooth_temporal_encoder_20260905/config.json
python experiments/smooth_temporal_encoder_20260905/rank_controls.py --config experiments/smooth_temporal_encoder_20260905/config.json
python experiments/smooth_temporal_encoder_20260905/assess.py --config experiments/smooth_temporal_encoder_20260905/config.json
python -m pytest tests/test_smooth_density.py -q
python experiments/smooth_temporal_encoder_20260905/summarize.py --config experiments/smooth_temporal_encoder_20260905/config.json
```

The `--labels-only` command records a completed assay-verification stage and can
recompute labels without rebuilding geometry. It is retained to reproduce the
corrected September 5 run. Detached launches, PIDs, and exact commands are saved
under the output directory. Generated data belong there, not in this record.
