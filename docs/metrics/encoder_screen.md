# Native snapshot encoder screen, September 23, 2026

This extends the **unchanged calculations** of `geoframe_evolution` to pinned native
snapshot exports. `tables/<model>.csv` is flattened numerical output; full arrays,
confusions and eligibility counts remain in each `technical/evaluations/<model>/metrics.json`.
`summary.csv` contains arithmetic means across the same three Al frames, separately
for encoder/projector. Means use defined frame values; class counts and undefined
frames must be inspected in the detailed tables. They are not source confidence
intervals. All 37 existing GeoFrame evaluations are reused with source links.

## Fixed references and probes

The existing hashed GeoFrame reference supplies 4096 anchors and their nearest
neighbors per snapshot, 1024 topology anchors, fitting/test spatial halves with
an excluded gap, three Al/Ta/Zr snapshots, full-cell PTM at RMSD 0.08/0.10/0.12,
Al fault types, q4/q6/normalized w4/w6/averaged q6/coherence/density, and fitting-only
material thresholds. Read `geoframe_evolution.md` for exact label precedence and
neighbor counts. Unknown static potentials remain unknown. The original GeoFrame
recipe contains these frames: this is a transductive encoder screen, not an
unseen-material or independent-trajectory generalization test. Other models have
different training populations; the reference probes themselves use fixed rows.

Unchanged `frame_metrics` supplies standardized logistic probes (C1, balanced,
max_iter2000), per-class AP with fitting support >=20, cut-off sensitivity,
within-liquid ridge10 continuous-order/topology fidelity, density-conditional
R² gains, participation ratio, KMeans K7/n_init5, test cluster-context counts,
nonbulk adjusted mutual information, and nearest-neighbor boundary AUROC with
physical-distance matching, shuffled and collapsed controls. It reuses the exact
calculation, not a faster approximate clustering/probe. CPU BLAS/OpenMP thread
counts are bounded to one. Floating reduction differences across thread counts
are measured by the separate benchmark; historical exports are unchanged.

## Native observations

Geometry MACE uses physical radius 8Å, cutoff 5Å, original GraphBank and trained
calibration buffers. Neighborhood JEPA uses the original scaled radius 8 local
observation. Paired-relaxed models use their original nearest 80 candidate cap
before radius cropping. Shared MACE/GATr use the pinned producer's local radius 8
and material scale. GeoFrame uses its original nearest 80/material radius. All
are **snapshot** models; history/velocity/full-halo models are excluded explicitly.
Native support, species, precision and checkpoint identity are per-task fields.
Shared MACE retains the training compiler; GATr and neighborhood exports follow
their verified eager static pathways. Geometry MACE is native eager float32.
These differences are part of the model comparison; they do not isolate loss or
architecture. Hot-trained models receive the common relaxed assay as a labelled
domain-transfer test. No further relaxation is performed.

The new perturbation test displaces up to 256 fixed candidate patches, holds the
tracked center fixed, and rebuilds cutoff edges/tapered support. Per-coordinate
Gaussian amplitudes are 1e-4, .01, .1Å. Median/p95 embedding displacement is divided
by the unperturbed patch population's pair RMS. Candidate membership is fixed;
this differs from the older GeoFrame perturbation which also displaced its center.
Keep the two fields separate and do not combine their magnitudes into a ranking.
Repeated-batch, singleton and reorder errors are retained in smoke receipts;
BF16 batch effects are visible and must not be interpreted as physical response.

## Conditional future assay

The same 2880 relaxed-present patches, 45 ancestry roots (25 fit/5 tune/15 reused
development), original-MD future targets and scaler/calibration rules are used.
The 9ps future-order residual is conditioned on current order8, relaxed geometry89,
temperature, time and time², following `structural_state.dynamics.targets`. Ridge10
from fit-standardized encoder features predicts that residual. MSE averages within
root then across roots. A zero-residual prediction is the baseline.

The same 1024-update linear five-bin hazard adds temperature/current order8/relaxed
geometry89 conditions, with fit-only scaling and tuning-NLL checkpoint selection,
including a true constant-risk step0 candidate. Report AP, source-balanced Brier,
NLL, calibration, timing plus misses, selected update, and coarse 108/120ps
same-atom physical-change response. This is not the older temperature-only hazard
or a short-lag thermal-continuity measurement.

All exports here have 128 features per stage, allowing reuse of the already
computed 128-zero-feature current-physics hazard baseline. Compare exact hazard
row/source/event identities before taking differences. At 12 ps calculate per-root
Brier(model)-Brier(current physics); for 9 ps calculate per-root residual MSE minus
zero baseline. `paired_mean_ci` resamples the 15 paired roots 2000 times, seed 20260923,
then reports percentile95% intervals. Intervals condition on the fitted encoder,
readout, target calibration and development split; they do not cover seed variation
or repeated model selection. Positive AP alone is not a predictive success gate.

## Provenance, caching and reporting

Checkpoints are pinned by file SHA256; tensor-identical encoders are deduplicated.
Native inference runs in the named frozen producer checkout with critical producer
hashes checked and strict state loading. Full source neighborhoods are prepared
once from verified snapshot bytes and cached independently of weights. Embeddings
and physical references are retained so plotting and CPU readouts do not repeat
inference. Output `complete.json` appears only after numerical metrics and saved
predictions are written. Failed and pending models remain visible. Plotting is a
separate stage and cannot make an incomplete evaluation appear successful.

## Additional dynamics diagnostics (September 24)

New evaluations append an `embedding_dynamics` block from the shared predictor.
It reports source-balanced state, within-track, movement and fluctuation spectra
and exact-lag jumps. The four-snapshot assay only supports 108/120 ps stability;
individual state ranks are capped at three. See [the full definitions](embedding_dynamics.md).
The existing static, onset and residual calculations are unchanged; historical
exported metric definitions and frozen-code evaluations remain unchanged.
