# Descriptor-free temporal hypotheses — September 6, 2026

Question: can more temporal data, predictive objectives and a complete larger
MACE neighborhood produce smooth, useful embeddings without training against
TDA, SOAP or bond-order descriptors?

Findings: all 33 training trials completed. Predictive density gives the strongest
overall results; MACE improves over its previous trained version. Temporal
invariance learns a useful but very low-dimensional representation, and GeoFrame
remains sensitive to small perturbations. The interrupted final evaluation was
replayed on September 6 without training or reselection. See the
[detailed review](../../docs/temporal_hypotheses_review_20260906.md),
[full results](../../output/temporal_hypotheses_12h_20260906/RESULTS.md) and
[equal-budget screening table](../../output/temporal_hypotheses_12h_20260906/screen_analysis/comparison.csv).

## Twelve-hour schedule

The clock starts with the detached campaign. Cache preparation may use up to
one hour. Screening allocates **15 minutes to each of 24 trials**: twelve
hypotheses, each at learning rates 0.0003 and 0.001. The best validation candidate
from each encoder family is identified; the strongest three families receive
longer confirmation with **three fresh seeds each** (123, 456, 789).

Confirmation uses equal wall-clock budgets for its nine runs, calculated from
the time remaining after preparation and screening. The final 30 minutes are
reserved for evaluation and reports, with additional startup headroom. Thus
there are **33 scheduled training runs**, plus a separate short preflight.
These are explicit compute budgets, not claims that every model converged.
Step counts, examples seen, validation curves and stop reasons are retained.

| Hypothesis | Purpose |
|---|---|
| density_static | Same-frame jitter VICReg baseline with direct latent regularization |
| density_temporal | Multi-lag temporal invariance versus predicting the same future pairs |
| density_predictive | Multi-lag prediction of a frozen-gradient EMA future embedding |
| density_motion | Add physical nonaffine-motion and displacement gradients to the encoder |
| density_smooth | Add explicit sensitivity regularization under 0.001 Å perturbations |
| density_small_data | Match predictive training using 1/16 of uniformly selected center IDs |
| mace_static | Reference MACE with the same direct-latent static objective |
| mace_predictive | Reference MACE with multi-lag future-embedding prediction |
| mace_wide_halo | Increase MACE edge cutoff from 4 to 6 Å with a complete 12 Å halo |
| mace_capacity | Increase MACE channels from 64 to 128 at fixed cutoff |
| schnet_predictive | Continuous-filter scalar message-passing control |
| geoframe_predictive | GeoFrame control, retaining its canonicalization limitations |

All learned encoders start from scratch. MACE uses the corrected fused `ir_mul`
backend, two interactions, ell=2 and correlation=3. Its neighbor normalization
is measured on training inputs for each cutoff. The larger patch contains the
center plus 768 neighbors; preparation verifies that the complete two-hop
support survives bounded coordinate jitter. Self edges are excluded by identity,
because floating-point `cdist` diagonals need not be exactly zero.

## Training signal and selection

TDA, SOAP, bond-order and PTM targets are absent from encoder training and
checkpoint selection. The density architecture retains its invariant density
input; descriptor-free here refers to supervision, not absence of architectural
inductive bias. Ordinary branches supply equal-material temporal pairs;
shooting branches supply position-conditioned and within-branch future pairs.

Static/temporal controls use direct-latent VICReg terms. Predictive variants
predict a stop-gradient EMA representation conditioned on physical lag and
the shooting temperature. EMA decay is 0.995. The variance/covariance penalty
removes each material's mean first, so species separation cannot satisfy it.
All objectives and motion probes see the same lag distribution; the temporal
control is not restricted to short lags.
The shooting-temperature feature is used for Al; it is deliberately inactive
for the ordinary Mg/Ta trajectories, not an assertion of zero Kelvin.

Every run also fits the same two-output physical-motion probe on
`log1p(D2min)` and `log1p(MSD)`. Its input embedding is **detached**, except in
the explicit `density_motion` ablation. D2min tracks the same 24 neighbor IDs.
Motion targets are standardized using training data within each material.
Checkpoint and learning-rate selection minimize equal-material validation motion
MSE on fixed draws. No test metric influences selection or seed confirmation.

## Data and independence

The cache contains approximately **6.14 million local clouds**, of which
**5.64 million are training clouds**, representing approximately **1.19 million
distinct training input states** and 6.03 million available temporal pairs.
These counts include correlated snapshots and sibling trajectories; they do
not count as independent simulations.

Al training uses 2,048 centers from each of 22 parents, eight shooting branches,
four anchor times and four lags (0.3, 1.2, 6 and 12 ps). Validation uses 512
centers per parent. Mg uses 2,048 centers from each of four source branches;
Ta uses 8,192 centers from its single branch. Mg/Ta training lags are
0.1, 0.4, 2 and 4 ps. Material sampling is balanced despite unequal data counts.

Al shooting sources retain the original **11 train / 3 validation / 6 test**
split, with all parents and siblings assigned together. Mg's fifth source is
validation and its sixth source is excluded. Ta validation uses disjoint IDs
and frames 150–170; training stops by frame 120. Ta therefore has no independent
validation source. Mg/Ta source coordinates are float16, which limits fine-scale
motion accuracy; Al shooting coordinates are float32.

The primary final assay retains the original 3,054 initially noncoherent Al
test centers and 12/24/48 ps future targets. New 769-point patches reproduce
the original 193-point prefix for exactly the same benchmark centers.

## Evaluation

After all selection, frozen features are assessed with the original
coarse-order/temperature-augmented ridge forecasts, matched-neighbor future
agreement, and independent current TDA/SOAP/order probes. Fitting these readouts
does not update an encoder. Source-bootstrap intervals accompany the results.
Screening-only and three-seed confirmation rows are labeled as different tiers.

The learned embedding predictor is separately tested on held-out Al futures
at 0.3/1.2/6/12 ps against persistence and condition-matched shuffled futures.
Target effective rank accompanies latent errors to expose collapse. Rotation
and small-perturbation controls use identical sample draws across architectures.
The existing test set has been examined previously; this is a follow-up study,
not newly blind phase discovery.

## Storage, launch and outputs

The home quota is 100 GB. Large caches and resumable optimizer checkpoints go to
`/home/ids/vmorozov/experiments/temporal_hypotheses_12h_20260906/`.
Reports, configs, logs, compact selected weights, scalers and final embeddings
stay under `output/temporal_hypotheses_12h_20260906` in the repository; expected
new repository use is below 0.5 GiB. The cleanup ledger records 0.82 GiB freed
from obsolete/regenerable files while retaining source data and valid results.

```bash
conda activate pointnet
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/temporal_hypotheses_12h_20260906/run.py --config experiments/temporal_hypotheses_12h_20260906/config.json
```

The actual detached launch has a process-group watchdog, a twelve-hour budget,
and a short shutdown grace period. `launch.json`, `schedule_runtime.json`,
`status.json`, `prepare_status.json`, per-trial histories and `run.log` expose
progress and failures. Existing runs are not silently overwritten or restarted.

Shared scientific implementation resides in `src/data_utils/temporal_campaign.py`,
`src/training_methods/temporal_campaign.py` and `src/analysis/temporal_campaign.py`.
This directory is the versioned experiment record. Generated launch scripts,
cache audits and diagnostics reside in the output directory.

Detached launch: September 6 at 00:44 Paris time; training PID `387274`.
The twelve-hour deadline is **2026-09-06 12:44 Europe/Paris**, with a
separate short watchdog shutdown grace. See
[run status](../../output/temporal_hypotheses_12h_20260906/status.json) and
[launch record](../../output/temporal_hypotheses_12h_20260906/launch.json).

The short initial attempt was intentionally stopped before temporal screening to
standardize lag coverage across objectives and motion probes. Its logs and
checkpoints are archived under `attempt_before_lag_coverage_fix`; the corrected
campaign reuses prepared data and starts every scheduled training run afresh.

## Post-training commands

The existing entry point now exposes explicit analysis-only modes. All use the
original saved configuration and fixed selection manifests. `--analysis-only`
requires that any existing `evaluation/`, `embeddings/` and `checkpoints/`
directories in the repository output have first been archived. The recovery
retains the previous partial artifacts under `interrupted_analysis/`.
`--screen-analysis` creates a new `screen_analysis/` directory and fails if it
already exists. Neither command trains or reselects models. Review-only can be
rerun after both evaluations complete.

```bash
conda activate pointnet
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python experiments/temporal_hypotheses_12h_20260906/run.py --config experiments/temporal_hypotheses_12h_20260906/config.json --analysis-only
python experiments/temporal_hypotheses_12h_20260906/run.py --config experiments/temporal_hypotheses_12h_20260906/config.json --screen-analysis
python experiments/temporal_hypotheses_12h_20260906/run.py --config experiments/temporal_hypotheses_12h_20260906/config.json --review-only
```

These modes call shared implementation in `src/analysis/temporal_campaign.py`.
Review-only writes training audits and curves, representation summaries, paired
source-bootstrap comparisons and the figure of future-assay confidence
intervals. It fits no readout and performs no model inference. The original
training snapshot remains unchanged; analysis source copies and hashes are
stored separately in `analysis_source_snapshot/` and `analysis_provenance.json`.

## Full static-Al transfer analysis

Question: what spatial environments does the validation-selected predictive
density encoder distinguish across the six complete static-Al snapshots?
The explicit settings are in [full_static_Al.json](full_static_Al.json).
The existing entry point calls `src/analysis/temporal_static.py`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/temporal_hypotheses_12h_20260906/run.py --config experiments/temporal_hypotheses_12h_20260906/config.json --static-config experiments/temporal_hypotheses_12h_20260906/full_static_Al.json
```

This command requires a new output directory. It encodes all 772,953 existing
regular-grid centers, verifies source hashes and coordinate identity against the
previous static analysis, and rebuilds physical-Angstrom patches appropriate to
the trained encoder. It reuses repository spherical k-means and interactive MD
rendering, with seven shared clusters fitted to all rows. The independent local
geometry/TDA functions are reused on 512 uniformly sampled centers per frame.
PTM is retained as an imperfect reference assay, not a training target or phase
assignment. No artificial temporal history or prediction lag is supplied.

Results, embeddings, full spatial labels, plots, interactive HTML and provenance
are saved at [full_static_Al/RESULTS.md](../../output/temporal_hypotheses_12h_20260906/full_static_Al/RESULTS.md).
The new JSON is a versioned experiment configuration, the shared Python module
is scientific implementation, and the output files are generated artifacts.

Completed on September 6: 772,953/772,953 centers, all six source hashes and
coordinate arrays matched, 3,072 continuous-geometry/TDA assays, and six
interactive displays. The selected checkpoint is confirmation seed 123. The
cosine silhouette is 0.3773 on the fixed 3,000-row diagnostic subset, effective
rank is 12.52/128, and cluster/PTM ARI is 0.2412. Cluster 0 grows from 1.94% of
centers at 166 ps to 93.22% at 240 ps and has high bond coherence. The remaining
clusters differ in continuous geometry in the less ordered material; this does
not establish distinct phases or pre-crystallization precursors. In particular,
the older GeoFrame representation's larger silhouette (0.4580) is not a common
accuracy comparison across latent spaces.

## Standard static pipeline: encoder only

This is the requested standard post-training workflow. The preceding custom
full-static run is retained as an exploratory diagnostic; it used an additional
Linear/LayerNorm representation map and did not run the complete standard pipeline.

[static_Al_encoder_analysis.yaml](static_Al_encoder_analysis.yaml) is an exact copy
of `configs/analysis/static.yaml` except for checkpoint path, output directory and
input-data configuration path. The full runtime profile, connected-regime analysis,
representatives, t-SNE/UMAP, MD outputs, transition analysis and Blender settings
remain those of the standard configuration.

[static_Al_encoder_data.yaml](static_Al_encoder_data.yaml) retains the six-frame
static loader and its 160-point sampling/auto-cutoff settings, but uses physical
Angstrom offsets (`normalize: false`), all 160 points for representative geometry,
and a distinct physical-input cache on IDS. Raw encoder inference is invariant to
point ordering; the adapter places the center first for the density implementation.
On 48 centers per frame the standard 160-point inputs reproduce the original
193-neighbor raw encoder to about 1e-6 relative RMS. These differing point counts
have negligible effect here because the density cutoff is 8 Angstrom. The audit
is saved beside the exported checkpoint; it is a sampled check, not a proof for
all centers.

The checkpoint export contains only `DensityMLPEncoder` weights and its learned
density scaling buffers from confirmation seed 123, selected previously by
validation. There is no `representation.output` Linear/LayerNorm, forecast head,
motion head or EMA teacher. The shared adapter exposes these raw 128D features as
the existing pipeline's first model output. No analysis implementation is copied.

Export once to a new checkpoint directory:

```python
from pathlib import Path
from src.analysis.density_encoder_adapter import export_encoder
export_encoder(
    Path('output/temporal_hypotheses_12h_20260906/checkpoints/confirm_density_predictive_seed123.pt'),
    Path('output/temporal_hypotheses_12h_20260906/encoder_only_checkpoint/density_encoder.ckpt'),
    Path('experiments/temporal_hypotheses_12h_20260906/static_Al_encoder_data.yaml'),
)
```

Run the maintained analysis directly:

```bash
conda activate pointnet
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m src.analysis.pipeline experiments/temporal_hypotheses_12h_20260906/static_Al_encoder_analysis.yaml
```

Output: `output/temporal_hypotheses_12h_20260906/static_pipeline_encoder_only/`.
The adapter is shared model-loading implementation; the two YAMLs are versioned
experiment configurations; the exported checkpoint, cache and pipeline outputs
are generated artifacts. The main analysis cache and figures remain in the repo.

Completed successfully in 410 seconds: all 772,953 centers and six frames, four
connected-regime pairs, standard representatives and geometry profiles, t-SNE,
UMAP, 24 Blender renders and the standard transition summaries. Read the
[standard pipeline report](../../output/temporal_hypotheses_12h_20260906/static_pipeline_encoder_only/real_md/README.md)
and [interactive structure transitions](../../output/temporal_hypotheses_12h_20260906/static_pipeline_encoder_only/real_md/representatives/11_connected_regime_transitions_3d.html).
`encoder_only_audit.json` verifies unchanged analysis settings, complete raw
encoder output and identical saved-center coordinates. Static transitions use
the pipeline's coordinate-based matching; atom-identity flicker is unavailable
for these static inputs, as recorded in the standard report.

## Spatial geometry diagnosis

Question: why does predictive density lose the spatial organization seen in the
user's exact August 31 grouped-FactorVAE GeoFrame checkpoint? The
[diagnosis and proposed experiments](../../docs/static_structure_vs_prediction_20260906.md)
record findings and distinguish measurement from causal hypotheses.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/temporal_hypotheses_12h_20260906/diagnose_spatial_geometry.py --config experiments/temporal_hypotheses_12h_20260906/spatial_diagnosis.json
```

The thin dated recipe calls `src/analysis/static_spatial_diagnosis.py` and requires
a fresh configured output directory. It aligns 684,723 exactly shared centers
between shifted old/new grids, measures spatial label coherence and continuous
feature variation, and checks whether eight-PC clustering restores density's
spatial organization. It does not: silhouette improves but spatial coherence
barely changes. Encoder-only and projected density are both substantially weaker
than the older field on these measurements. Output CSVs, alignment provenance and
PCA-control assignments are generated diagnostics under `spatial_diagnosis/`.
The Python entry point and JSON are versioned experiment records; computation is
shared implementation in `src/analysis/`.
