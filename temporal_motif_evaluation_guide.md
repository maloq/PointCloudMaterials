# Implementation Guide for Coding Agent — Evaluation and Full Analysis

## Goal

Extend the current analysis pipeline so that after training a Temporal Motif Field model, evaluation produces a **complete dynamic motif analysis** rather than only static latent clustering.

The output should answer:

1. what stable motifs exist,
2. what bridge motifs exist,
3. how motifs evolve over time,
4. how long motifs persist,
5. what transition graph exists between motifs,
6. whether motifs recur,
7. whether nearby atoms help explain motif evolution.

This guide is designed to **reuse** the existing `src/analysis` pipeline instead of replacing it.

---

## Grounding assumptions from the current repo

The repo already has a strong analysis base. Reuse it.

Existing analysis currently provides:
- inference caching,
- latent statistics and PCA,
- clustering and clustering-method comparison,
- figure-set rendering,
- t-SNE / UMAP-like latent projections,
- real-MD qualitative outputs,
- cluster proportion time-series,
- transition flow analysis,
- temporal animations,
- representative structures,
- descriptor analysis,
- equivariance evaluation.

The new dynamic motif analysis should be **added on top** of that.

---

## Deliverables

Create these files:

```text
src/analysis/dynamic_motif.py
src/analysis/dynamic_motif_metrics.py
src/analysis/dynamic_motif_plots.py
src/analysis/dynamic_motif_cache.py
```

Modify these files:

```text
src/analysis/pipeline.py
src/analysis/config.py
src/analysis/real_md_qualitative.py
```

Optional:
```text
configs/analysis/checkpoint_analysis_tmf.yaml
```

---

## Design principle

Do **not** route dynamic motif evaluation through generic clustering of `inv_latents` only.

Instead, evaluation should prefer model-native outputs when available:

- stable motif probabilities / IDs,
- bridge motif probabilities / IDs,
- residual magnitude,
- hazard predictions,
- future motif predictions,
- optional field logits.

Fallback to latent clustering only when model-native outputs are absent.

---

## Output directory structure

Write dynamic outputs under:

```text
<analysis_output_dir>/
    analysis_metrics.json
    analysis_inference_cache.npz
    dynamic_motif/
        summary.md
        assignments/
            sample_assignments.csv
            frame_motif_proportions.csv
            stable_usage.csv
            bridge_usage.csv
        transitions/
            transition_counts.csv
            transition_probs.csv
            transition_matrix.png
            transition_flow.png
            stable_to_bridge_to_stable.csv
        dwell/
            dwell_times.csv
            survival_curves.csv
            hazard_by_motif.csv
            hazard_calibration.csv
            dwell_histograms.png
            survival_curves.png
        recurrence/
            recurrence_scores.csv
            motif_revisit_counts.csv
            recurrence_heatmap.png
        representatives/
            stable_motif_*.png
            bridge_motif_*.png
            representative_index.csv
        temporal/
            motif_timelines.csv
            motif_proportion_area.png
            per_atom_event_atlas.csv
            motif_umap.png
            bridge_event_gallery.png
        field/
            neighbor_gain.csv
            neighbor_influence_heatmap.png
```

Keep `analysis_metrics.json` as the top-level machine-readable summary.

---

## Extend inference cache

The existing pipeline already saves and reloads an `.npz` inference cache. Extend it with optional dynamic arrays.

### Required new cache keys

If the model exposes them, save:

```python
cache["stable_probs"]          # [N_samples, K_stable]
cache["stable_ids"]            # [N_samples]
cache["bridge_probs"]          # [N_samples, K_bridge]
cache["bridge_ids"]            # [N_samples]
cache["residual_norm"]         # [N_samples]
cache["hazard_probs_lag_1"]    # [N_samples]
cache["hazard_probs_lag_2"]    # ...
cache["future_pred_ids_lag_1"] # optional
cache["future_pred_ids_lag_2"] # optional
cache["center_atom_id"]        # [N_samples]
cache["frame_index"]           # [N_samples]
cache["timestep"]              # [N_samples]
cache["source_path"]           # [N_samples]
```

### Important rule

Inference cache must remain backward-compatible:
- load old caches if dynamic keys are missing,
- skip dynamic analyses gracefully with warnings.

### Spec versioning

Update cache spec hash / metadata so dynamic and non-dynamic caches do not collide.

---

## Config extension

Add a new config section in `src/analysis/config.py`:

```yaml
dynamic_motif:
  enabled: true
  export_per_sample_arrays: true
  use_model_outputs: true
  stable_k: null
  bridge_k: null
  representative_samples_per_motif: 12
  transition_top_k: 20
  bridge_min_support: 50
  dwell_min_length: 1
  recurrence_max_gap: 64
  render:
    heatmaps: true
    timelines: true
    representatives: true
    event_gallery: true
    sankey: true
  field:
    enabled: false
    top_neighbors: 8
```

### Parsing

Create a dataclass similar to existing analysis settings and merge it into resolved analysis config.

---

## Pipeline integration

Modify `run_post_training_analysis(...)` in `src/analysis/pipeline.py`.

### Recommended insertion point

Run dynamic motif analysis **after** inference cache has been gathered / loaded and **after** clustering state has been created, but **before** writing final metrics.

### Flow

1. load model / gather cache
2. run current latent statistics
3. run current clustering analysis
4. run current figure-set / real-MD analysis
5. **run new dynamic motif analysis**
6. merge metrics into `analysis_metrics.json`

### New call

```python
dynamic_metrics = run_dynamic_motif_analysis(
    cache=cache,
    out_dir=out_dir,
    model_cfg=cfg,
    analysis_cfg=analysis_cfg,
    cluster_labels_primary=primary_cluster_labels,   # fallback only
    step=_step,
)
all_metrics["dynamic_motif"] = dynamic_metrics
```

---

## Core analysis functions

Implement these in `dynamic_motif.py` / `dynamic_motif_metrics.py`.

### 1. Resolve assignments

Function:
```python
resolve_motif_assignments(cache, cluster_labels_fallback=None) -> AssignmentBundle
```

Priority:
1. use `cache["stable_ids"]` if available,
2. else derive from `stable_probs.argmax`,
3. else fallback to primary cluster labels.

Return:
- stable assignments
- bridge assignments if available
- metadata arrays aligned to samples

### 2. Frame-wise motif proportions

Group by:
- `source_path`
- `frame_index` or `timestep`

Compute:
- counts per stable motif,
- fractions per stable motif,
- bridge motif fractions separately,
- unknown / unassigned fraction if any.

Write:
- `frame_motif_proportions.csv`

### 3. Transition graph

For each atom track:
- sort by frame index,
- compare motif at `t` and `t+1`,
- aggregate transitions,
- separate:
  - stable->stable
  - stable->bridge
  - bridge->stable
  - bridge->bridge

Write:
- `transition_counts.csv`
- `transition_probs.csv`

Render:
- heatmap
- flow diagram

### 4. Dwell time and survival

For each atom track and each motif:
- segment consecutive runs,
- measure dwell length,
- aggregate by motif.

Compute:
- mean dwell,
- median dwell,
- p90 dwell,
- survival curve,
- empirical hazard by elapsed dwell time.

Write:
- `dwell_times.csv`
- `survival_curves.csv`
- `hazard_by_motif.csv`

### 5. Hazard calibration

If model hazard predictions exist:
- compare predicted hazard vs empirical change event,
- compute Brier score,
- calibration bins,
- optional AUROC / AUPRC.

Write:
- `hazard_calibration.csv`

### 6. Bridge motif statistics

For each bridge motif:
- support count,
- mean confidence,
- median lifetime,
- entering stable motifs distribution,
- exiting stable motifs distribution,
- top `(stable_i -> bridge_b -> stable_j)` triplets,
- recurrence count across different atoms / times.

Write:
- `stable_to_bridge_to_stable.csv`
- `bridge_usage.csv`

### 7. Recurrence analysis

Goal: distinguish recurring motif events from one-off noise.

For each motif:
- count revisit events by the same atom after leaving and later returning,
- compute recurrence score,
- compute motif revisit gap histogram,
- optionally motif mutual information across long lags.

Write:
- `recurrence_scores.csv`
- `motif_revisit_counts.csv`

### 8. Field / neighbor influence (optional)

If field outputs or neighbor metadata exist:
- compare future prediction quality with and without neighbor features,
- compute neighborhood gain,
- summarize which motifs are most spatially contagious / correlated.

Write:
- `neighbor_gain.csv`

---

## Reuse existing real-MD qualitative outputs

Do not duplicate what already exists in `real_md_qualitative.py`.  
Extend it.

### Existing outputs already useful

Preserve:
- cluster proportions plots,
- transition flow,
- temporal spatial cluster animations,
- latent projection animations,
- representative structures,
- descriptor summaries.

### Add new dynamic motif outputs

Inside or beside the existing summary markdown, add sections for:
- stable motifs,
- bridge motifs,
- dwell/hazard,
- recurrence,
- field effects.

If model-native motif outputs exist, label figures as “motif” rather than “cluster”.

---

## Representative motif extraction

Implement representative selection for both stable and bridge motifs.

### Stable representatives

Select examples by:
- highest assignment confidence,
- diversity in residual norm,
- diversity in source_path / time if possible.

### Bridge representatives

Select event-centered windows:
- anchor frame = bridge assignment,
- include neighboring frames `[t-2, ..., t+2]`,
- save montage or gallery.

Write:
- `representative_index.csv`
- image files under `representatives/`

---

## Per-atom event atlas

This is important for “whole analysis”.

For a selected set of tracked atoms:
- export timeline of motif IDs,
- dwell boundaries,
- bridge events,
- hazard spikes,
- future prediction correctness.

Write:
- `per_atom_event_atlas.csv`

Optional plot:
- one row per atom, one column per frame, colored by motif.

This is extremely useful for debugging motif flicker and bridge validity.

---

## Metrics to include in analysis_metrics.json

Add these keys under `dynamic_motif`.

### Assignment
- `num_stable_motifs`
- `num_bridge_motifs`
- `stable_active_count`
- `bridge_active_count`
- `stable_usage_entropy`
- `bridge_usage_entropy`

### Dynamics
- `transition_entropy`
- `self_transition_fraction`
- `mean_dwell_overall`
- `median_dwell_overall`
- `bridge_fraction`
- `revisit_rate`

### Prediction
- `future_top1_acc_by_lag`
- `future_nll_by_lag`
- `hazard_brier_by_lag`
- `hazard_auroc_by_lag`

### Field
- `neighbor_gain_by_lag` (if available)

### Artifacts
Store paths to key CSV / PNG outputs.

---

## Fallback behavior

The dynamic analysis must never crash when a field is missing.

### Cases

1. **No model-native stable outputs**
   - fallback to current primary cluster labels.

2. **No bridge outputs**
   - skip bridge sections, but still compute stable transitions / dwell.

3. **No hazard outputs**
   - compute empirical hazard only.

4. **No neighbor metadata**
   - skip field section.

5. **No tracked atom identity**
   - skip per-atom dwell / recurrence with a clear warning.

---

## Plotting requirements

Implement plots in `dynamic_motif_plots.py`.

### Required
- transition heatmap
- motif proportion area plot
- dwell histogram / survival plot
- recurrence heatmap
- bridge event gallery
- motif latent 2D plot colored by stable / bridge / hazard

### Nice-to-have
- Sankey / alluvial transition flow
- per-atom timeline heatmap
- hazard calibration curve

Do not make plotting a hard dependency for metric computation.

---

## Integration with summary markdown

The current real-MD summary already writes markdown. Extend it.

Add sections:
1. Stable motifs summary
2. Bridge motifs summary
3. Dwell / survival summary
4. Hazard summary
5. Recurrence summary
6. Neighbor influence summary

Use relative artifact paths so the markdown remains portable.

---

## Acceptance criteria

### Functional
- analysis runs from existing pipeline entry point
- old checkpoints still analyze successfully
- dynamic outputs appear only when supported
- `analysis_metrics.json` always writes successfully

### Content
For a temporal model with motif outputs, evaluation must produce:
- motif proportions over time
- transition counts / transition figure
- dwell statistics
- bridge statistics
- representative stable and bridge examples
- summary markdown

### Robustness
- works on cached inference and fresh inference
- works on subset runs
- works when `figure_set.figure_only == false`
- skips unsupported sections cleanly

---

## Suggested command-line target

Keep using the existing analysis entry point and pass a new config:

```bash
python src/analysis/pipeline.py configs/analysis/checkpoint_analysis_tmf.yaml
```

Do not create a parallel standalone evaluator unless absolutely necessary.

---

## Notes for the agent

- Reuse current analysis helpers whenever possible.
- Prefer CSV + PNG + JSON outputs over heavy custom formats.
- Keep model-native motif outputs primary and latent clustering secondary.
- Make every section independently skippable.

