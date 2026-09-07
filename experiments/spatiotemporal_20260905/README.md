# Matched spatiotemporal representation study

Question: how do the original, temporally fine-tuned VICReg, and VISReg GeoFrame
representations compare on identical Al/Mg/Ta temporal and static samples?

The [research report](../../docs/geoframe_spatiotemporal_vicreg_20260905.md) records
settings, results, checkpoint caveats and the external output locations. Shared
training/probe implementation is in `src/training_methods/spatiotemporal.py`;
shared comparison measurements are in `src/analysis/spatiotemporal.py`.

Run from the repository root. Inspect each recipe's `--help` for required paths:

```bash
conda run -n pointnet python experiments/spatiotemporal_20260905/prepare_spatiotemporal_vicreg_views.py --help
conda run -n pointnet python scripts/train_geoframe_spatiotemporal.py \
  --config-name vicreg_geoframe_v2_spatiotemporal_almgta_20260905 \
  --run-dir /path/to/new_experiment/training
conda run -n pointnet python experiments/spatiotemporal_20260905/analyze_spatiotemporal_vicreg_results.py \
  --experiment-root /path/to/completed_experiment
conda run -n pointnet python experiments/spatiotemporal_20260905/run_geoframe_spatiotemporal_post_analysis.py \
  --root /path/to/structural_analysis
conda run -n pointnet python experiments/spatiotemporal_20260905/compare_spatiotemporal_static_clustering.py \
  --root /path/to/structural_analysis
conda run -n pointnet python experiments/spatiotemporal_20260905/run_spatiotemporal_objective_comparison.py \
  --root /path/to/objective_comparison --static-cache /path/to/static_cache
```

The objective queue selects
`configs/{vicreg,visreg}_geoframe_v2_spatiotemporal_corrected_20260905.yaml` and calls
`compare_spatiotemporal_objectives.py`; that comparison can also be rerun directly
with the same `--root` and `--static-cache` arguments. The structural batch recipe
is [post_analysis.yaml](post_analysis.yaml): paths are relative to `--root`.
The recipe requires the saved per-material pipeline configurations in that root.

The original run artifacts are under
`/home/ids/vmorozov/experiments/geoframe_v2_spatiotemporal_Al_Mg_Ta_20260905/`.
Use the report for the completed-run findings; the code cleanup adds no new
scientific result and leaves checkpoint/config contents in their existing locations.

## Completed batch-8192 analysis stored in the repository

The September 5 rerun and full-Al analysis are stored at
`output/geoframe_v2_spatiotemporal_analysis_20260905/`, with copied best/final/last
checkpoints, exact configurations, JSON measurements, figures and interactive MD
outputs. The consolidated findings are in that directory's `RESULTS.md`.

Both objectives completed 60 epochs and 4,320 updates. The matched comparison
includes the original checkpoint and best/final checkpoints for each objective.
Full static Al uses all 772,953 neighborhoods from the existing six-frame regular
grid cache, including its standard boundary exclusions; it is not a per-atom scan.

Reproduce using existing commands and the saved configuration recipe:

```bash
python experiments/spatiotemporal_20260905/compare_spatiotemporal_objectives.py \
  --root output/geoframe_v2_spatiotemporal_analysis_20260905 \
  --static-cache /home/ids/vmorozov/experiments/geoframe_v2_spatiotemporal_Al_Mg_Ta_20260905/post_training/data_cache
python experiments/spatiotemporal_20260905/run_geoframe_spatiotemporal_post_analysis.py \
  --root output/geoframe_v2_spatiotemporal_analysis_20260905/full_static_Al \
  --recipe output/geoframe_v2_spatiotemporal_analysis_20260905/full_static_Al/recipe.yaml
python experiments/spatiotemporal_20260905/branch_temporal_comparison.py \
  --root output/geoframe_v2_spatiotemporal_analysis_20260905
python experiments/spatiotemporal_20260905/summarize_full_static_analysis.py \
  --root output/geoframe_v2_spatiotemporal_analysis_20260905
```

`branch_temporal_comparison.py` is an experiment diagnostic that computes drift
and effective rank within each source branch from the saved matched probes.
`summarize_full_static_analysis.py` consolidates the five full-Al analyses, verifies
identical sampled coordinates and complete coverage, and reports cluster agreement
and projector effective rank. These are experiment records; generated outputs stay
in the repository's `output/` tree.
