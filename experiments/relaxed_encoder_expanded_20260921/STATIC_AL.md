# Development-selected relaxed encoder on static Al

Use `cold-vic-temp01`, selected by the expanded study's minimum cold-domain
development Physical + .25 TDA score (0.29842378944158554). Its selected checkpoint
is step2048. Test AP and embedding rank do not select this model. Export the native
raw invariant128 state, preserving `export_norm=raw` and the identity projector.

Analyze the existing already-relaxed 166/170/174/175/177/240ps Al snapshots directly:
no additional relaxation. Retain the earlier cached interior centers and seven-
cluster analysis settings. Select the nearest80 candidate atoms on these static
coordinates, crop physical FP32 offsets at normalized radius8 before scaling,
then build native cutoff5 graphs with the trained taper and material scale.
Training selected candidate identities before quenching; those original hot-frame
identities are unavailable here. Static nearest80 selection is therefore an
explicit deployment convention, not an exact reconstruction of paired training
inputs. Eight real crops reproduce the frozen training graph_arrays producer
exactly once graph-local edges are packed with their node offsets.

Use native eager BF16 inference as in the preceding Epi analysis. Raw unnormalized
states show native repeat/implementation arithmetic differences around1e-5;
native-output verification declares atol2e-5 and rtol2e-5. Checkpoint weights and
all12 input tensors must match exactly. The user-disabled batch-replay stopping
check remains disabled; singleton/reordering differences are recorded.

Recipes: `configs/analysis/relaxed_best_static.json` (export/verify),
`configs/analysis/static_relaxed_best_al.yaml` (standard pipeline), and
`configs/data/loaders/static_al_relaxed_best.yaml` (static inputs). Execute from
the result's frozen `technical/runtime` in conda `pointnet-torch214`:

```bash
python -m src.analysis.neighborhood_adapter --config configs/analysis/relaxed_best_static.json --stage export
python -m src.analysis.neighborhood_adapter --config configs/analysis/relaxed_best_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_relaxed_best_al.yaml
```

Export requires a fresh output directory. Frozen checkpoint, exact producer hashes,
selection receipt and launch record are retained with the results.
[Results](../../output/structural_static/relaxed-cold-vic-temp01-step2048-al-20260921/).
