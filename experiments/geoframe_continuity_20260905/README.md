# GeoFrame continuity and dynamic-frame diagnostic

Research question: do discrete GeoFrame grouping and triad choices cause finite
embedding jumps along continuous, atom-matched coordinate paths, and does
transporting a frame with local motion remove the frame-related jumps?

`analysis.py` and `summarize.py` are experiment records for this controlled
intervention, not maintained training tools. No model weights or shared encoder
implementation are changed. Configuration is in [config.json](config.json).
Generated diagnostics, arrays, figures and the report are under
`output/geoframe_continuity_20260905/` in this repository, approximately 197 MB.

From the repository root, using conda `pointnet`:

```bash
python experiments/geoframe_continuity_20260905/analysis.py \
  experiments/geoframe_continuity_20260905/config.json
python experiments/geoframe_continuity_20260905/summarize.py \
  output/geoframe_continuity_20260905
```

The analysis requires a new output directory to preserve existing results.
Change `output` in a new configuration for a rerun. The summary can be rerun
against the completed measurements. The configuration references the existing
Al/Mg/Ta trajectory manifest and repository copies of the original and epoch-49
VICReg checkpoints. It selects 72 centers across all 13 source branches,
uses the 21.0–21.1 ps pair, and evaluates 257 interpolation fractions.

The interventions are full selection, fixed outer 80 atoms, fixed patch
identities, fixed patches with motion-transported frames, and fixed patches
with held frames. The held-frame intervention is a causal diagnostic; it is
not a proposal to freeze the laboratory axes in a production encoder. The
transport intervention uses the proper Kabsch rotation of atom-matched patch
vectors and passes a rigid-rotation equivariance control.

The audit completed on node53's H100 in about 31 seconds, followed by report
generation. Grouped-forward reconstruction and repeat controls have zero
error. Both checkpoints show frame discontinuities in all 72 tested paths.
For the VICReg model's selected Ta switches, median input separation is
2.13e-6 Å and median maximum patch rotation is 117.1°. Holding frames reduces
the selected encoder jump by a median 99.9976%. Motion transport removes a
similar amount. With patches fixed, transport lowers the Ta 95th-percentile
encoder step distance by about 89% over the full interpolation paths.

Intermediate coordinates are mathematical probes, not additional simulated
timesteps. Selection of one large jump per path is deliberately diagnostic;
it does not estimate an unbiased event frequency. Transported frames change
the pretrained model's input distribution, and retaining structure and
prediction quality requires a subsequent test. Smooth handling of grouping
changes also remains necessary.

See the [complete results](../../output/geoframe_continuity_20260905/RESULTS.md),
[isolated Ta example](../../output/geoframe_continuity_20260905/Ta_frame_counterexample.png),
and [grouping/frame comparison](../../output/geoframe_continuity_20260905/grouping_frame_ablation.png).
