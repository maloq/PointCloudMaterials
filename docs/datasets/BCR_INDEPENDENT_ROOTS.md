# Independent roots for the first BCR mechanism pilot

The complete audit is [bcr_independent_roots.csv](bcr_independent_roots.csv):150
existing independent-melt source trajectories, historical exposure/split roles,
generating potential, source/parent IDs, coordinate precision and available times.

Historical training sources associated with later520K histories supply18 independently
melted roots. Their **preceding melt endpoints are at1325K**, confirmed from the
actual input files; they must not be relabeled520K.12 are training roots and6 are
held-out pilot development roots. Existing historical final-test/calibration/selection
roots are excluded. None of this creates a genuinely untouched final test.

High-precision single-frame melt validation dumps preserve original integration
coordinates (nine significant digits) and boxes. The long subsequent histories are
float16, which is too coarse for the retained weakest artificial corruption level.
The preparer records both precision facts; it never treats a float16->float32 cast
as recovery. The fixture has one snapshot per root,256 spatially thinned centers,
and a complete radius8A support with no truncated neighbors.

The cache is registered as `bcr-g1-independent-20260921`. The precise frozen audit
and sampling recipe are in `output/bcr/g1-independent-20260921/technical/inventory.json`
and `configs/bcr/pilot_20260921/study.json`. No future trajectory labels choose
centers, patches, roots or training targets. The one-frame melt cohort is suitable
for a liquid structural conditioning assay, not an undercooled dynamics claim.
