# Matched encoder parameter search

This family adds **separate** diagnostics to the unchanged `encoder_screen` /
`geoframe_evolution` assay. Old metric exports remain unchanged. Full raw metrics
are in each evaluation folder; the campaign CSV has one row per fit, checkpoint,
representation and material. Frame averages are unweighted means of defined
values across the three fixed material frames, with eligibility counts in JSON.
They are descriptive, spatially correlated observations, not independent-source
confidence intervals. Undefined values remain undefined.

## Fit-normalized structure and liquid neighbors

Eligible liquid anchors are the original assay's non-FCC/HCP/BCC atoms with at
most 0.1 crystalline-neighbor fraction. Original spatial half-cell fitting and
evaluation assignments are retained. At least 40 fitting and 40 evaluation rows
are required. Targets are six bond-order channels or the topology producer's
descriptor columns. Targets are standardized with fitting-liquid mean and
population standard deviation. Columns with fitting variance <=1e-12 are omitted;
the exact indices and row counts are saved. No evaluation variance or model
score chooses columns.

`embedding_nmse`, `density_nmse`, and `joint_nmse` are evaluation mean squared
errors, averaged equally over retained columns and rows, for separate ridge
alpha10 fits on embedding, density, or concatenated inputs. Each input scaler
uses fitting-liquid rows. SVD ridge and float64 target arithmetic are explicit.
`constant_nmse` predicts the fitting mean. `conditional_nmse_gain` is density
error minus joint error, so positive means added information beyond this declared
linear density baseline. These are errors in fitting standard-deviation units;
they do not divide by nearly zero evaluation variances. Historical per-column
R² and its arithmetic mean are retained as diagnostics, not selection scores.

`embedding_neighbor_nmse` retrieves ten fitting-liquid neighbors by **raw
exported Euclidean feature distance**, then averages squared fit-standardized
target differences over queries, neighbors and columns. The density-only control
retrieves ten neighbors by density distance. `retrieval_gain_vs_density` is
density-neighbor error minus embedding-neighbor error. Positive is better. This
asks whether useful liquid neighbors are preserved, beyond crystal/liquid
separation. No trained readout or refitted projection determines these neighbors.

## Interface resolution and coherence

`nonbulk_context` applies the original logistic-probe protocol after restricting
both spatial halves to context IDs 0,2,3,5,6: liquid-other, mixed crystalline
neighborhood, Al planar fault, fivefold-liquid proxy and ordered-liquid candidate.
Perfect crystal and nontemplate crystal interiors are excluded. The original
minimum20 fitting/test class counts and class-coverage receipts apply. Its
one-versus-rest AP values therefore cannot be substituted for the old full-cohort
AP values. These labels are independently computed proxies, not learned cluster
colors or proven thermodynamic phases. Ta/Zr class3 is undefined.

Spatial AUROC, participation rank and fixed-center perturbation response are
copied from the unchanged native snapshot assay. Rank and smoothness alone are
not success criteria. Identity-projector results duplicate their encoder and
are not extra independent fits. MLP projector versus encoder comparisons are
separate exports of the same training run.

## Prediction and training comparisons

9ps residual physical-order error, 12ps conditional hazard AP/Brier, selected
probe step, and whole-root paired intervals retain `encoder_screen` definitions.
Only Al has this dynamic assay. Step0 selection explicitly denotes a constant
predictor. Training uses no onset labels. Fifteen reused development roots are
not a fresh final test. Existing shared current-physics baseline predictions are
checked against the same row IDs, root IDs and event labels before comparison.

GeoFrame checkpoints are initial and12/24/35 complete passes (61 encoder updates
per pass). MACE checkpoints are initial and1024/2048/4096 balanced-with-replacement
updates, batch256 on1600 fitting patches. These are655.36 presentation-equivalents
at completion, **not** literal shuffled epochs. Sampling, optimizer, input and
target protocols remain distinct. Within-family two-seed factorial effects are
the primary comparisons; cross-family ranks are descriptive and confounded by
data, input support and supervision. Fixed final checkpoints are primary; curves
diagnose convergence. No checkpoint is selected using a final-test outcome.


Table export: 2026-09-23T19:05:48.337874+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
