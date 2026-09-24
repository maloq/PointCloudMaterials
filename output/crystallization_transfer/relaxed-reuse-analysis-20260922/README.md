# Relaxed versus original MACE: matched forecast figures

PNG only,300 dpi. Completed eight-fit comparison; existing relaxed archives, no new simulation. All plots use the matched reused-data cohort; these are not direct replications of the earlier dense-history comparison.

## Prediction summary

![Prediction summary](plots/01_prediction_summary.png)

Matched test comparison across all four predictor families. Higher AP at 12 ps and lower integrated Brier over 0.75–96 ps are better. Points are one-seed estimates; connecting lines indicate matched methods, not confidence intervals. Both encoder and input domain change together.

## Precision recall

![Precision recall](plots/02_precision_recall.png)

12 ps precision–recall curves, equal total weight per test source. Dotted lines show source-weighted prevalence. Curves are computed from saved predictions and reproduce the exported AP values. There are 7654 test windows from 30 sources;226 raw windows are positive, and overlapping windows are not independent events.

## Timing and misses

![Timing and misses](plots/03_timing_and_misses.png)

Timing error must be read together with misses. Left: conditional event-time MAE only on detected positive 12 ps windows. Right: raw missed-window fraction among 226 positives. Each fit uses its own threshold calibrated to 5% FPR on calibration sources; realized test FPR can differ. The detected populations differ between fits, so lower left-hand bars alone do not establish better overall timing.

## Skill by horizon

![Skill by horizon](plots/04_skill_by_horizon.png)

AP as the forecast horizon changes for the same full test population. This is not the fixed-event offset analysis: longer horizons change which windows are positive and increase event prevalence. Source weights and saved CDFs match the primary evaluation.

## Physical trajectory error

![Physical trajectory error](plots/05_physical_trajectory_error.png)

Source-weighted mean-path squared error by future time for autoregressive and mixture predictors. Blocks use identical training-normalized original-MD targets in both arms, including the common original-MACE latent targets. Dotted persistence uses the original MD present state; it is a reference baseline and does not imply that state was supplied to the relaxed predictor. Mixture means are estimated using the saved evaluation samples.

## Calibration

![Calibration](plots/06_calibration.png)

Reliability diagrams in 10 fixed probability bins at 12 ps. Means and event fractions use equal-source weights. Marker area increases with bin probability mass; sparse high-risk bins should not be read as precise calibration estimates. Predictions are raw saved model risks, not fitted test-set recalibrations.

## Same event offset skill

![Same event offset skill](plots/07_same_event_offset_skill.png)

Fixed cohort of 66 local onset/control pairs from 18 test sources, present at all four lead bins. At nominal leadL the archived origin has actual lead in[L,L+12) ps; no prediction is interpolated. Controls come from the same source and survive beyond the case onset. Matched AP has 50% weighted prevalence and is not comparable numerically to natural-population AP. Timing uses the 96 ps restricted mean for every case, including weak or missed predictions. Bands:1000 whole-source bootstrap draws, conditional on the trained seed and selected cohort.

## Same event forecasts

![Same event forecasts](plots/08_same_event_forecasts.png)

Autoregressive onset CDFs for three label/availability-selected events from distinct sources. Both columns show identical atoms, origins and outcomes. Each color is an independently initialized open-loop forecast from a real archived origin; zero marks actual onset. Dots mark forecast origins. Examples were selected with a fixed random seed before reading forecast quality. Exact source, atom-index, onset and origin coordinates are in technical/matched-events.json.

## Structural forecasts

![Structural forecasts](plots/09_structural_forecasts.png)

Original-MD q 6 trajectories and autoregressive forecasts for the same three example events at the shortest available lead bin. Grey shading marks the available past span; only the three selected snapshots entered the predictor, not the dense line shown as truth. Dashed vertical lines mark actual sustained onset. Model checkpoint replay reproduced saved event CDFs within 2e-5; physical states use the saved training normalizers.

## Mixture trajectories

![Mixture trajectories](plots/10_mixture_trajectories.png)

Mixture forecasts for the first two fixed examples:32 replay samples, mean, and pointwise 5th–95th percentile envelope; black is original MD truth. These are model sample intervals, not demonstrated 90% coverage. Individual paths are not independent posterior parameter draws. Example choice is identical across domains.

## Embedding umap

![Embedding umap](plots/11_embedding_umap.png)

Separate UMAP maps of instantaneous central embeddings. Each domain uses the same sampled windows (up to 24 per source); StandardScaler and UMAP fit only training sources, then transform held-out sources. Top: grey training reference and test temperature. Bottom: test windows colored by future 12 ps onset; these labels were not used to fit UMAP. Axes between encoders are not aligned, and apparent clusters do not establish predictive sufficiency.

## Original vs relaxed atoms

![Original vs relaxed atoms](plots/12_original_vs_relaxed_atoms.png)

Real matched atomic neighborhoods from the reused test archives: one originally liquid, one 0–12 ps before original-MD sustained onset, and one originally crystalline. Every pair has exactly the same 80 observed-nearest atom IDs, centered on the same tracked atom (dark marker), with one common orthographic camera and scale. Thin lines connect pairs closer than 3.5 Å in each displayed structure. No alignment or synthetic lattice is used. Right: center-relative observed-to-relaxed displacement arrows at their true scale; none of the panels is magnified or deformed. State names refer to original MD, not reassigned relaxed PTM. Exact identities and RMS displacements are in technical/paired-clouds.json.

## Full spatial context

![Full spatial context](plots/13_full_spatial_context.png)

Whole 25-slot context for the same pre-onset sample in the paired-atom illustration. Center plus 12 slots at 10 Å and 12 at 20 Å; hollow markers are nominal symmetric queries, filled markers the actual assigned atoms. Query IDs are selected once in observed geometry and retained after relaxation. All 25 neighborhoods are shown. Circles mark the maximum 7.94 Å support. Original MACE uses all neighbors inside this radius; the relaxed checkpoint retains its 80 observed-nearest candidates and crops them after relaxation. Thus the actual local supports differ, as in training. Colors distinguish query shells, not phase labels. Projection can overlap supports; the queries are symmetric in 3D.

## Learning curves

![Learning curves](plots/14_learning_curves.png)

Open-loop selection-source integrated Brier by epoch. Stars mark the selected checkpoint. Curves end at the actual early-stopping point or 36-epoch ceiling. Test outcomes were not used for checkpoint selection; this is a one-seed optimization trace.
