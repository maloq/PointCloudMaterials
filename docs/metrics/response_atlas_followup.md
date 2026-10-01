# Response-atlas follow-up metrics

Protocol: `configs/simulation/response_atlas_followup_20261001.json`.
Development evidence only: no atomistic encoder or active-acquisition benefit.

## Toy controls

Retain the pilot Gaussian feature law, width32/latent4 MLP, optimizer, 128 training
coordinates and 64 selection coordinates. Three arms use eight value shots,
32 value shots, or the same eight value shots plus exact pathwise responses.
Streams are paired; values8/responses8 labels must be bitwise equal. All arms
share new 64-shot selection labels. Feature scales use eight-shot training labels;
the auxiliary response scale is training-only. Minimize fixed-variance Gaussian
feature NLL, plus scaled response MSE for responses8. Select solely by shared
selection feature NLL. Inputs are u,v; no temperature/time/history inputs.
Full-batch128/microbatch128 and the small MLP are explicit toy exceptions.

Five paired initialization seeds share data: seed spread measures initialization
variation, not independent-dataset uncertainty. Save the best selection checkpoint
by 250/1000/2500 updates and by 15/45 seconds of total measured CPU acquisition
plus optimization/selection. Charge each arm its own training-label construction
and common selection-label construction, even when selection is physically reused.
Forward-only arms do not pay for response labels; response acquisition runs the
actual AD oracle. Online initialization and final evaluation are excluded. Loop
bookkeeping, copies and training logging are included. One update may overshoot
a time budget; actual elapsed time is exported. Continue until both the final
epoch and time checkpoints exist. Rotate model order across seed workers.
Shared-machine contention and cold starts limit these short wall-time comparisons.
Toy CPU costs do not estimate atomistic derivative/value costs. Wall-time fields
are not repeatedly logged to W&B metric history.

Exact feature-mean and Jacobian MSE average outputs/directions on 4096 new fixed
points from an independent RNG seed. Freeze all choices/checkpoints before scoring
these points. New selection precision and evaluation points define a separate
protocol; pilot tables are preserved. Save checkpoints and per-point predictions.
Scientific training is online in teshbek/PointCloudMaterials.

## Atomistic precision

Parents 0,1,8,12 are selected development states: original verification parents
and weak/noisy responses. All share the FCC prototype, not independent ancestry.
Fresh namespace 20,000,000 + 1000*parent supplies 32 AD branches and 16 independent
CRN finite-difference pairs per direction. None reuse pilot streams. Retain both
20/100 fs joint prefixes, potential, units, features, basis and RNG layout.
Rerun energy/force, HVP and path-response gates; no 500-fs extension.

At fixed nested budgets 4/8/16/32, squared mean response is trace of the off-diagonal
Gram, without clipping. Sample-mean variance trace is summed unbiased coordinate
variance divided by B. Bootstrap endpoints are 2.5/97.5 percentiles of 1000 ordinary
branch-resampling statistics at each parent/horizon/budget. These are descriptive
shot-resampling intervals, not calibrated acceptance bounds: the squared-mean
statistic is degenerate near zero and ordinary bootstrap can be biased there,
especially at small B. Prefixes are dependent. No optional stopping, multiple-look
inference, equivalence or population-generalization claim. Always run 32 shots.

The atomistic-v1 tables retain their separate response_atlas contract and record
AD-versus-fresh-FD differences, screening and whole-path costs. Horizon rows repeat
whole-path costs: never sum them across horizons.
