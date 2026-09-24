# GeoFrame 12-pass versus 35-pass review

Inputs are completed epoch-011 (12 passes) and epoch-034 (35 passes) assay
exports, plus the archived epoch-34 reference for visual context. Endpoints were
defined by the requested convergence floor and reproduction endpoint, not by
selecting a favorable measured metric peak. Base definitions remain frozen in
the original run's `tables/METRICS.md` and `technical/metric-contract.json`.

For each fresh endpoint and representation, `future_mse_minus_current_baseline`
subtracts the zero-future-residual error from the ridge readout's standardized
future-residual error, separately per development source. `onset_brier_minus_current_physics`
subtracts the conditions-only hazard readout's 12 ps Brier error from the
embedding+conditions readout's error, separately per source. Match exact indices,
source IDs and event labels before comparison. **Negative means improvement**.

Both means give each of the fifteen independent development roots equal weight.
`ci95` is the percentile interval from 2,000 paired resamplings of those fifteen
source effects, seed 20260923. These intervals condition on this encoder seed,
fitting/tuning partition, target calibration and readout recipe. They are not
training-seed uncertainty, multiple-checkpoint-adjusted tests or evidence of a
causal relationship between representation metrics and nucleation.

`onset_ap` retains the original source-weighted AP; `selected_hazard_step` is the
step selected by tuning NLL, including zero. AP alone is particularly sensitive
to ranking of nearly constant risks; interpret it with Brier and the constant/
current-physics controls. An undefined contrast is not replaced by zero.

Cluster-context plots use saved held-out K=7 contingency matrices. Each reference
class column is divided by its held-out count; an absent class stays NaN/blank.
Rows keep arbitrary cluster identities. No diagonal assignment or physical
meaning is inferred from color. Context 2 is a mixed-neighborhood proxy and can
include grain boundaries/isolated motifs as well as solid–liquid interfaces.
