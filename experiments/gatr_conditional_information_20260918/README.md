# GATr information beyond radial structure and density

September 19: [repeat for the final local MACE and GATr checkpoints](LOCAL_LAST622.md),
using identical trajectories and matching populations with symmetric radial controls.

Question: among environments with similar radial structure and density, does
the exported representation distinguish bond order, angular arrangement and
future crystallization? This follows the
[directional audit](../gatr_equivariant_20260918/README.md), but tests the actual
exported scalar z128 rather than interpreting internal multivectors as arrows.

**Completed:** frozen Al v6 GATr checkpoint 1216 on A100/node07. No encoder was
trained or modified. [Full findings](../../output/gatr_conditional_information/al-v6-node07-20260918/RESULTS.md),
[dense matched extension](../../output/gatr_conditional_information/al-v6-node07-20260918/RESULTS_spatial.md),
[figure gallery](../../output/gatr_conditional_information/al-v6-node07-20260918/index.html),
and [exact metric definitions](../../docs/metrics/gatr_conditional_information.md).

## Protocol

Reuse 32,040 observations from ten encoder-held-out Al trajectories at five
temperatures, four atom tracks per source, 801 frames at 0.75 ps. These sources
were already explored; the results are an exploratory follow-up. Original
checkpoint, observation and embedding hashes are verified. Probe fitting holds
out an entire simulation source, with nested source splits for regularization.
Linear and nonlinear random-Fourier-feature ridge probes use training-only
scaling and source weighting. Uncertainty resamples whole sources, stratified
by temperature; frames and overlapping pairs are not independent replicates.

The radial control combines the inner radial descriptor, 80 neighbor radii,
33 full-support radius quantiles, radial moments, counts, density, coordination,
temperature and time. R-star additionally contains the same frozen GATr applied
to a deterministic angular arrangement of the identical sorted radii. Test the
additional information from the original z128, its difference from this radial
state, MACE, SOAP and TDA. The synthetic radial control is outside the physical
data distribution and does not remove all limitations of finite radial models.

Structure targets are six bond-order channels and sixteen invariant angular
Legendre moments. Natural pairs share source and frame; both inner-radii RMS
and full-support radius-quantile RMS differences must be at most 0.05 Å, with
at most 2% relative density difference. Predeclared sensitivities are 0.025 and
0.10 Å. Targets and embeddings do not select matches.

Only three primary-caliper pairs exist among the four tracked atoms, so that
population cannot support inference. A separately recorded spatial extension
uses the preceding audit's 70 fixed snapshots and 22,381 local environments.
The extension reuses the original trajectory probes and their selected settings;
spatial outcomes do not tune them. It supplies **609 primary-caliper pairs across
nine sources**, and 125,006 pairs across ten sources at 0.10 Å. Dense patches
are not a uniform full-system sample. There are no matches at 0.025 Å.

Prospective tests remain on the identity-preserving trajectories: 17,674 at-risk
observations, before first sustained local FCC/HCP/BCC onset, with complete
24/48/96 ps follow-up and eight-frame confirmation. Control current bond/angular
order separately. Duplicate the radial-only state as an additional control:
redundant inputs change ridge regularization and the nonlinear feature geometry
without adding information. This is observed first onset, not a committor or
an iso-configurational propensity estimate.

## Findings

The frozen GATr export adds little structural information after radial control.
For the 609 strictly matched spatial pairs, nonlinear error reductions are:

| Target contrast | Add GATr z128 | Add SOAP |
|---|---:|---:|
| q6 | +0.2% [-7.4, +13.8] | +95.4% [+91.8, +96.7] |
| qbar6 | -7.6% [-17.6, +7.9] | +66.2% [+49.3, +74.7] |
| Angular arrangement | +1.0% [approximately 0, +2.6] | +66.7% [+49.0, +73.7] |

Brackets are paired source-bootstrap 95% intervals, conditional on fitted
readouts. The small angular effect should not be described as literally zero
information. At the looser spatial threshold, GATr gains are -1.2% for q6 and
-0.1% for angular arrangement. On the complete trajectory population, MACE
provides positive nonlinear gains of 19.5% for q6 and 14.9% for angular moments;
SOAP gains are 91.5% and 43.7%. These latter numbers use a different population
and score individual environments rather than matched contrasts.

The apparent 0.5–1.4% original-GATr forecasting gains at 48/96 ps are almost
entirely reproduced by duplicating the radial-only state. Against that duplicate,
nonlinear original-state gains at 24/48/96 ps are +0.0046%, -0.0008%, -0.0192%.
The explicitly scaled angular difference has a small 24 ps increment before
current-order conditioning, but neither probe family has a positive interval
at any horizon after both current order and duplicate-radial controls.

Forecasting remains inconclusive about the existence of exploitable information:
the nonlinear GATr readout does not beat training-prevalence-only Brier error
at any horizon, and there are only forty atom tracks. There is no demonstrated
robust forecasting benefit here, not a proof that a better predictor cannot
extract one. Structural positive controls provide stronger evidence that the
current export loses useful angular detail. One checkpoint and previously
explored sources do not establish an architecture-wide result.

This supports testing explicit angular targets connected to the exported state,
as proposed in [meaningful equivariance](../gatr_equivariant_20260918/MEANINGFUL_EQUIVARIANCE.md).
It does not demonstrate that the proposed tensor branch will help.

## Reproduction

Use the requested node07/A100 allocation and conda `pointnet-torch214`:

```bash
python -m src.research.gatr_conditional_information \
  --config configs/analysis/gatr_conditional_information.json
python -m src.research.gatr_conditional_information.spatial \
  --config configs/analysis/gatr_conditional_spatial.json
python -m pytest -q tests/test_gatr_conditional_information.py
```

The main module supports `--stage prepare|probe|report|all`. The six targeted
tests passed on node07, covering radius-only construction, prospective censoring,
matching, paired improvement, training-only probe scaling and TDA block scaling.
Initial sparse-match exports and the original unstable coordinate-scaled TDA
diagnostic remain in the run's `technical/initial-sparse-matching/`; final TDA
fits use one training-derived scale for their 144-coordinate block. Other
descriptor fits are unchanged. CSV exports carry frozen metric definitions and
implementation hashes.
