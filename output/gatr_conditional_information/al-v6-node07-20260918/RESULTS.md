# Conditional information beyond radial structure and density

**The tested GATr export carries very little additional angular information after radial conditioning. Its small apparent forecasting gains are largely reproduced by redundant radial inputs.** The dense matched-environment extension supplies a positive SOAP control and supports the structural finding.

Completed on the A100 on node07, using frozen GATr Al checkpoint 1216. No encoder was trained or modified. Probes were fitted separately and evaluated on unseen simulation sources.

## Question and controls

Does z128 add useful information about local angular structure or later crystallization among radially similar environments? We reused 32,040 observations from ten Al trajectories, four atom tracks per source. All encoder test ancestries were verified in the parent audit. This is exploratory reuse of that cohort.

**R** includes the inner radial descriptor, 80 sorted neighbor radii, 33 radius quantiles across native support, counts/radial moments, local density, coordination, temperature and elapsed time. The stronger control **R-star** additionally includes a radius-only GATr control: the identical radius multiset is placed on a deterministic Fibonacci angular pattern and re-encoded. This uses the same checkpoint but removes the original angular arrangement. **Angular difference** means z(original) − z(radial control). It is computed from the exported state, not internal vectors. Together with the radial-control state it is an invertible reparameterization of the original state; it isolates small angular responses for finite-capacity probes. The replacement geometries are synthetic, not physical trajectories.

Every outer fold holds out one complete source; three inner source folds select ridge regularization. Both linear and nonlinear random-Fourier-feature probes are reported. Training samples every fourth frame; held-out evaluation uses every eligible frame. All feature/target scaling uses fit sources only. Intervals resample sources within temperature; they do not treat frames as independent or include refitting uncertainty.

## Bond order and angular arrangement

Numbers are percentage reductions in held-out squared error relative to R-star (positive is better), with paired 95% source intervals. Angular arrangement comprises 16 rotation-invariant Legendre moments, not absolute laboratory orientation.

| Target | Add original GATr z128 | Add its angular difference | Add SOAP |
|---|---:|---:|---:|
| q4 | +0.0% [-1.2, +1.1] | +0.7% [-1.0, +2.2] | +73.9% [+71.7, +76.0] |
| q6 | -1.2% [-2.3, -0.1] | -0.9% [-1.7, -0.1] | +91.5% [+91.0, +92.1] |
| qbar6 | +0.2% [-4.7, +4.7] | -0.8% [-6.3, +4.8] | +46.0% [+40.6, +51.7] |
| bond_order | -0.1% [-0.6, +0.3] | -0.1% [-0.9, +0.7] | +39.3% [+36.0, +42.8] |
| angular_arrangement | -0.3% [-0.6, +0.1] | +0.2% [-0.4, +0.7] | +43.7% [+40.3, +46.5] |

The original state has an interval above zero for: **none of these individual target groups**. This is evidence about recoverable information under the tested controls and probes, not a proof of conditional independence when gains are absent.

## Directly matched environments

Primary matching retains **3 of 48,060** same-source, same-frame pairs. Both the first-80 radial RMS gap and full-support radial-quantile RMS gap must be ≤0.05 Å, with ≤2% relative density difference. Matching uses no embeddings, angular targets or future outcomes. This primary sample is too small for an inferential conclusion. The 0.025 Å sensitivity has no matches. Pairs can share atoms/frames, so uncertainty is source-level.

The predeclared 0.10 Å sensitivity retains **1,267 pairs** across ten sources. The following table uses that looser population and scores predicted target contrasts:

| Matched target contrasts | Add original GATr | Add angular difference |
|---|---:|---:|
| q6 | -5.8% [-9.0, -1.4] | -7.0% [-12.7, +1.4] |
| qbar6 | -3.4% [-10.8, +2.2] | -2.7% [-11.6, +3.9] |
| angular_arrangement | -0.2% [-0.9, +0.6] | +0.1% [-0.4, +0.7] |

These score predicted A−B target differences, not only each environment separately. The dense spatial follow-up keeps the original 0.05 Å threshold and is reported separately in [the spatial extension](RESULTS_spatial.md); it was added to address match scarcity.

## Future crystallization

There are **17,674** eligible prospective observations. The atom must have been PTM-noncrystalline for the current and previous two frames and be before its first sustained FCC/HCP/BCC onset. A sustained onset starts eight consecutive crystalline frames. All horizons have full follow-up, including confirmation; incomplete terminal windows are excluded. The outcome is first sustained local crystallization within 24, 48 or 96 ps, not a committor probability.

The outcome probe uses clipped least-squares probabilities. Primary loss is Brier error; source AUROC and average precision are supplementary. These are nonlinear readout results:

| Horizon | Add original GATr to R* | Add angular difference to R* | Add angular difference after also controlling current bond/angular order |
|---|---:|---:|---:|
| 24 ps | +0.1% [-0.2, +0.5] | +0.3% [+0.2, +0.6] | +0.2% [+0.0, +0.4] |
| 48 ps | +0.5% [+0.3, +0.6] | +0.5% [+0.5, +0.6] | -2.8% [-6.0, -0.1] |
| 96 ps | +1.4% [+0.1, +2.6] | +1.5% [+0.0, +2.9] | +1.4% [+0.0, +2.8] |

Adding a nearly redundant state can change ridge penalties and random-feature geometry. A stricter control appends a second copy of the radial-only GATr state, matching the added 128 dimensions without adding information. The following comparison replaces that duplicate by the original state or angular difference:

| Probe | Horizon | Original GATr beyond duplicated radial state | Angular difference beyond duplicated radial state |
|---|---|---:|---:|
| linear | 24 ps | +0.0008% [+0.0001, +0.0016] | -0.1990% [-0.3775, -0.0341] |
| linear | 48 ps | +0.0022% [-0.0007, +0.0049] | -0.0093% [-0.0983, +0.0730] |
| linear | 96 ps | +0.0018% [+0.0005, +0.0031] | +0.3378% [-0.0225, +0.7094] |
| nonlinear | 24 ps | +0.0046% [-0.0008, +0.0098] | +0.2458% [+0.0257, +0.4210] |
| nonlinear | 48 ps | -0.0008% [-0.0093, +0.0075] | +0.0398% [-0.0525, +0.1453] |
| nonlinear | 96 ps | -0.0192% [-0.0382, +0.0012] | +0.1338% [-0.0333, +0.3024] |

After conditioning on current order **and** the duplicate-radial control, the nonlinear interval is above zero at: **none of the tested horizons**. The same comparison has no positive interval in the linear probe either. Thus the small R-star-only gains do not establish a useful extra angular forecasting signal. Matched future tables also report discordant-outcome counts and ranking concordance; sparse discordant pairs limit interpretation.

The risk readouts also need an absolute reference. The nonlinear original-GATr readout does not beat the training-prevalence-only predictor in source-mean Brier error at any horizon:

| Horizon | Training-prevalence Brier | R-star + GATr Brier |
|---|---:|---:|
| 24 ps | 0.05381 | 0.05415 |
| 48 ps | 0.10746 | 0.10932 |
| 96 ps | 0.19279 | 0.20501 |

This limits forecasting conclusions to a lack of demonstrated benefit in these finite readouts and forty tracks. It is not proof that no more capable, well-calibrated predictor could extract information.

## Interpretation and limits

Invariant z can encode angular arrangement through invariant functions of geometry. Useful angular information therefore does not require the previously inspected internal arrows to be persistent. Tiny angular changes in z can be informative after scaling; their magnitude alone is not an information test.

The controls are rich but finite and readouts approximate: a gain can partly reflect easier access to residual radial information. Matched contrasts and the radius-only counterfactual strengthen the test but do not establish causation. The radial replacement is out of distribution, input coordinates originate from float16 storage, and only one checkpoint/ten previously explored sources are covered. Future outcomes are one realized trajectory per initial condition, not replicated iso-configurational futures. No encoder retraining or directional-head proposal was tested.

## Artifacts

- [Main comparison](plots/conditional-information.png)
- [Linear/nonlinear probe comparison](plots/probe-sensitivity.png)
- [Radial-input duplication control](plots/radial-duplication-control.png)
- [Illustrative matched environments](plots/matched-example.png)
- [Dense spatial matching extension](RESULTS_spatial.md) and [its figure](plots/spatial-matched-information.png)
- [Exact metric definitions](tables/METRICS.md); source, matching and conditional-gain CSVs in `tables/`.
- Inputs, held-out predictions, selected penalties, hashes and hardware record in `technical/`.

Reproduction on node07/A100:

```bash
conda run -n pointnet-torch214 python -m src.research.gatr_conditional_information \
  --config configs/analysis/gatr_conditional_information.json
```
