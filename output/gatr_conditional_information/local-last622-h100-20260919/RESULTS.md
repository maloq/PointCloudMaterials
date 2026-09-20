# Latest MACE and GATr: information beyond radial structure

Completed on the user-approved H100 on nodesumo01. Both final training checkpoints are pinned at **update 622**, after five epoch equivalents. Their validation-selected exports were at update 576; this assay intentionally uses the final weights. The newer expanded-data MACE campaign had no checkpoint when this run was frozen. No encoder was trained or modified.

## Fixed population and controls

The test retains the previous 32,040 observations from ten Al trajectories, four tracked atom identities per trajectory, 801 frames at 0.75 ps. The spatial extension retains 22,381 environments from the same 70 snapshots. All test source ancestries are excluded from both encoders’ training and selection. These previously explored sources make this an exploratory comparison, not a blind confirmation.

Each encoder runs with its own hash-verified frozen training implementation, native BF16/FP32 boundaries and compiled execution. Both now crop to normalized radius 8 with a 6–8 taper, about 7.94 Å outer support for Al. They were trained on mixed materials with equivariant bond-order supervision. The previous Al-only checkpoints used larger neighborhoods and different training objectives/budgets; changes do not isolate the effect of bond supervision.

The common radial control contains the previous 152 radial/density features, 33 new local-support radius quantiles and five local radial moments, temperature/time, and **both** new encoders’ radius-only states. Those states use identical sorted radii placed on a fixed Fibonacci angular pattern. Each original state and its difference from its own radial-only state are tested separately. Additional duplication controls add another copy of that encoder’s radial state without adding information.

Linear and nonlinear random-Fourier-feature ridge probes use ten whole-source holdouts and nested source splits for regularization. All scaling uses training sources only. Spatial probes reuse the trajectory-selected settings. Brackets below are 95% whole-source bootstrap intervals stratified by temperature; overlapping frames/pairs are not independent replicates. They do not include refitting uncertainty.

## Structural information

Percentage reductions in held-out prediction error beyond the common radial control; positive is better. The table uses nonlinear probes. Old and new exports are read out under the same new controls.

| Target | Previous GATr | Latest GATr | Previous MACE | Latest MACE | SOAP |
|---|---:|---:|---:|---:|---:|
| q4 | -0.93% [-2.24, +0.47] | -0.44% [-1.47, +0.64] | +6.81% [+4.50, +9.35] | +8.15% [+3.58, +12.29] | +73.64% [+71.35, +75.85] |
| q6 | -0.15% [-1.88, +1.84] | +0.24% [-1.12, +1.66] | +16.53% [+15.09, +17.78] | +21.32% [+17.97, +24.98] | +91.30% [+90.75, +91.94] |
| qbar6 | +3.94% [+2.62, +5.35] | +1.58% [+0.17, +3.05] | +46.48% [+43.05, +49.94] | +46.07% [+43.52, +48.45] | +43.13% [+37.60, +48.29] |
| angular_arrangement | -1.13% [-1.90, -0.15] | -0.78% [-1.63, +0.31] | +13.69% [+8.27, +17.82] | +10.26% [+6.34, +13.32] | +42.47% [+39.16, +45.29] |

## Directly matched angular differences

**609 pairs across nine sources** pass both ≤0.05 Å radial RMS criteria and ≤2% density difference. Matching preserves the previous first-80 and larger-support radius-quantile criteria exactly; changing the encoder support did not select new pairs. The looser 0.10 Å sensitivity retains 125,006 pairs. The sparse four-track population still has only three strict pairs and cannot support an inferential conclusion.

| Matched target contrast | Latest GATr | Latest MACE | SOAP |
|---|---:|---:|---:|
| q6 | +2.47% [-5.10, +15.89] | +7.91% [-0.15, +16.29] | +95.14% [+90.78, +96.33] |
| qbar6 | +17.02% [+5.69, +25.12] | +46.58% [+30.74, +60.53] | +65.78% [+56.71, +70.18] |
| angular_arrangement | +0.55% [-0.96, +1.77] | +2.65% [+1.44, +3.91] | +64.76% [+48.00, +71.30] |

## Structural gains after radial duplication

The preceding tables alone can overstate angular information: adding a nearly redundant state changes the ridge prior and nonlinear kernel. The stronger comparisons below replace a second copy of the corresponding radius-only state with the original state. Both candidates have the same added dimension; no extra angular information exists in the duplicate.

| Population | Target | Original GATr beyond its radial duplicate | Original MACE beyond its radial duplicate |
|---|---|---:|---:|
| All trajectory rows | q6 | -0.03% [-0.06, +0.01] | +21.55% [+17.85, +25.22] |
| All trajectory rows | qbar6 | +0.01% [-0.13, +0.14] | +46.59% [+44.47, +48.66] |
| All trajectory rows | angular_arrangement | -0.01% [-0.01, -0.00] | +10.38% [+6.45, +13.36] |
| 609 matched pairs | q6 | -0.03% [-0.19, +0.14] | +6.49% [-8.86, +15.52] |
| 609 matched pairs | qbar6 | +0.04% [-0.45, +0.79] | +31.65% [+24.83, +43.04] |
| 609 matched pairs | angular_arrangement | -0.01% [-0.05, +0.00] | +3.06% [+1.40, +7.17] |

For example, the apparent matched GATr qbar6 gain above becomes +0.04% [-0.45, +0.79] against redundant radial input. This control must accompany the raw R-star gains.

The explicitly scaled GATr angular difference yields an angular-moment gain of +2.48% [+1.45, +3.30] on all trajectory observations, but -3.86% [-6.95, -2.42] on strict spatial pairs. At the looser 0.10 Å spatial threshold the gain is +0.96% [+0.50, +1.34]. Thus small recoverable angular responses should not be described as literally absent.


## Future crystallization

There are 17,674 eligible prospective rows before first sustained local FCC/HCP/BCC onset, with the current and previous two frames noncrystalline. Eight consecutive crystalline frames confirm onset; all 24/48/96 ps horizons include complete confirmation. This is one realized future, not an iso-configurational propensity or committor.

The stricter nonlinear comparisons below control redundant radial inputs. The final two columns also condition on the six current bond-order and sixteen angular descriptors, then add the encoder’s angular difference. Scores are reductions in Brier error.

| Horizon | GATr original / duplicate control | MACE original / duplicate control | GATr after current order | MACE after current order |
|---|---:|---:|---:|---:|
| 24 ps | +0.0044% [-0.0009, +0.0102] | -0.0198% [-0.0832, +0.0271] | -0.11% [-0.30, +0.05] | -0.16% [-0.30, -0.02] |
| 48 ps | +0.0007% [-0.0075, +0.0079] | -2.4582% [-5.5711, +0.1156] | -0.06% [-0.50, +0.41] | -0.47% [-1.15, +0.13] |
| 96 ps | +0.0064% [+0.0003, +0.0123] | -0.3237% [-0.6344, +0.0052] | -0.12% [-0.85, +0.63] | -1.17% [-2.27, -0.02] |

Absolute Brier errors are needed to judge forecasting usefulness, not only relative increments:

| Horizon | Training prevalence only | Radial control | Radial + GATr | Radial + MACE |
|---|---:|---:|---:|---:|
| 24 ps | 0.05381 | 0.05408 | 0.05403 | 0.05399 |
| 48 ps | 0.10746 | 0.10875 | 0.11103 | 0.11172 |
| 96 ps | 0.19279 | 0.20420 | 0.20332 | 0.20357 |

An absent gain with these finite, clipped least-squares readouts does not prove absence of future information. If absolute errors do not beat prevalence, useful forecasting has not been demonstrated. There are forty tracks and ten independent source replicates; no causal effect is identified.

## Temporal stability

Consecutive 0.75 ps RMS change is divided by the independent-pair distance scale from the original five training-reference sources. Smaller is smoother relative to that representation’s variability. Smoothness is useful only alongside retained structural information.

| Representation | Normalized RMS jump | Reversing successive increments | Effective rank on reference |
|---|---:|---:|---:|
| GATr previous | 0.680 [0.659, 0.701] | 89.0% | 6.28 |
| GATr latest | 0.518 [0.502, 0.533] | 83.9% | 1.87 |
| MACE previous | 0.723 [0.696, 0.749] | 74.6% | 1.87 |
| MACE latest | 0.716 [0.684, 0.746] | 73.7% | 1.66 |
| SOAP | 0.587 [0.571, 0.601] | 85.1% | 2.76 |
| TDA | 0.371 [0.369, 0.372] | 78.1% | 1.20 |

## Artifacts and reproduction

- [Structural comparison](plots/conditional-information.png)
- [Strictly matched environments](plots/spatial-matched-information.png)
- [Structural gains after radial duplication](plots/structural-duplication-control.png)
- [Future crystallization controls](plots/future-crystallization.png)
- [Trajectory stability](plots/trajectory-stability.png)
- [Metric definitions](tables/METRICS.md); source scores, linear/nonlinear results and matching sensitivities in `tables/`.
- Frozen checkpoint identities, producer paths, native inference checks and source-held-out predictions in `technical/`.

```bash
conda run -n pointnet-torch214 python -m src.research.gatr_conditional_information.comparison \
  --config configs/analysis/conditional_information_local_last.json
```
