# Conditional information in the latest local MACE and GATr

September 19 follow-up to the [original v6 assay](README.md). Does the exported
state now retain bond/angular information beyond radial structure and density,
and does it improve prospective local crystallization or trajectory smoothness?

The user requested the latest checkpoints of both models and approved H100
execution because node07's A100s were occupied. The pinned weights are the
**final update 622** of `mace-local-20260918` and `gatr-local-bond-20260918`,
not their validation-selected update-576 exports. Both are completed five-epoch
local, mixed-material, bond-supervised fits. The expanded-data MACE campaign
had no checkpoint when the assay was frozen.

[Results](../../output/gatr_conditional_information/local-last622-h100-20260919/RESULTS.md),
[figures](../../output/gatr_conditional_information/local-last622-h100-20260919/index.html),
[metric definitions](../../docs/metrics/conditional_checkpoint_comparison.md),
and [pinned recipe](../../configs/analysis/conditional_information_local_last.json).

## Findings

**Completed.** MACE retains substantial conditional bond/angular information;
the original GATr state remains nearly redundant with radial controls. The
nonlinear reductions in trajectory prediction error beyond each model's radial
duplicate are:

| Target | GATr original state | MACE original state |
|---|---:|---:|
| q6 | -0.026% [-0.065, +0.013] | +21.55% [+17.85, +25.22] |
| qbar6 | +0.010% [-0.126, +0.141] | +46.59% [+44.47, +48.66] |
| Angular moments | -0.007% [-0.013, approximately 0] | +10.38% [+6.45, +13.36] |

The apparent +17.02% matched GATr qbar6 gain against R-star falls to +0.043%
[-0.452,+0.791] against its radial duplicate. Added redundant inputs account
for that apparent benefit. Strict spatial pairs still support MACE qbar6 and
angular information: +31.65% and +3.06% with nonlinear probes. Its strict q6
gain is uncertain with nonlinear probes but positive with linear probes.

GATr is not literally angular-information-free. Subtracting its radial-only
state exposes +2.48% [1.45,3.30] angular-moment improvement on trajectory rows
and +0.96% [0.50,1.34] on looser spatial pairs; it fails on strict spatial pairs.
Those are small signals and do not establish useful forecasting.

Neither model demonstrates practically useful additional first-onset prediction.
No positive interval survives for either angular-difference state after current
bond/angular order and radial duplication. The original GATr has tiny positive
increments (~0.01%) in some nonlinear future comparisons; these should not be
confused with useful probability prediction. Both original-state readouts remain
worse than training-prevalence Brier error at all three horizons.

Normalized 0.75 ps RMS jitter falls from 0.680 to 0.518 for GATr (**23.9%**),
while reference effective rank falls from 6.28 to 1.87. The variation is more
concentrated, so smoothness alone is not evidence of a richer state. MACE jitter
changes little, 0.723 to 0.716 (**0.9%**). Under the identical common radial
baseline, the latest MACE reduces q6 readout error by 5.74% versus its previous
checkpoint, but angular-moment error increases by 3.97%; there is no uniform
improvement across targets.

## Protocol

Keep the previous ten test trajectories, forty tracked atom identities and
32,040 observations. Keep the 70 spatial panels and all 22,381 environments.
Matching uses the identical old radial/density values, giving the same 609
strict spatial pairs across nine sources and 125,006 pairs at the looser
threshold. No angular outcome or new embedding selects a pair. All test source
ancestries are excluded from both models' training and selection.

Run each checkpoint using its own hash-verified frozen training implementation.
Use its native radius-8 crop and 6–8 smooth taper, fixed Al coordinate scale,
compiled execution and BF16/FP32 boundaries. Check repeatability, batch order
and proper-rotation invariance on actual observations. Export only the native
z128; the auxiliary equivariant bond-order heads are not included.

The common radial baseline retains the old radial/density controls, adds
radius quantiles and moments for the smaller local support, and includes both
models' radius-only states from the same deterministic Fibonacci construction.
Fit the original state and its angular difference separately. Duplicate each
radial-only state as a matching dimension/regularization control. For prospective
prediction, repeat these comparisons after current bond/angular conditioning.
SOAP and TDA remain physical comparators. The old MACE/GATr exports are freshly
read out under the same new common controls for a direct generation comparison.

Nested ridge and random-Fourier-feature readouts hold out whole simulation
sources. Their preprocessing and regularization use training sources only.
Spatial readouts reuse the trajectory-selected settings. Targets, risk
eligibility, 24/48/96 ps horizons, source weighting and source bootstrap retain
the original definitions. Absolute Brier errors against training prevalence
are reported alongside conditional gains.

Jitter is remeasured on the same forty trajectories, normalized separately for
each representation using the same 420 training-reference observations. The
report includes normalized 0.75 ps jumps, lag-dependent changes, successive
increment reversals and effective rank. Smoothness must accompany retained
information; reduced variation alone is not a success criterion.

This is an exploratory repeat on previously inspected sources. The models
changed support, data and training objectives as well as weights, so the
comparison does not isolate bond supervision or an architecture effect. A
finite readout's absent gain does not prove information absence. The observed
future is one trajectory per initial condition, not an iso-configurational
probability or a causal estimate.

## Reproduction

On the approved H100, using conda `pointnet-torch214`:

```bash
python -m src.research.gatr_conditional_information.comparison \
  --config configs/analysis/conditional_information_local_last.json
python -m pytest -q tests/test_gatr_conditional_information.py \
  tests/test_conditional_checkpoint_comparison.py
```

Stages `prepare`, `extract`, `probe`, `report` support resuming. Preparation
freezes checkpoints and verifies the cohort; extraction imports the exact
producer for each model in separate subprocesses. Metric tables freeze their
definitions and source hashes. Eight targeted tests passed, including actual
native input/edge parity after the new preparation crop.
