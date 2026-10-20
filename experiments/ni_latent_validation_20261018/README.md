# Ni trajectory and temporal-representation validation

Question: did the first comparison fail because of inadequate training, inadequate
source coverage, an unconstrained measurement decoder, or a representation that
does not preserve predictive dynamics? A good present-state nucleus readout is
insufficient evidence for a useful simulator.

The [frozen recipe](../../configs/ni_latent_sim/validation_20261018.json) declares
42 new fits and 21 reference-checkpoint evaluations. All seeds, budgets, final
checkpoints, comparison pairs and evaluation grids are fixed before test scoring.
See [execution](../../docs/ni_latent_validation.md) and the exact
[metric definitions](../../docs/metrics/ni_latent_validation.md).

## Data and physical scope

We reuse actual Ni Mendelev2012 EAM NPT histories at exactly 0.1 ps. Each sequence
follows one central atom, reselecting its nearest 80 neighbors at each observation.
Four observed frames span 0.3 ps; 60 future frames span 6 ps. The 244 measurements
contain bond order, topology, smooth density and relative velocity information.
No temperature, simulation age, absolute physical time or phase labels are model
inputs. Algorithmic interpolation time in a flow model is not physical time.

The original bank had 14,400 windows from 60 ancestors. The expansion adds 7,200
windows from 30 previously unused motion-bank histories with positions and
velocities. Fifteen duration-stratified random anchors and 16 uniform central
atoms per anchor are selected without future-outcome filtering. Cadence is checked
against actual saved arrays. Existing train/test ancestry is preserved.

The combined bank contains 21,600 windows: 6,720 available training windows from
28 ancestors and 14,880 test windows from 62 ancestors. Actual arm training subsets
are smaller, as declared below. The common test bank has 56 strict-liquid local
persistent crystalline entries within 6 ps across 27 ancestors; the previous
464-window forecast panel had only one. These are local entries, not 56 independent
connected nucleus births. Additional histories were previously exposed through
birth assays, so this is not a pristine external replication.

Original and additional test cohorts are reported separately. The new training
sources cover 950–1150 K, while the old sources also include 1200/1400 K. Adding
sources therefore changes environmental coverage as well as sample count. Old
windows include outcome-enriched regimes and histories can end early; aggregate
event frequencies must not be interpreted as natural nucleation rates.

## Frozen comparisons

Each treatment uses seeds 20261025, 20261026 and 20261027.

| Question | Comparison | Fixed training budget |
| --- | --- | --- |
| Does longer training help? | Direct measurements and inherited VAE, block flow versus four-frame autoregressive MDN | 32,000 updates, versus 3,200 previously; fixed snapshots at 3,200/12,800/32,000 |
| Are invalid outputs causing apparent failure? | Raw versus scalar-support direct measurement coordinates, both predictors | Same 32,000 updates and old mixed 3,840-window training set |
| Does source coverage help? | Bounded block/MDN, 16 versus 28 training ancestors | 80 original-regime windows per ancestor: 1,280 versus 2,240; same 32,000 updates |
| Do native temporal representations learn their intended information? | SelfTICA and bidirectional FMRC, 16 versus 28 training ancestors | 48 representation epochs, no decoder or separate propagator |
| How do the earlier encoders compare on these objectives? | 21 frozen linear/VAE/hybrid/SelfTICA/FMRC references, including regional inputs | No retraining; method-native diagnostics and matched future-information probes |

This is 30 predictor fits and 12 encoder-only fits. Predictor batches and
microbatches are 256, AdamW maximum learning rate 0.001, 1,600-update warmup and
cosine decay. Representation fits use a one-epoch warmup and cosine decay. Matrix
operations use BF16 mixed precision with FP32 parameters, optimizers and loss
reductions; no new physical integration is performed. Inherited VAE weights stay
frozen. No broad spatial collection is loaded for motion fitting.

All normalization remains fitted on training ancestors. The old 16-source
measurement normalization fixes common units. Predictor/support transforms use
each arm's actual training subset. The phase classifier, increment scales and
ordering thresholds use the same original training reference for every predictor.
No test scores select checkpoints, hyperparameters, ensembles or thresholds.

## What constitutes evidence

**Connected trajectory prediction:** report proper sample-based scores on whole
paths, increments and increment correlations; also report uncertainty coverage,
temporal fluctuation amplitude, continuous-order persistence and local-entry
Brier scores. Compare with persistence, source-balanced resampling of saved
training futures, and the oracle that encodes/decodes actual future measurements.
The oracle separates information lost by the representation from prediction error.

**Conditional information:** compare genuine input history with a different
history from the same source/regime, using common random draws. MDN teacher-forced
predictions distinguish one-step fit from autonomous error accumulation. Fixed
flow sampler grids and training snapshots distinguish integration resolution from
optimization budget. None chooses a new winning checkpoint.

**Method-native evaluation:** SelfTICA receives its own held-out objective and
nonreversible VAMP-E diagnostics. FMRC receives held-out forward/reverse vector
field risks and samples from its actual conditional-flow heads at 0.5/1 ps.
Those are conditional future marginals, not generated connected 6-ps paths.
Train-fitted Gaussian future probes compare current/history latents against
current/history raw measurements, a prior and persistence at 0.1/0.5/1/3/6 ps.
Matched-order conditioning controls test information beyond current order alone.
No equilibrium, detailed-balance or implied-timescale interpretation is assumed.

**Validity:** the support parameterization enforces necessary scalar ranges,
including nonnegative topological and radial measurements. It does not clip
generated samples. Independent audits test density Gram positivity, the exact
zeroth-order radial identity and velocity moment inequalities, including on saved
MD and encoded MD. Scalar bounds cannot establish joint atomistic realizability.
The native FMRC diagnostic retains its explicitly named legacy scalar audit.

Source-level uncertainty and paired contrasts accompany all applicable scores;
32 neural samples are not 32 independent physical continuations. Report all seeds,
nonfinite generations, missing fits and undefined comparisons. A low energy score
with collapsed fluctuations or failed constraints is not a successful simulator.

## Remaining decisive experiment

This campaign can test conditional predictive information, observed-path quality,
source generalization and local ordering. It cannot establish exact conditional
ensemble calibration: each origin has one saved 6-ps future. Existing short
physical teachers have only two stochastic 100-fs branches per origin (and one
deterministic NVE branch). Repeated physical continuations through several ps,
with the original NPT protocol and a declared branch distribution, remain a
separate data-collection experiment. Full connected nucleus generation also
requires spatially coupled observations beyond independent local trajectories.

## Reproduction and outcomes

Use the [maintained workflow](../../docs/ni_latent_validation.md). Runtime cap is
12 hours from detached launch on two workers; allocation-boundary continuations
use alternative compatible GPU partitions. Incomplete fits remain explicitly
partial. No new scientific result is asserted by this protocol.

Run outputs: [validation-20261018-v2](../../output/ni_latent_flow/validation-20261018-v2).
Historical [first comparison](../ni_latent_flow_20261018/README.md) and its frozen
metric exports remain unchanged.
