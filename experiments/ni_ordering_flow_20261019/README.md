# Ni ordering: observed history, regional context and dense response

The October18 direct joint-flow predictor retained fluctuations but did not beat
negative local-entry Brier. This comparison asks whether longer observation,
regional context, predictive compression or dense physical supervision helps.

## Frozen treatments and budgets

Three seeds20261025/26/27 for each treatment: local4, local13, regional4,
regional13, bottleneck128, dense_value and dense_derivative. All consume the
same native origins and predict the same60 future244-measurement states over6ps.
Four and13 observed frames span0.3 and1.2ps at exact0.1ps. Regional arms add87
complete-ball density/motion measurements at12 Angstrom, only in observed past.
Local arms supply zeros in those87 channels. No temperature, age, absolute time,
box, phase or future context enters the predictor. History token order and flow
noise-to-data integration coordinate are distinct from physical-time covariates.

Every arm uses a two-layer128-wide observed-history attention encoder and the
same four-layer128-wide joint-path flow. The baseline is retrained with this
shared encoder, so comparison with October18 is not an isolated history effect.
The bottleneck arm pools all history into one128-vector before conditioning the
flow; other arms retain individual history tokens. It is trained jointly for
future prediction, without static reconstruction pretraining or crystalline loss.

Each fit has32000updates, batch/microbatch256, AdamW1e-3 maximum learning rate,
1600-update warmup and cosine decay to1% of peak, weight decay1e-5 and gradient
norm clip10. EMA0.999 covers all trained parameters; fixed final EMA is evaluated.
Flow operations areBF16, reductions/optimizerFP32. Every fourth update, dense
arms add32 teacher examples with0.1 auxiliary weight; response contributes0.25
inside that auxiliary. Value-only is the required matched extra-data control.

Dense teachers are the existing630 full-cell deterministic Ni EAM/FS NVE
continuations,21 observations from0 to100fs at5fs. Train320/test310; physical
states and tangents remain float64, force arithmeticfloat32. The shared history
encoder also receives a single present token containing only91 smooth density/
velocity measurements; BO/TDA and regional channels are absent for this task.
An auxiliary head predicts20 future smooth states. Its exact directional
derivative is matched to teacher JVPs through the train-fitted asinh transform.
Only this smooth observed subspace is differentiated: no missing BO/TDA JVP is
silently treated as zero. The primary NPT task still receives full observations.
This is short-NVE auxiliary supervision, not a claim of NPT derivative targets.

## Data and evaluation

Reuse21600 native windows,6720train/28ancestors and14880test/62ancestors. Keep all
original roles as provenance and map nontrain roles to test. Every original origin
has the required exact1.2ps past. Same central atom throughout, nearest80 reselected
per frame. Birth histories are remeasured with this convention, resolving the
historical fixed-membership mismatch;19800 assay rows/19652unique states remain,
with no neighborhood filter or revised birth labels. Regional normalization uses
native train observations only. Central transformations and phase diagnostic are
inherited unchanged from the16-source reference, also train-only. The old cohort
contains future-selected regimes; none of its event frequencies is a natural rate.

Full native evaluation uses32neural samples/input and the existing proper path,
increment and lag-product scores at1/3/6ps. Separate outcome-blind panel at two
rows per source/regime uses256samples and reports raw and Monte-Carlo-corrected
entry Brier. Actual-MD phase-readout and always-negative controls expose readout
and rarity effects. Ordered-liquid occupancy/dwell and fixed trajectory plots
remain secondary diagnostics. No physical-validity statistic blocks these fits.

Birth evaluation compares observed-history, learned-history and generated-future
features with fixed nonlinear likelihood probes and linear logistic diagnostics.
Generated features are endpoint means and standard deviations at1/3/6ps, using
32neural samples. Tasks remain persistent32 and64connected-atom births. AP is
only a diagnostic; no test selection, calibration fitting or early stopping.
Source bootstrap accompanies nonlinear birth and trajectory scores; all seeds,
missing stages and numerical failures are visible. Linear readouts are descriptive
point estimates. Response evaluation compares5/20/50/100fs errors with zero-JVP.

A separate physical branching pilot uses the original NPT potential/integration
settings and an explicitly perturbed velocity distribution; it is not used for
this queue's training or test-based changes. See its simulation documentation.

Run: `python -m src.research.ni_ordering_flow.queue launch --config configs/ni_latent_sim/ordering_20261019.json`.
