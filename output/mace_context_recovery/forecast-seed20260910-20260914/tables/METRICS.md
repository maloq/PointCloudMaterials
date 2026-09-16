# MACE context information recovery

The experiment uses the exact retained cohort of the context pilot: 5,760 anchors,
18/6/6 whole-source splits, and 144 test tracks with 17 frames at 0.75 ps cadence.
It is an exploratory cohort previously examined, not a fresh confirmatory sample.
No test labels select readout weights, regularization, epochs or training recipes.

Cached probes compare the original, smooth-inner and center embeddings with
their concatenation, before and after pilot training. All individual embeddings
are 256D. Fusion is 512D; its increased width is reported, not treated as a
matched-capacity architectural improvement. Frozen fusion uses features of the
same original backbone. Trained fusion combines two separately optimized
backbones and is a complementarity diagnostic, not a single-backbone model.

The targets are the original 144D hot and 144D relaxed TDA descriptors and four
nearest-12 observables, q4, q6, shell density and mean distance. TDA target blocks
use the exact training-only fit_targets scales, with floor fraction 0.05 and equal
H0/H1/H2 weights despite their 16/64/64 widths. Structural targets use their
training standard deviations. Means also use training rows only. Raw test MSE
columns for TDA are dimensionless balanced errors, not percentage accuracies.

Ridge uses float64 SVD and the pilot's alpha grid. Each of hot, relaxed and the
four-observable group selects its alpha on validation data. Features are centered
and standardized using training rows; exactly constant channels use scale one.
`alpha_at_grid_edge` flags a selection at a grid endpoint.

The nonlinear probe is a residual MLP added to the fixed ridge prediction. Three
independent heads each have one 128-unit SiLU hidden layer; the output layer starts
at zero. Full-batch AdamW uses the configured learning rate, two weight decays,
two initializations, at most 400 epochs, validation every ten epochs and 60-epoch
patience. Each head independently selects its configuration and epoch using its
validation error, including the unmodified ridge prediction as epoch zero.
The initializations are validation candidates, not replicates used to estimate
training-seed uncertainty. Encoder embeddings never change during cached probes.
Selected coefficients, nonlinear states, validation histories, target transforms,
predictions and source errors are retained. Predictions are archived float32;
ridge scoring and accumulation use float64 before that archival round trip.

`q6_test_reduction` is 1 minus normalized test squared reconstruction error divided
by the training-mean baseline error, for the absolute q6 value. Temporal increment
reductions instead decode both observed endpoints and evaluate their difference
against zero-change persistence. They are not future predictions from the first
endpoint. `q6_increment_reduction_075ps`, density and distance use individual
observable MSEs; `hot_increment_reduction_075ps` balances the three TDA blocks.
Higher reduction is better; negative means worse than persistence. All configured
lags and all four observables remain in the machine scores.

Latent temporal MSE scales each 256D feature block by the square root of its mean
training feature variance, then averages squared temporal increments. This equals
the original scalar normalization for single blocks and gives each half of fusion
equal training variance. The denominator is training-only. Per-block changes are
also retained, so concatenation cannot hide its center contribution. These metrics
include physical evolution and do not measure noise alone. They are invariant to
uniform rescaling of each block, not arbitrary channel transformations.

The original 72 controlled rank-80 crossings and four epsilon values are reused.
`boundary_latent_fraction` is the smallest-epsilon squared latent change divided
by that representation's natural 0.75 ps change, with the same block scales.
`boundary_decoded_hot_fraction` measures the hot-TDA readout's crossing change
relative to the observed natural hot-TDA change. This catches a readout amplifying
small continuous feature differences. The entire curve remains in JSON. The
legacy TDA target itself has a finite boundary jump and is not redefined here.

Paired error reductions use 4,000 resamples of six whole test sources. Cached
alternatives compare with the original encoder using the same frozen/trained
state and readout class. Joint-training alternatives compare with their matched
dual-SSL control when it is available. Intervals are exploratory and conditional
on fitted encoders and readouts; overlapping temporal pairs are not independent.

Joint training is a distinct 12-epoch warm-start protocol from the original
forecast checkpoint. One shared MACE backbone emits the smooth-inner 256D block
and tracked-center 256D block in one graph pass. Both variants receive identical
initial weights, training source rows, augmentation draws and warm-started physical
heads. Heads are initialized on cached frozen fusion features for 200 full-batch
epochs, selecting their aggregate validation loss. Their input standardization
uses frozen training features and remains fixed throughout encoder optimization.

`dual_ssl` updates its encoder with the original spatial/temporal VICReg objective
on the inner block only. Its physical heads receive detached encoder features.
`dual_physics` additionally lets physical-head gradients update both encoder
blocks. Each batch supplies three original augmented SSL views and a fourth,
unaugmented anchor view matching the supervised labels. No hot/relaxed target is
assigned to the jittered SSL views. The supervised objective is the mean of hot,
relaxed and four-observable group losses, multiplied by the configured weight 10.
The original VICReg projector, invariance/variance/covariance weights and temporal
weight are unchanged. Local center features are not directly forced to agree
across spatial/temporal views; the shared backbone still receives inner SSL gradients.

Both variants use batch 256, encoder learning rate 1e-4, head learning rate 1e-3,
AdamW with original weight decay, and separate encoder/head gradient clipping.
Separating clipping prevents control-head gradients from indirectly rescaling
the control encoder update. Complete-batch VICReg gradients are replayed through
encoder microbatches of eight, with direct-autograd equivalence checked on GPU.
The control omits replay of the fourth view because its encoder gradient is zero.

Lowest aggregate physical-head validation loss selects the checkpoint in both
variants, including their common epoch-zero initialization. Encoder optimization
in the control uses no physical labels, but its checkpoint selection does use
validation physical labels. This is not the original pilot's SSL-only selection
rule. Both last and best checkpoints preserve model/head/optimizer states,
normalizations, source and implementation hashes, and RNG states. Selected
embeddings receive a newly fitted ridge readout using the same training/validation
protocol; `trained_head` separately evaluates the selected supervised heads.
No held-out temporal trajectories train the encoder or select checkpoints.
The cached-readout plot displays the retained trained representations with ridge
and nonlinear probes; during initial partial collection it displays the available
frozen representations. It is refreshed from score tables without model fitting.


Table export: 2026-09-14T20:56:18.535059+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
