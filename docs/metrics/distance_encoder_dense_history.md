# Six MD observations: nominal 0.75 ps, actual 0.75/0.70 ps

CD-MACE128-D6-075nominal uses six saved observations of the same tracked atom.
Native Al uses exact 0.75-ps intervals, offsets -3.75/-3/-2.25/-1.5/-0.75/0 ps.
External Al/Mg/Ti/Ta uses every seventh 0.10-ps saved frame: exact intervals
0.70 ps and offsets -3.5/-2.8/-2.1/-1.4/-0.7/0 ps. The user explicitly accepted
this approximation on 2026-09-26. No interpolation or alternating 7/8 strides.
Each sequence has constant spacing; the two source groups have different actual
physical spans. The model receives no time/cadence values, temperature, species,
material ID, scale, velocity or surrounding-patch embedding.

Raw saved timestamps are checked; extraction and the loader enforce the declared
per-kind cadence, and record actual offsets for every source. The common model
configuration uses the nominal offsets only to describe six ordered positions.
No source-specific value is fed into the model.

There are 11,375,472 training and 190,080 Al selection windows, identical to H6:
117 train sources (90 native Al and 27 external trajectories) and 15 selection
sources. Native source roles are the fixed Al64 contract; external ancestry is
train-only. Omit the first two retained 3-ps label frames uniformly. Train loss
gives equal material mass; selection gives equal Al source mass. Both dense arms
use the same population, labels, initial distance checkpoint, seed, batch and
12 epochs. The initialization already saw the external training families, which
remain training data. The head is larger than the three-frame H6 head, so this
is not a pure cadence-only ablation; use the matched repeated-current control.

Six shared-MACE 128-vectors and their five adjacent increments are concatenated
in oldest-to-current order. A two-layer 128-wide MLP produces a residual on the
current embedding, followed by the distance-distribution head. Both MACE and
predictor are trained end to end. The repeated-current control has the identical
six-frame architecture and fitting population but repeats one differentiable
current embedding. Its redundant deterministic encoder evaluations are eliminated.
The six-frame predictor has more parameters than the three-frame predictor;
compare each against its own matched control before attributing gains to cadence.

Coordinates are extracted from existing trajectories, normalized by the fixed
material length factor and stored in float32. There is no additional coordinate
quantization. Overlapping windows share frame/center geometry. Existing anchor
patches are copied from the checksummed parent release; intervening raw frames
use the same 80-neighbor periodic extraction. Model features are never cached
during fitting. All raw IDs, timeline cadence, source manifests, parent label
checksums and geometry receipts are bound into the release/run identities.

The effective global batch remains 1024 sequences. Two GPUs each accumulate
two microbatches of 256 sequences (1536 encoded patches per forward for real
history). Each microbatch contribution is multiplied by world_size divided by
the actual global sample count; DDP averages across ranks. Zero-weight padding
is excluded. Gradients accumulate before a single clipping and optimizer step.
The sum thus preserves the same weighted global likelihood, including a partial
last batch. No batch-moment regularizer is used; accumulated VCReg is explicitly
rejected. Throughput is sequences per optimizer update per wall second, not
individual frame encodings. cuEquivariance/compiled MACE use BF16 autocast with
FP32 likelihoods and master weights; geometry is a resident float32 GPU bank.

Training, validation and selection retain the `distance_encoder` objective:
zero-inflated lognormal NLL, censored at 64 Al-equivalent Å, plus twice the
Bernoulli log losses at 8/12/20/32 Å weighted .05/.15/.40/.40. The target is the
current nearest atom of a causally confirmed >=64-atom crystal component.
Selection begins at epoch 12 and uses only this predictive objective, never AP
or test warning distance. `validation.csv` reports combined objective, distance
NLL, early log loss, capped-mean RMSE and Brier scores at 20/32 Å.

## Spatial-front evaluation

The `distance_encoder_history` evaluation protocol is unchanged except for the
six actual MD frames. Every original fixed held-out row and scan-position atom
is retained. The test contains 45,291 fixed observations from 30 sources and
495 approach/292 far-control paths from 28 sources. Each spatial position uses
its own tracked atom's preceding MD frames, not previous spatial positions.
Current geometry/targets/visibility are checked against the old spatial producer.

`distance.csv` reports equal-source continuous NLL, 64-Å capped-median MAE,
capped-mean RMSE and CDF Brier scores at 4/8/12/20/32 Å. Current Al distances
are physical Å. Infinity is censored for likelihood and capped for point errors.
`alarms.csv` and `paths.csv` use P(distance <= R) at strict thresholds >.5/.75/.95
for each R. The primary rule requires two consecutive spatial positions;
single-position alarms are secondary. Warning distance is measured at the first
completed alarm. Conditional quantiles exclude misses, which are reported.
Recall at D divides detected paths with warning >=D by all approaches; false
alarm rate divides alerted far-controls by all controls. Far-controls stay >32 Å;
near misses and tangential approaches are absent.

Visibility includes every MD observation actually supplied at every contributing
alarm position. The history model uses all six; repeated/current controls use
only current observations. Raw predictions also retain current-only visibility.
Clear-history subsets consequently differ by model and are not matched cohorts.
An early-clear alarm also requires warning >8 Å. These are recognition/warning
diagnostics about existing crystal, not future nucleation or onset probabilities.

`confidence-reliability.csv` gives equal source mass before subsetting. Precision
and mean probability normalize retained mass, coverage divides by subset mass,
and empty denominators are blank. Probability thresholds do not guarantee empirical
precision. Single-seed pooled path rates have no source-bootstrap or training-seed
uncertainty claim. The inherited temporal-onset AP tests are not this experiment's
primary task or a training/selection objective.


Tracking revision (2026-09-26): diagnostic frozen readouts and per-checkpoint
evaluations keep their logs and results locally. Associated final scores update
a recorded scientific training run through the API, without creating or
restarting runs. Scientific training remains online. This changes logging and
validates identity/hash before cached readout reuse; objectives, selectors,
metric calculations and historical exported definitions are unchanged.

Implementation revision (2026-09-27): the shared trainer also supports full-history checkpoint initialization and explicit material subsets for the separate material-adaptation protocol. Historical calculations and frozen exports are unchanged.
