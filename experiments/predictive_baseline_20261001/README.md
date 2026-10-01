# Can present local geometry retain the selected future distribution statistics?

Train an encoder and a prediction head jointly against repeated-shooting feature
means. This implements the value-only baseline from the user's formulation using
the already available local Al observations. The local and pooled-condition scope
is explicit: this experiment does not observe the full cell or hold temperature
and cell fixed across the entire parent population.

The data are the sealed historical 40-parent/480-future Al release, with 7,661
local observations. Preserve its 11 training, three selection and six historical
test sources. The 12 futures per parent are compatible Langevin shots; the CSLD
top-up and nested ensembles are excluded. No new simulations or source resplit.

At 3/6/12 ps measure crystalline fraction, qbar4 and qbar6. Fix a 274-coordinate
feature map consisting of nine standardized observations, nine squares and 256
random Fourier features. Fit all normalization, bandwidth and block scaling on
training sources, then hold them fixed. Average features across shots before
training. Equal block weights measure information about means, variability and
joint future structure without letting the RFF count dominate.

Three random initializations train native geometry-only MACE128 -> z128 ->
274-coordinate head with batch/microbatch256. Fixed Gaussian feature likelihood
selects checkpoints on held-out selection sources. Compare to the training prior,
442-descriptor ridge and frozen VICReg/Epi ridge. Independent selected z readouts
separate predictive performance from linear accessibility; extra future-observable
probes ask narrower transfer questions. Coordinate-noise response is diagnostic.

Report held-out feature error, an unbiased finite-shot correction, raw physical
mean/variance errors and source-paired intervals. Unconstrained second moments can
imply negative variance; retain and quantify violations. Evaluate liquid-only and
per-temperature groups in addition to the full population. Fit-seed variability
does not create new physical sources. A finite feature target and z128 bottleneck
do not prove full-law identification, minimality or useful Euclidean latent geometry.

Recipe: [`al480_20261001.json`](../../configs/predictive_baseline/al480_20261001.json).
Reproduction: `python -m src.research.predictive_baseline.queue submit --config configs/predictive_baseline/al480_20261001.json`
after the [documented preparation and numerical gate](../../docs/predictive_baseline.md).
Results: [`output/predictive_baseline/al480-274-20261001`](../../output/predictive_baseline/al480-274-20261001).
Exact calculations: [metric specification](../../docs/metrics/predictive_baseline.md).

No scientific result is claimed before completion. The 274-target metric differs
from the earlier 24-observable RFF reliability study and must not be compared as
if its numerical errors shared the same scale.
