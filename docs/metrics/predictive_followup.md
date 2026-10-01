# Predictive baseline: matched heads and target/variance factorial

Recipe: `configs/predictive_baseline/followup_20261001.json`. This follow-up keeps
the sealed Al480 baseline's exact 7,661 observations, 12 branches per parent,
11/3/6 train/selection/historical-test source roles, population inclusion weights,
nine physical future measurements, normalization, 256 RFFs and three block weights.
The original baseline metric definitions remain frozen under `predictive_baseline`.
All reported uncertainties concern six historical held-out sources, not thousands
of independent patches. This is the historical local shooting assay, not Al64's
all64/legacy16 window benchmark. Conditions remain pooled and excluded as inputs.

## Encoders and real inputs

New joint encoders remain native geometry-only MACE128, z128, radius8/nearest80,
5-Angstrom edges, two interactions, angular order2/correlation2, no halo, history,
motion, relaxation, temperature, time or species inputs. Geometry includes the
existing center indicator and constant atom channel; Al's fixed coordinate
multiplier is one. No training-only teachers or reconstruction loss are added.

Frozen controls reuse the recorded VICReg128 and Epi128 exports and add precisely
MM-TDA-BLOCK-DIRECT-FULL's label-free-validation-selected epoch20/update2580
checkpoint, SHA256 f91d2a3cca9b4e57a3fda2c7e190ba8c624780ca5f874a9a9cabf31af6ca4373.
MM-TDA preserves its historical 256 channels/export, depth3/angular3/correlation3,
geometry-only radius8 input, fixed material normalization and original descriptor
training objective. Its actual exporter uses the checkpoint's scalar normalization
and recorded frozen producer, not a substitute plain CapacityEncoder. Pretraining
used 1,056,768 Al/Mg/Ti/Ta patches; its capacity and pretraining data are not matched
to the new MACE128. No MM-TDA pretraining is restarted. Fitting-source paths are
audited from selected packed shards. No exact shooting-source overlap was found;
archived non-native preparation ancestry remains incompletely known and is not
represented as a proven independent preparation.

## Factorial and common selector

Four variants cross full274 versus moments18 supervision with free versus
nonnegative-variance output. Each runs at three declared paired seeds for the
joint encoder and for each of the three frozen encoders: 12 online joint fits
and 36 local frozen diagnostic heads. Frozen heads and ridge probes create no
W&B runs. Joint fits are online with stable resumable IDs.

Every head has 128 hidden SiLU units. Frozen 128- or 256-dimensional exports are
standardized using training weights only. The first layer adapts to the recorded
export width; MM-TDA's larger first layer is disclosed. Each initialization
constructs the same 274-output linear layer within an encoder/seed. Moment-only
variants return its first18 outputs; unused RFF rows receive zero data gradient
and are not presented as predictions. This preserves paired initial weights and
sampler streams. The fixed first18 metric weights are unchanged when the RFF
term is removed; no re-normalization of moment loss accompanies that ablation.

All new neural fits and primary ridge controls use the SAME held-out
18-moment feature likelihood for checkpoint/alpha selection. Selection MSE is
sum squared first18 scaled-coordinate errors, weighted equally across sources;
selection NLL = .5*(selection MSE + 18 log(2pi)). Full supervision adds the
existing RFF block to training loss only. No AP objective or selector. This
predeclared follow-up selector differs from the original full274 selector;
the original baseline predictions appear separately as historical_full_selector.

Training otherwise uses the recorded baseline batch/microbatch256, 17 weighted
resampling updates per epoch, AdamW, encoder/head learning rates1e-4/5e-4,
weight decay1e-5, norm cap10, at most200 epochs and patience25. Geometry and
neural layers use float32; every head's feature transform and squared loss use
float64 to preserve numerical moment consistency. All four joint arms rerun
under this common policy. Initial encoder pooling statistics use training only.
No step-dependent schedule, dropout, or extra label-dependent sampling is used.

Let t denote network output in the baseline's centered/scaled feature units,
a its frozen training phi center and s=sqrt(metric). The free output is t.
For the positive-variance head, m=t_first/s_first+a_first, and
v=softplus(t_second/s_second + inverse_softplus(v_prior-floor))+floor,
where v_prior=a_second-a_first^2 and floor=1e-8 in normalized-Y units.
Return t_first and (m^2+v-a_second)*s_second, plus the unchanged RFF coordinates
when active. Zero network output reproduces the training prior moments.
The population loss still targets E[Y] and E[Y^2], not noisy sample variance.
This enforces nonnegative marginal variance only: support constraints, covariance
positivity and joint feasibility with the RFF predictions are not guaranteed.

## Evaluation

Scopes are mean(0:9), second(9:18), moments(0:18), rff(18:274), full(0:274),
all using the baseline's fixed metric. A moment-only head has no rff/full rows.
For each active scope, error is squared distance to the 12-shot feature mean;
corrected_error subtracts unbiased sample covariance trace/12. Negative corrected
estimates remain. Conditional group scores normalize inclusion weights within
each source/group before equally averaging sources, as in the original baseline.

Predicted physical means and variances invert the fixed feature transform.
Mean MSE compares with 12-shot means; variance MSE with unbiased 12-shot variance.
Neither physical-moment MSE is noise-corrected. Negative variance fraction counts
the nine implied variances before clipping; no clipping is applied. A positive
head fails explicitly if saved predictions imply a negative variance.

Three-seed results average individual errors rather than forming an ensemble.
Source-paired percentile95% intervals resample sources2,000 times after seed
averaging. seed_sd is population SD over source-averaged fit-seed scores. Per-T
subgroups contain only two sources. Intervals omit model/target-selection uncertainty.

Predeclared contrasts report LEFT minus RIGHT, so negative favors left:
full versus moments supervision for each variance/encoder, nonnegative versus
free variance for each target/encoder, frozen nonlinear versus frozen ridge on
the full target, and new joint versus frozen nonlinear full/free models.
The first two comparisons use common moments; the latter use full274 error.
No contrast selects a promoted model or new hyperparameters.

For every selected joint embedding, a local ridge predicts the full274 target,
including embeddings trained on moments only. Alpha is selected on moments18;
full outputs are then fitted at that alpha. These linear_z rows assess retained
information and are separate from a model's trained head. Ridge reference controls
use the same selector. Extra-observable ridge probes retain the prior18 withheld
future descriptors and training-only standardization, with full extra-target
squared error selection. They stay local. Detailed per-source rows are exported.

Final metrics update the existing joint W&B IDs through the API. Source, target,
checkpoint, inference and metric identities remain recorded; all historical
definitions and results are preserved.
