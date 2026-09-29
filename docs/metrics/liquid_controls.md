# Liquid signal sensitivity and rich-feature prediction

This is a distinct, one-seed assay. Synthetic labels are diagnostic generated
distances, not physical crystal distances or future events. No temperature, time,
material ID, phase label, source identity, distance or clearance enters a model.
Source and frame IDs organize splits, pairing, label draws and uncertainty only.

## Known and zero signal

All 325,970 original eligible rows and their four frozen roles/conditional weights
are retained. The feature bank and 25-patch coordinates are unchanged. The local
generator uses `shell4_mean/bond_order/l6_q`; the spatial generator uses
`gradient_norm/bond_order/l6_q`. A generator is `tanh((feature-training_mean)/
training_sd)`. No held-out statistics tune it. The null generator is identically 0.

Let p0 be weighted training frequencies plus 1e-8 in the original ten distance
bins. Conditional probabilities are softmax(log(p0) + eta*g(X)*a), where a is the
standardized bin midpoint, using p0's within-bin-inclusive distance variance.
Finite-bin distances are uniform; the censored tail is represented by 64 Å.
The exact conditional first and second moments therefore include within-bin
variance. A root solve chooses eta using training inputs only, such that
1-sqrt(E conditional_variance / total_variance) is 1%, 2% or 5%. The null has eta=0.
The oracle gain is recomputed, never retuned, on held-out inputs. Conditional
categorical mutual information is E sum_b p(b|X) log(p(b|X)/E p(b|X)).

One fixed synthetic draw per original source/frame/query is shared by duplicate
sampling rows. Common random uniforms couple all signal strengths; no predictor
receives these IDs or uniforms. The training seed and synthetic-label seed are
separately recorded. This is not an empirical estimate of repeated-fit power.

Boosting and MACE predict the same histogram probability model. Checkpoints use
validation categorical NLL (distance density NLL differs by a target-only bin-width
term). Metrics inherit the censored histogram convention from liquid_descriptors:
NLL, capped-distance RMSE, Brier at 20/32/48 Å. The reference is the fitted no-input
histogram. A paired bootstrap resamples independent sources, 2,000 draws; ordinary
95% intervals exclude training-seed uncertainty and are exploratory across arms.
Report expected RMSE and NLL integrated over the known conditional target law as
well as realized-label scores; exact expectations remove label-draw noise, not
source uncertainty. Positive gain favors the model.

## Rich-feature prediction

The geometry-only MACE128 and two vector-context blocks feed one 128-dimensional
context state. A 128→256→3536 decoder predicts every rich context descriptor.
No target descriptor enters MACE or the context predictor. Identical patch MACE
weights apply to all 25 patches. Random initialization, no crystal-supervised
parent weights. Targets and scales come from training rows of the corresponding
domain only. Training SD below 1e-4 marks a constant/unresolved training target:
that output is fixed to its training mean, excluded from the selector, and retained
in evaluation and column records. No feature is silently removed.

The task objective is fixed-unit-variance Gaussian NLL after standardization,
averaged equally over geometry, bond_order, cna and tda; within a family, average
over its varying features. VCReg is .05 variance/.01 covariance with 512-update
warmup and is excluded from checkpoint selection. This user-requested feature
learning experiment does not automatically feed into the distance models.

Report source-weighted per-feature RMSE in native units and R2=1-MSE/held-out
target variance (undefined for zero variance), family mean standardized MSE,
and improvement over predicting the training feature mean. The mean baseline's
held-out standardized MSE need not be 1. Publish all 3536 predictions and 128-D
states with exact row IDs. The raw-full run and matched raw/relaxed pair are
separate cohorts; compare domains only within the matched pair.

## Paired relaxed inputs and both labels

Freeze the intersection of original rows and successful archived full-cell FIRE
relaxations. Verify source manifest, frame, timestep, atom identities, potential,
fixed box, force tolerance <=0.01 eV/Å, archived conversion and coordinate hashes.
Archived global positions are float16; retained conversion quantization is
reported. Extract offsets in float32. These are not the original unquantized
converged coordinates. Full-cell relaxation can convey information from outside
the model's observation, so it is not a strictly local denoising operation.

Retain hot patch centers and nearest-80 atom identities. Atoms outside the hot
8-Å consumed support are excluded from cold inputs as well; the encoder/descriptor
8-Å cutoff still applies to cold displacements. Shell membership is frozen to
the original 1/12/12 patch assignment, while spatial moments use cold offsets.
Actual cold spatial extent may differ slightly from the nominal 32-Å hot envelope.

Old labels are unchanged original MD established-crystal distances. New labels
reclassify the archived relaxed full cell with PTM RMSD cutoff 0.1, FCC/HCP/BCC
types 1/2/3, connectivity 3.6 Å and components of >=64 atoms. They are instantaneous
relaxed clusters: sparse quenches do NOT satisfy the original three-MD-frame
temporal establishment criterion. New distances use relaxed coordinates and
periodic nearest qualifying-crystal atom distances.

All four input×label arms share one exact cohort: original eligibility, available
relaxation, a relaxed qualifying crystal elsewhere, a relaxed-liquid query, and
no relaxed qualifying-crystal atom in any originally consumed neighborhood.
Record absent, inside and visible cases in a challenge/eligibility artifact; do
not drop different rows for different models. Source roles never change. Original
weights are conditioned and renormalized per role over the common cohort.

Raw/relaxed input effects are paired within a fixed label definition. Old/new
label score differences are different tasks, not model improvements. Report label
and membership changes separately. Fixed depth-4 all-feature boosting supplies
the paired input-domain contrast; model selection is validation likelihood only.

The full-coverage extension requires relaxation availability for every original
eligible row before freezing its paired cohort; it never silently substitutes
the earlier available-only intersection. New quenches use the same generating
potential, fixed box, FIRE and force tolerance, with larger execution limits.
