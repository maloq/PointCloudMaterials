# Feature dominance and generalization diagnostics

Analyze the selected completed CIV-MACE128 distance+direction+VCReg checkpoint,
without changing it or its scientific selector. Verify frozen prediction and
checkpoint checksums. No W&B diagnostic runs. No explicit time, temperature,
material/source identity or labels are predictor inputs. Label-side cue correlations
and visibility conditioning are analyses, not newly permitted inputs.

Complete saved-feature analysis uses original fixed/uniform populations. Within
each role, each population has half mass and each source within it equal mass;
then condition/renormalize for visible/invisible subsets. Train/selection/test are
reported separately; calibration is unused. Never reselect the encoder.

Fit weighted scalar PCA on all training rows only, separately for exported local
128-vectors and mixture-pooled context 128-vectors. Apply that same basis to every
role. Report variance captured by first 1/2/4/8/16 training PCs, native-channel
standard deviations and largest native-channel variance share. Correlate training
PC coordinates with capped distance, interface visibility/existence, interior
phase, weighted atom counts and bond-order powers. Physical fields follow
encoder_context.geometry.physical_targets: counts at indices 24/25; powers of
orders 2/4/6 at radii 5/8 at 26..31. Report weighted Pearson correlations by role.
The source-between variance fraction is weighted variance of source means divided
by total variance, not a Pearson correlation; its row explicitly says so.
Source variation can be physically legitimate and is not proof of a shortcut.

Local diagnostic logistic probes predict d<=20 and d<=32 A. They use only the
named saved embedding/design (or the 32 fixed physical descriptors as a control).
Designs: all 128 native coordinates, all 128 training PCs, embedding norm, first k training PCs, and the
remaining 128-k PCs, k=1,2,4,8,16. Standardize each design using training weighted
mean/std (floor 1e-6). Minimize population-weighted Bernoulli log loss plus
0.01/2 times squared coefficient norm; intercept unpenalized. L-BFGS converges to
the declared objective; no hyperparameter/feature selection on held-out labels.
Also report fixed training-prevalence probability and original trained predictor.
Use all-PC probes as the reference for retaining/removing PC subsets, preserving
the scaling and penalty of the retained coordinates. Native-coordinate and
standardized-PC ridge fits have different regularization geometry; their scores
are not a pure information comparison.
These refitted probes measure accessible information; they are not ablations of
the original nonlinear head. Log loss converts probabilities to float64 before
clipping to [1e-9,1-1e-9], including stored float32 values rounded to exactly 1.
Report Brier, log loss, mean probability and prevalence. These are localization
probabilities, not 3/6-ps future crystallization AP.

Frozen-head interventions use 16 uniformly selected rows per source and population
from train/selection/test, seed 20260928, without labels or visibility stratification.
Every original source contributes; sample counts are explicit. Export actual
normalized scalar/vector inputs for all 25 patches using the frozen MACE checkpoint.
Use the unmodified model forward with its encode result replaced by cached fields.
Keep relative patch geometry fixed. Verify original probabilities against saved
inference: record mean/max difference; max >0.02 fails (rebatching BF16 may drift).

Fit a separate scalar PCA on these training patch inputs, equal patch mass within
each query. Interventions: replace all scalars with training mean; keep first
1/2/4/8 PCs; remove first 1/2/4/8 PCs; clip scalar deviations to three training
standard deviations; zero vector fields. Retain scalar training mean when projecting.
Vectors remain unchanged except for the explicit vector intervention. The original
head is never fitted after intervention. Such changes can leave the training
feature distribution; damage measures reliance, not proof that a feature is spurious.

Report the original predictive objective (including distance/direction and proximity
log scores, excluding VCReg), distance marginal NLL and the same proximity scores.
Paired 95% percentile intervals use 1000 whole-test-source bootstrap draws, seed
20260928. Sum weighted loss differences and masses within each source, resample 30
sources with replacement and divide summed difference by summed mass. Invisible
subsets retain conditional masses. This captures source uncertainty only, neither
training-seed nor within-source sampling uncertainty. No multiplicity-corrected
significance or treatment selection is claimed.

Dominance becomes evidence of harmful reliance only with a generalization penalty
or systematic proxy failure. Low rank alone, importance under ablation, large
weights and legitimate predictive features are insufficient to establish overfit.
