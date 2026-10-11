# Unrelaxed response geometry: predictive and distance objectives

Fixed Al64 identity e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d.
720 training patches from90 train ancestors (original90 plus7 per source), original
15 selection and30 test queries. No calibration sources. Fixed centers/frame cycle,
no outcomes for sampling. All input is unrelaxed center-relative nearest80 geometry,
one constant atom channel, center indicator, support8A, edges5A, native128 channels
and128 embedding. No temperature, age/time, history, motion or halo tensor inputs.
MACE-MPA-0 teacher with450K BAOAB,1fs step,100fs friction is training-only; parent
MEAM dynamics differ. Target prefixes20fs and20+100fs. Moving24A versus36A gate on
four train inputs; new target AD checked by common-noise finite differences.

One trajectory returns768 coordinates: original256 Fourier features of12-observable
changes; rich256 changes; rich256 joint future endpoint and change features. Rich
observations contain12 radial densities (1.5 to5.25A,Gaussian width.35A) and angular
powers l1..6 in two radial windows (2.7/4.5A,width.65A), all tapered at5.5A. Each rich
coordinate uses mean/SD from720 training initial observations, SD floor.01. This
normalization is not a reconstruction target. Independent fixed Fourier maps use
seed202610031,128 features per prefix, Gaussian frequencies divided by sqrt(input
width), uniform phase. Same paths/tangents supply all three target ablations.

Training has8 value/response branches per parent with one independent execution
audit. Selection has32 value-only branches; test has32 values and8 derivatives on
disjoint streams. Original query seeds and inputs are retained for execution
comparison. New parent streams use181000000 + index*10000. Legacy feature scaling
and response scale come from the original90 train bank. Rich feature means/SD
(SD floor1e-4) and response RMS (floor.001) use720 training parents only. Derivative
normalization uses mean8-shot derivatives divided by feature SD. Selection's unused
empty response tensor in the inherited collector is explicitly translated from
(0,256,2) to(0,768,2) at sealing; no responses are invented.

Eight arms ×three seeds: legacy90/720 × value/response, rich-delta720 response,
rich-joint720 value/response, and rich-joint720 response+distance(weight.1). Each
receives400 optimizer updates of256 samples;90-parent arms sample with replacement,
720-parent arms without. Microbatch8 explicitly bounds higher-derivative memory.
No crystal/descriptor cluster labels enter fits. Selection every5 updates plus
initialization uses the target-specific Gaussian feature NLL only. NLL across
*different target definitions* is not a model-selection comparison.

value_nll = .5*mean((prediction-mean_shot_target)^2)+.5 log(2 pi), with fixed unit
variance in normalized feature coordinates. response_half_mse = .5 mean squared
error of directional derivatives, divided by training response RMS squared.
For distance supervision, use raw fixed Fourier feature vectors f with B=8 shots.
D_ij = ||mean(f_i)-mean(f_j)||² - trace(sample_cov(f_i))/B - trace(sample_cov(f_j))/B.
This is unbiased for squared mean-feature distance for independent banks. Negative
estimates are retained; repeated parent IDs are excluded. Divide D by its mean over
all distinct training-parent pairs (must be positive). Predicted distance is
mean_coordinate((z_i-z_j)^2) divided by twice mean-coordinate variance of initial
train embeddings, frozen per fit. distance_half_mse is half squared distance error
over distinct pairs within each uniformly sampled microbatch. No phase supervision.

The combined training loss adds response_half_mse with weight0/1 and optional
distance_half_mse with weight.1. gradient_norm is pre-clipping total gradient norm;
clipping10. update counts optimizer steps; selection_nll is blank when not evaluated.
Final predictive table: selected_update, selection_nll, test_nll, corrected test
value MSE subtracting the32-shot mean noise variance, corrected response MSE
subtracting8-shot mean derivative noise variance. Corrected MSE may be negative.
Test response MSE uses feature-normalized units, not additional response-RMS units.
Each of30 test parents has equal weight. No claim of full atomic-law likelihood.

Gate table records value_rms difference across24/36A and response_relative norm
error, passed iff <=1e-4 and <=.05 respectively, for each target and gate parent.
Collection costs are actual total_batch_seconds per parent, not per-shot guesses.
Elapsed training time is a W&B summary only. Scientific fits use stable online run
IDs; frozen clustering stays local and updates the existing training summary.
Clustering definitions remain in response_geometry_clusters.md; primary K7
many-to-one agreement is accompanied by occupancy and clustering-seed stability.
