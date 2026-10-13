# Ni density-rate conditioning and teacher expansion

This family covers the October13 physical-only experiment. Inputs are present
BO40/TDA113/density74/velocity17, optionally conditioned on74 instantaneous
density rates D_x rho[V] at the current fixed box. Rates have inverse-ps units;
they use actual saved velocities, and are recomputed for each perturbed input.
Both arms retain the identical244 future targets. The two-token transformer
receives an initial conditioning token and the evolving current token; the
control masks the rate input to zero with identical parameter capacity.

## Data and roles

Only train and test exist. Original16 training ancestors are retained; the
four former selection ancestors join the40 test ancestors. Of30 additional
independent sources,12 join train and18 join test. Thus28 training and62 test
ancestors are available. Banks are80 states/original16,320 states/original16,
and320 states/all28. The two320 banks have exactly matched temperature counts;
temperature is audit metadata only. States use outcome-blind centers/times in
available post-cooling histories; native crystallinity stopping still affects
which history is available. Reused and new anchors retain source/frame/center
identity. Test is five anchors per ancestor (310 total), with pooled and old/new
cohort scores; these cohort panels are not extra dataset splits.

Historical training-only244 means/scales are common to all arms. Density-rate
normalization, OU rates and increment/response training normalizers use only
the corresponding fitting bank. No test or former selection rows enter any fit.
Models use the predeclared final1600-update checkpoint. Readouts use fixed
ridge penalty0.01 and state-kernel length2, without hyperparameter selection.

## Physical and neural time

Unchanged FastNi EAM/FS full-cell fixed-box BAOAB uses original Q/V, each
source's thermostat temperature,1-fs integration and two independent shots.
Integration, noise and tangents are float64; force arithmetic is float32.
Local observations follow the same central atom and initial80 identities.
Measurements and JVPs are float32. These are separate from saved native NPT.
Neural steps are5 or1 fs; both emit exactly5,10,...100 fs, without interpolation.
Density rates condition the initial state; they are not asserted to be current
rates at future times. There are no physical-time or temperature model inputs.

## Scores and losses

The shared measurement-followup calculation is reused: scalar Gaussian-mixture
NLL (bandwidth0.25 in normalized coordinates), unbiased ensemble energy of
whole paths, energy of increments scaled by train RMS, and energy of lag1/5
increment products. Loss weights are1/1/0.5/0.25. Family weights are one quarter
each for BO/TDA/density/velocity, masking train-constant coordinates. This NLL
is marginal composite likelihood, not joint-path likelihood. Both physical
shots are individually scored; the response target is their mean.

Response loss is normalized MSE on91 smooth channels, equally weighting
density74, shell9, bond-rate4 and flow4 groups, two query directions and times.
Training denominators are fitting-bank target powers with1e-5 floor. Query
directions are initial Q or V; common neural randomness is used for plus/minus
secants. Epsilon0.25 is primary; epsilon0.125 is an evaluation control. Added
rate conditioning is recomputed at both endpoints of each secant.

Evaluation uses32 neural particles and three independent outer keys. Forecast
scores are reported through50/100fs, including endpoint MSE, raw-particle90%
coverage and temporal powers. OU, mean, persistence and shuffled-particle
controls are retained. Source means receive equal weight, with1000 source
bootstrap draws for95% intervals. Sensitivity exports per-group/direction/time
error-to-target-power ratios with paired numerator/denominator bootstrap;
zero response equals1 except where the declared power floor is active. Do not
pool these ratios into a different metric without explicitly defining it.

Readouts test known present response R0 plus current measurements, optionally
rates, using linear transport or a state-conditioned residual kernel linear in
R0. Their information contract differs from ordinary trajectory inference.
The two-shot mean-noise estimate is |R1-R2|^2/4, aggregated per group/source;
two shots do not establish an exact per-state irreducible noise floor.

No birth/AP objective or evaluation is included. Historical exports retain
their definitions and original split roles. Scientific fits stay W&B online;
readouts, integration checks and teacher preparation stay local.
