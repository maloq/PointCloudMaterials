# Frozen Ni phase-model nucleus prediction

The source contract is the completed phase-space campaign's ninety Ni ancestors.
Existing sixty-source birth rows are reused exactly; thirty-source additions use
the same producer and seed. Inputs are exact native 0.10 ps saved observations,
1, 3 or 6 ps before the start of a connected 32/64-atom nucleus sustained for
1.5 ps. Only liquid-origin root events qualify. Four sampled founder atoms and
sixteen controls per event follow the original case-control definition. Controls
exclude the founder sets of scored roots; they are not asserted to remain liquid
forever. There is no new neighborhood filter. Repeated task/lead/input identities
are deduplicated; conflicting labels fail. Leads are separate classification
tasks, not cumulative horizon labels. Different thresholds can share roots.

The original nucleus producer scaled both Q and V by the fixed Ni length factor.
This consumer reverses that factor and centers both Q and V before dividing by
the phase model's 5 Angstrom and 10 Angstrom/ps units. Input support is the same
nearest eighty atoms. No history, future physical observations, temperature,
absolute age, box tensor, source identity, element or relaxation enters probes.

All declared phase-model training ancestors remain training (28). All other
inherited roles map to test (62); both phase-parent roles and older raw-catalogue
roles are stored. Four raw-catalogue training sources had already become held
out upstream of phase training; they remain test here. No new inner selection
is created. The four phase treatments have three final
8000-update checkpoints each. No representation fine-tuning is performed.

For each atom, form the upper-triangular Gram matrix of latent vector channels.
Pool the central atom, the noncentral mean, and three weighted radial-shell
means with centers 2.5/3.5/4.5 Angstrom, width 0.45 Angstrom, and quadratic taper
to 5.5 Angstrom. Membership and shell weights use the initial Q throughout.
`phi` contains these initial invariants (180 features for eight channels).
`path` concatenates phi and its neural changes at 20/50/100 fs (720 features).
For the two-channel direct model the same construction has 15/60 features.
`measurements_path` appends path to 244 original BO40/TDA113/density74/velocity17
measurements. Controls use the 227 position measurements, all 244 measurements,
or a training-prevalence constant. Structural BO/TDA keep their historical fixed
Ni length normalization; density and velocity measurements use native units.
These fixed invariant summaries can discard information from the full field.

Every learned readout is CatBoost with exactly 600 trees, depth 4, learning rate
0.04, L2 10 and 64 borders. Train-only quantization, binary log loss, unmodified
class prevalence, no inner selection, early stopping, test tuning, calibration
or AP objective. Each event has total weight one within its task and lead;
training weights are divided by their mean. The readout seed is shared across
all feature treatments. Model seeds repeat representation training, not MD.
All readouts stay local; metrics update existing W&B training IDs only.

`scores.csv` exports event-weighted average precision (AP), AUROC, binary negative
log likelihood (NLL), Brier score and prevalence. Probabilities are clipped to
[1e-7,1-1e-7] for shared score calculations. Each 95% interval resamples whole
test ancestors 1000 times, keeping all events/rows together; resamples missing
a class are excluded and their valid count is recorded. This case-control AP
has approximately 20% prevalence, not the natural nucleation rate. Multiple
founder rows are not independent births, and multiple births within an ancestor
are not independent MD realizations.

`calibration.csv` gives fixed probability bins [0,.1),...,[.9,1], event-weighted
predicted and observed fractions, weights and row counts; it fits no transform.
`summary.csv` averages individual scores across checkpoint seeds and reports
population seed standard deviation. `paired.csv` subtracts right from left,
averages paired seed differences, and bootstraps the same sources/draws for both
models. Positive AP/AUROC and negative NLL/Brier differences favor the left.
Intervals condition on the fitted readouts and observed checkpoint seeds. No
probability ensemble, fitted seed weights or model promotion is performed.

Scientific interpretation: diagnostic readout information in representations
trained over 0.1 ps, not direct multi-ps neural MD and not a demonstrated
first-crystalline-atom detector. Historic results on some sources were already
viewed; this is an exploratory matched comparison with frozen new probe settings.
