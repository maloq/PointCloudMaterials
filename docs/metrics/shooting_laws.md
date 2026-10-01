# Physical future laws from shooting, v1

This family is a new numerical protocol. Historical shooting scores, input
contracts, first-passage labels and selected models are preserved unchanged.

## Population and uncertainty

The primary release has 40 position parents, two per recorded source trajectory,
and 12 independently seeded Langevin futures per parent. The original outer
train/validation roles are preserved. Source seed 35863 at each temperature is
selection within old train: 11 fitting, three selection and six historical test
sources. These reused sources do not constitute fresh prospective confirmation.
Source IDs are the recorded sampling/bootstrap units, not a new certification
of independent melt ancestry. The Al64 release identifies frozen encoder
ancestry only; this assay does not replace or resplit all64/legacy16 evaluation.

At each parent, up to 64 atoms are drawn without replacement in each disjoint
current stratum: no PTM crystal within 8 A; noncrystalline center with PTM crystal
within 8 A; crystalline center. Every consumed nearest-80 atom is masked at 8 A.
Weights are N_stratum/(N_cell * n_sampled_stratum); sums are one per parent.
Sources then have equal total mass. No future outcome chooses centers or strata.
Historical parent times themselves were selected relative to nucleation.

Reported source scores renormalize these weights over eligible rows within each
source/population. Aggregate scores average source scores and the three declared
fit seeds. Paired percentile intervals draw whole sources 2,000 times after
averaging seeds, comparing the same sources with the prior. They condition on
these fitted seeds; atoms and shots are not independent bootstrap replicates.

## Joint structural path

The eight fixed coordinates are l4_qbar, l6_qbar, l6_coherence_mean,
l6_neighbor_q_std, n80_h1_loglife_sum, n80_h2_loglife_sum, n80_h2_image12 and
n80_h2_image13 from the retained `patch_descriptors` producer. Concatenate the
same branch's 3/6/12 ps observations, giving 24 joint coordinates. Each coordinate
is standardized using source/population-weighted fitting observations only;
standard deviations have a 1e-5 numerical floor. Every arm uses identical targets.

Each readout predicts a four-component diagonal Gaussian mixture over this
24-dimensional path. Component means and scales vary with the input; mixing
induces cross-time dependence. Scale floor is 0.10 standardized units. This is
an approximate descriptor-path law, not an injective raw-coordinate law. No
physical-reconstruction encoder treatment is trained. Frozen readouts fit and
select by source-weighted joint path NLL; maximum 200 epochs, patience 25,
fixed source-role selection, AdamW .001 and weight decay .001. The prior has the
same mixture family with no observation input. All heads use 64 hidden units.

- **path_nll:** mean negative log mixture density over all sibling paths for
  each observation. Units refer to the common standardized coordinates; the
  omitted physical-unit Jacobian is constant across arms.
- **mean_squared_error:** mean squared error of the predictive mixture mean
  against each realized standardized branch path, averaging 24 coordinates.
- **energy_score:** E||X-y|| - 0.5 E||X-X'||, averaged over all observed siblings.
  Two independent sets of 64 predictive draws estimate the expectations. The
  second term pairs independent draws, avoiding the finite-ensemble diagonal
  bias. Lower values are better. Monte Carlo error is not bootstrap uncertainty.

## Local event-time distribution

Full-cell PTM uses cutoff .1, FCC/HCP/BCC; full periodic components use 3.6 A
connectivity and the repository's strong/weak-overlap ancestry calculation.
The established-cluster threshold is 64 atoms for six consecutive 0.3 ps frames
(1.5 ps from first to last). Local onset requires the tracked center to be
crystalline for the same six observations. Analyze exact frames 0:0.3:15 ps;
only onsets in (0,12] are targets, with sufficient subsequent confirmation.
Parent-crystalline centers are excluded from event fits/scores, retained for
structural paths. All selected branches have full fixed follow-up; partial
branches fail admission rather than becoming negatives.

There are 13 mutually exclusive outcomes: no sustained onset by 12 ps, or one
of four causes in (0,3], (3,6], (6,12]. Causes are parent-present crystal arrival,
local isolated establishment, external/interface-associated new formation,
and unresolved ancestry/formation in progress. A parent-present crystal is a
component present at frame zero and subsequently confirmed persistent; no
unobserved pre-parent establishment history is asserted. Unresolved events are
explicit events, not liquid controls. These are operational finite-horizon
events, not a thermodynamic committor or validated critical-nucleus labels.

- **event_nll:** mean negative log predicted probability of the branch's category.
- **brier_Hps:** mean squared cumulative-onset probability error against branch
  onset indicators through H. Raw probabilities; no fitted calibration map.
- **calibration:** weighted predicted and empirical branch frequencies in ten
  fixed probability bins, separately by horizon and fit seed. Counts are centers.
- **AP/AUROC:** diagnostic source-weighted branch scores only. Undefined
  single-class populations are omitted from the diagnostic table. They never
  select parameters, checkpoints or models.

## Reliability and atlas

A fixed 256-feature Gaussian RFF map uses training-path median distance bandwidth
and a recorded seed. For total shot budgets 2/4/8/12, partition a random subset
into two DISJOINT halves of 1/2/4/6 shots. Report squared mean-feature discrepancy,
averaged with observation weights; 16 repeated partitions are dependent
diagnostics, not 16 new experiments. This measures finite-shot reproducibility,
not a theoretical predictability ceiling.

Future-neighbor retrieval uses test candidates at the same temperature/current
stratum from a different source. Temperature organizes evaluation only. The
static reference is fitting-standardized descriptors. Predicted RFF means use
64 model samples, then an unweighted mean over the three declared fit seeds for
descriptive retrieval only. The empirical oracle chooses neighbors using shots
0--5; every representation is scored using shots 6--11. Five nearest candidates
are used; ties use stable row order. The score is squared held-out-half kernel
mean distance and has finite-shot sampling noise. No future-derived caliper,
geometry fine-tuning or AP objective is used. The random-feature law captures
only the declared physical path coordinates.

## Separate protocol diagnostics

Four CSLD futures at each original parent retain the same atom centers/weights;
the thermostat contrast is their mean physical path minus the 12-shot Langevin
mean. It is descriptive, not evidence of protocol equivalence. Nested Al uses
192 uniform centers per parent and preserves the original two-momentum/two-noise
dependence. Its fixed-24 ps continuations are not extra independent futures.
Historical first-passage outcomes and censoring are not rewritten.

Ta uses 192 uniform tracked centers per each of six archived configurations,
four shots each. Coordinates are multiplied by the frozen Al/Ta material length
ratio before the common descriptors/encoders. No species, scale, temperature or
time feature is passed. All Ta parents belong to one conservatively grouped
preparation. These are structural conditional-transfer diagnostics, with no
new event classification or independent-material-generalization claim. Explicit
source-path overlaps with encoder training and original readout roles accompany
every parent score; absence of a path match does not certify unknown ancestry.
