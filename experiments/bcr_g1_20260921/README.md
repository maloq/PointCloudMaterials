# Independent-root BCR conditioning pilot

Question: does learning the clean invariant code improve held-out geometric
reconstruction over independently trained unconditional and frozen-random-code
controls, and does the frozen code retain useful liquid structural information?

Architecture is unchanged from BCR-v1: native two-interaction MACE, 128-dimensional
signed invariant export, separate two-block equivariant decoder. FP32 eager only.
No temporal inputs, velocities, TDA/physical auxiliary loss, SIGReg/VICReg in BCR,
crystallization labels, EMA teacher, or architectural hyperparameter sweep.

## Cohort and limits

Existing end-of-melt validation snapshots: 300 ps independently seeded full-box
melts at **1325 K**, Lee2003 Al MEAM. This temperature is checked in the actual
melt input, not inferred from the later undercooling-directory label. Archived
text coordinates are printed at nine significant digits, before float16 history
conversion. Native float64 box parsing and a conservative text/storage precision
bound preserve all five fixed noise levels. No new simulation or quench is needed.

Select 18 roots from the historical training population associated with later
520 K histories, using a fixed random source permutation independent of outcomes.
12 roots fit the encoder; six roots are held out for this pilot. These are
historically studied roots, not a fresh final test. All historical calibration,
selection and test lineages are excluded. Each source contributes one snapshot
and 256 centers chosen with 8 A minimum separation: **3,072 training /1,536
development patches**. Neighborhoods can overlap; independent uncertainty units
are roots, never patches or corruptions. One-frame-per-root sampling is explicit.
This is a narrow high-temperature liquid mechanism assay; it does not establish
undercooled or crystallization-transfer performance.

## Three matched fits

BCR, learned-constant unconditional decoder, and frozen-random-code decoder.
Same initialization tensors where applicable, root/anchor stream, Gaussian noise
stream and levels [0.01,0.02,0.04,0.08,0.12]*d0. Batch256, one seed, 10,000 updates,
AdamW3e-4->3e-6 with 5% warmup/cosine, global clip1 and weight decay1e-5. Effective
batch weighting is preserved through microbatches. Relative parameter-update norms
for encoder/decoder are logged every50 updates. High rank is not an objective.

## Predeclared readouts

Paired checkpoint comparisons at 0/1,000/3,000/10,000 updates. Fixed development
bank:384 anchors balanced over six roots, two noise draws at each level, four
strict matched derangements. Match condition/count/density/q6 with bins calibrated
on training data. Report partial-match coverage, relaxed matching and unrestricted
shuffle separately. Report absolute errors, gains over both trained controls and
swaps, globally and within q6<0.35, with paired root-bootstrap intervals. One seed;
bootstrap intervals do not include optimization-seed uncertainty.

Pilot G1 rule: at two or more levels >=0.04, within-liquid mean improvement over
unconditional >=5%, positive 95% paired intervals over unconditional, frozen random
and strict matched swaps, strict-match coverage >=50% with >=4 roots. These are
engineering thresholds, not a physical theorem. Unresolved evidence is inconclusive.
Passing G1 alone triggers no claim of general representation quality.

Frozen ridge/128-wide nonlinear probes at each checkpoint measure radial/density,
angular and rich moment targets. Ten of the encoder-training roots fit probe
transforms/readouts; two other training roots choose ridge regularization. None
of the six development roots tunes these readouts. Report within-liquid results
and per-root errors, including comparison with the initial/random representation.

Matched VICReg and a reinitialized decoder on frozen trained BCR are follow-up
experiments after this screen, not part of this initial three-arm launch. G3 trained
perturbation behavior and crystallization transfer remain subsequent evaluations.
No architecture/corruption choice will be selected using crystallization outcomes.

Recipe: `configs/bcr/pilot_20260921/study.json`; output:
`output/bcr/g1-independent-20260921/`. Maintained workflow:
`python -m src.research.bcr_pilot.queue prepare|preflight|submit --config CONFIG`.

Prelaunch cohort audit: all384 evaluation anchors satisfy the fixed q6<0.35
liquid stratum; strict cross-root code-donor coverage is86.20% for each of the
four independently randomized assignment maps. Unmatched anchors remain explicit.
The two-patch weak-noise optimization diagnostic reduced NMSE1.010->0.895 after
1,000 updates, below the arbitrary20% smoke threshold. This observation is retained;
the fixed-architecture optimization check additionally uses the already declared
0.08/0.12 levels. This diagnostic choice does not alter the scientific training grid,
noise mixture, held-out banks or G1 criterion.
