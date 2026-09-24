# Which combinations preserve liquid detail and interfaces through convergence?

The broad native screen shows a tradeoff: GeoFrame resolves Al fault/interface
contexts well, while several MACE exports better decode continuous liquid order.
Stronger variance regularization does not consistently preserve useful neighbors.
The two-seed distance/future study found only ~0.27% better withheld neighbor
error from distance weight0.1 and no replicated benefit from future-residual
supervision. The residual task was barely learned. We therefore test exported
geometry and optimization directly before adding more future objectives.

## Predeclared training design

28 fresh fits, 14 parameter combinations, two seeds per combination:

| Family | Factorial axes | Held fixed | Budget |
| --- | --- | --- | --- |
| GeoFrame V2 | projector MLP/identity × covariance coefficient1/5 × FactorVAE off/on(gamma0.1) | VICReg sim25/std25; learning rate0.001; five-material cache; batch16384; original160-epoch schedule | 35 full passes; seeds123/456 |
| GeoFrame VISReg controls | projector MLP/identity; FactorVAE on; lambda0.4,4096 projections, scale1/shape0.5/center0.1 | Same35-pass budget, architecture/data/optimizer as VICReg controls | seeds123/456 |
| Native geometry MACE | encoder LR1e-5/1e-4 × physical-distance weight0/1 | relaxed8Å input; geometry targets; current-order weight0.25; no future loss; head LR3e-4; batch256/micro64 | 4096 updates; seeds20260923/20260924 |

All GeoFrame arms keep original architecture, augmentations and160-epoch warmup/
cosine clock. The MLP applies the loss to a projected code; identity applies it
directly to the invariant encoder export. Evaluate both exports, without counting
identity duplicates twice. The FactorVAE on/off ablation is warranted because
its historical weighted loss and discriminator signal were small; gamma0.1 is
not assumed to be an effective mechanism merely because it appears in the name.
Higher covariance weight tests retention of dimensions, with physical metrics
guarding against meaningless rank inflation.

The supplementary review of29 completed native snapshots found the historical
VISReg epoch159 raw encoder had the lowest liquid-neighbor error and highest
nonbulk fault AP. Four new VISReg fits test whether that benefit appears at a
matched35-pass budget. This addition is based on reused development evidence,
not a confirmatory preregistration.

Native MACE starts from scratch, retains its pooled+learned128 export and bounded
physical heads. v4 is a separate protocol; the historical v3 four-arm future
factorial remains strict. Distance weight1 is ten times the earlier weak0.1;
encoder LR1e-4 is ten times its earlier rate. Every new MACE arm gets the same
current-order labels; none gets onset or future-supervision labels. Construction
of an unused future residual head is retained to match architecture/RNG state.

Within each seed, initial encoder weights are verified equal across treatments.
MACE uses the identical balanced pair stream. GeoFrame uses explicit per-pass
sampler seeds and per-batch augmentation/grouping seeds, so projector and
discriminator initialization cannot determine those streams. This is a new
matched protocol rather than a bitwise replay of the historical training.

GeoFrame has1,009,002 fitting patches,61 batches/pass (999,424 presentations),
2135 encoder updates at35passes. FactorVAE adds discriminator steps. MACE has1600
fitting observations drawn with replacement;4096updates gives655.36 equivalent
presentations per row, not shuffled epochs. No scientific arm stops at fewer
than12GeoFrame passes; disposable correctness fits are explicitly excluded.

## Evaluation and selection

Frozen independent classical reference centers, labels and spatial splits;
GeoFrame is transductive on these static datasets. Generating potentials for the
static snapshots remain unknown. MACE training has45 independent Al2NN-MEAM
roots,25fit/5tune/15reused development. No new simulation or dataset split is made.
Ta/Zr structured-liquid labels are candidates without validated future fate.

Retain every GeoFrame epoch and assess initial,12,24,35 passes; assess native
MACE initial,1024,2048,4096 updates. Fixed final checkpoints are primary. Assess
continuous order/topology error beyond density, raw-space liquid-neighbor
retrieval, nonbulk interface/fault AP with coverage, boundary-sensitive spatial
AUROC and perturbation response, plus conditional crystallization forecasts.
Supplement sparse-column R² with fit-scaled NMSE; do not overwrite old scores.

Choose a **set of nondominated combinations**, separately by family and exported
representation. A structural promotion needs >=1% lower liquid-neighbor error in
both seeds versus its matched baseline, <=2% worse order and topology NMSE, and
no >0.02 absolute loss of supported nonbulk boundary/fault AP or spatial AUROC.
For GeoFrame, require the order/topology guard in Al,Ta,Zr; report every material.
These practical thresholds are screening rules, not significance tests.
Inspect failures/collapse/undefined coverage before applying rules.

A predictive promotion additionally requires negative paired-root Brier change
against current physics and no adverse9ps residual-error change, in both seeds;
report intervals, AP and selected-step0 separately. Better AP alone does not pass.
Do not declare a universal best encoder from a UMAP, rank, one seed, or the
strongest observed AP. Final confirmation requires independent, previously unused
sources. Cross-model metric associations are exploratory and not causal evidence.

[Configuration](../../configs/encoder_parameter_search/campaign.json) ·
[Execution](../../docs/encoder_parameter_search.md) ·
[Metrics](../../docs/metrics/encoder_parameter_search.md) ·
[Live comparison](../../output/encoder_research/parameter-search-20260923/index.html)
