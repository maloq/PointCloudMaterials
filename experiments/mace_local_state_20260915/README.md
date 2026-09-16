# Frozen features for informative, smooth local-group states

Question: can short-time canonical coordinates or a learned group-physics
distance organize the existing frozen MACE features into informative, smooth
local states with honest uncertainty?

The object is a local group within a smoothly pooled 5–7 A region, supported by
complete surrounding message-passing neighborhoods. A separate two-frame variant
uses only current and preceding observations, spanning 0.75 ps. Forecasting and
whole-process progress are outside this experiment's objectives and selection.

Compare nine affine representations: inner/dual/projector PCA16 controls;
inner/dual temporal canonical coordinates in 8 and 16 dimensions; and inner/dual
10D learned physical coordinates. The latter use a ridge map to means and spreads
of local bond order and density. Squared distance is an explicit learned
positive-semidefinite metric. MACE and the existing projector remain unchanged.

The 18/6/6 simulation-source split is retained. Context-centered fitting and
within-context neighbor comparisons prevent simple temperature/phase separation
from satisfying the objective. Evaluate independent group observables and the
existing instantaneous/relaxed TDA targets, plus temporal and boundary smoothness.
This is the existing encoder-development cohort, not a fresh final test set.

Density discovery allows rejected points and mixed memberships. It runs first
on training-source local states, then separately on the spatial training slab of
the mostly disordered Al 166 ps frame. The latter uses a separated test slab and
an unchanged catalog across the remaining static frames. The number of clusters
is unconstrained; lack of supported discrete states is a valid result.

Definitions, exact scaling, selection and uncertainty caveats:
[metric protocol](../../docs/metrics/mace_local_state.md).
Rationale: [literature review](../mace_context_clusters_20260915/LITERATURE_REVIEW.md).
Recipe: [mace_local_state.json](../../configs/analysis/mace_local_state.json).

```bash
conda run -n pointnet python -m src.research.mace_local_state.run --config configs/analysis/mace_local_state.json --stage all
```

Stages `prepare`, `fit`, `evaluate`, and `static` can be run individually.
Preparation verifies the frozen checkpoint hash and replayed anchor features,
retains per-context shards, and permits only exact-input preparation resumption.

Output: `output/mace_local_state/frozen-local-group-20260915/` (WORK analysis).
Tables contain definitions and implementation fingerprints; density models,
affine maps, memberships and all source-separated scores are retained.
The run is complete. [Results and interpretation](RESULTS.md): the physical map
improves group information and distance geometry while losing instantaneous
topology; temporal maps show no consistent advantage, and the tested density
discovery finds no supported subdivisions of the static disordered region.
