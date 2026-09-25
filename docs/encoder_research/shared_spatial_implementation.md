# Proposed implementation: shared atom computation and spatial context

This is the implementation design for item 4 of the runtime audit, not an
implemented model or a claim of improved predictive information. The current
runtime refactor implements items 1 and 3 only.

## Core change

Compute atom features once for each distinct observed frame/overlapping spatial
union, then obtain many focal states from those shared features:

```text
frame/union positions, species, periodic edges
                    ↓
       shared equivariant atom backbone
                    ↓
       smooth focal and regional readouts
                    ↓
       local–context tensor interactions
                    ↓
           one exported state per center
```

The current backbone adds a focal-center indicator before message passing and
applies a focal-patch taper to edges and atom features at every layer. Therefore
two overlapping crops do not compute interchangeable atom features. Simply
deduplicating their atoms would change the model silently. The proposed shared
backbone removes that center dependence; focal identity enters the readout.
This is a new architectural experiment with a matched reference, not an exact
runtime optimization or a continuation of old optimizer checkpoints.

## Interfaces and implementation sequence

1. **`FrameGraphBank`** stores one geometry per `(source, observed_frame, domain)`:
   species, physical positions, box, stable atom identities and periodic neighbor
   edges. Store edge displacement/image information explicitly. For sparse focal
   samples, construct the union of needed supports plus computational halos;
   do not default to a whole 70k-atom cell. Each bank owns its immutable geometry
   and batching plans, as in the new `GraphBank`.

2. **`SharedAtomMACE`** uses the fused native-layout spatial blocks, but returns
   per-atom scalar/vector/tensor features instead of a focal pooled state. There
   is no focal-center flag or focal taper in this shared computation. Ordinary
   smooth interatomic cutoffs remain. Keep degrees 0, 1 and 2 initially so the
   implementation builds directly on the tested runtime.

3. **`FocalReadoutPlan`** maps each sampled center to atom indices, minimum-image
   center-relative displacements and smooth region weights. Begin with local
   and two broader nested supports, for example 8/12/20 Å. Preserve weighted
   counts. These are proposed ablation settings, not established optimal radii.

4. **`ContextReadout`** pools equivariant atom features and geometric moments at
   each scale, then combines the focal state with regional evidence before the
   final invariant 128-dimensional export. Do not independently scalarize each
   region first: retain tensors and allow cross-region contractions. Include
   center-relative geometry, for example scalar atom features coupled with
   `Y_l(r_i-r_center)`, as well as tensor-valued atom summaries. Otherwise the
   readout can know that outer ordering is strong while losing where that order
   lies relative to the local environment. Normalize with smooth weights and
   declared scale factors, not hard membership counts.

5. **A later fine/coarse feedback variant** can add sparse regional nodes with
   actual spatial centers. Use their periodic displacement vectors, radial
   envelopes and spherical harmonics in equivariant messages. If region-to-atom
   feedback is shared at frame level, atom computation remains reusable. If
   feedback depends separately on each focal center, those updated atom states
   become center-specific again; confine such work to a small final local block
   and measure whether its information benefit justifies the duplicated work.

The first implementation should be steps 1–4. A fixed small set of smooth
regional readouts is easier to audit than learned clustering or hard nearest
representative reassignment, and directly addresses the orientation loss in the
previous invariant context tokens.

## Support and sampling must stay explicit

Two atom message-passing layers at a 5 Å cutoff need up to a 10 Å computational
halo beyond atoms whose features enter a readout. A 20 Å regional readout can
therefore observe geometry out to 30 Å. Broader context is additional prediction
information; record this effective support rather than advertising only the
8 Å focal radius. Additional coarse message-passing layers enlarge support too.
Build periodic unions with exact atom/image identity and verify they match a
whole-frame reference on the requested readouts.

Keep the existing draw of 256 training windows and its source/importance weights.
Group those draws by source/frame to reuse geometry and atom computation, then
restore the original draw mapping for loss reduction. Repeated centers retain
their sampling multiplicity. Do not silently replace the natural source-weighted
objective with uniform-frame sampling. Gradients from all focal losses must
accumulate into the shared atom features/backbone.

Use existing registered MD frames and their paired relaxation products. The
current flattened 80-atom patch cache is insufficient to reconstruct shared
outer neighborhoods reliably; a new derived frame/union cache is needed, not
new simulation. Resolve the actual source manifests through the dataset registry
and keep observed/relaxed domains, ancestry, atom IDs and original MD labels
aligned. Consult [DATASETS.md](../../DATASETS.md) before choosing releases.

No temperature, simulation age or explicit time covariates enter either encoder
or predictor. The first version is snapshot-only. Timestamps identify observed
frames and future labels; they are not learned context features. Relaxed inputs
still carry the declared full-current-cell relaxation access.

## Break-even and scientific validation

Independent patch cost scales with the sum of their edge counts. Shared cost
scales with the union edge count, plus halo and readout work. For a few separated
centers, union/whole-cell computation may cost more than independent crops.
Measure overlap and edge counts on the actual sampled cohort before choosing a
cache strategy. Parameter count or a GPU-utilization percentage does not decide
that comparison.

Use three separate comparisons:

* Shared backbone plus local-only readout versus the current center-conditioned
  crop model: isolates moving center conditioning to the readout.
* Shared local-only versus shared multiscale context at matched backbone size:
  tests whether surroundings add predictive information.
* Shared versus independently computed instances of the **same new model** with
  matched support: tests numerical parity and the computational saving.

Train/select the supervised branch by predictive likelihood. AP3/AP6 are
evaluation diagnostics only. Compare raw and calibrated log loss/Brier,
constant/descriptor controls and matched frozen linear/stronger probes. Measure
0.75 ps trajectory response where genuine pairs exist, movement/dataset spectra,
and noise RMS normalized by local spacing. Include support-boundary tests and
rotation/permutation tests. A matched predictor given original local/context
observations in addition to the state can test whether useful information was
discarded; it is not automatically a mutual-information estimator.

See [the prior context implementation](spatial_context_20260925.md) and
[runtime audit](performance_refactor_20260925.md). Literature motivations and
their limits are recorded there; no paper result is treated as an Al-onset result.
