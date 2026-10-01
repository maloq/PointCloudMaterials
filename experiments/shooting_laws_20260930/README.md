# Physical distributions of possible futures

Question: which observed structural differences change the distribution of
joint 3/6/12 ps local futures, and is this information accessible beyond simple
bond-order, topology and smooth radial-density controls?

This implements the September 30 return to the shooting/predictive-atlas idea.
The first experiment uses frozen observed-geometry MACE VICReg and aligned Epi
epoch24, seed20260926, plus descriptor controls. It trains no new encoder and
uses no temperature, simulation age, absolute time, velocities or relaxed inputs.
Raw 128-dimensional encoder exports are distinct from the VICReg loss projector.
The retained encoders had same-time relaxed training views and a random Epi
reference; those are training-only, not prediction-time observations.

Nine inputs x three fit seeds x two separately selected likelihood heads:
no input; radial density; bond order; TDA; order+TDA; frozen VICReg; frozen Epi;
VICReg+order/TDA; Epi+order/TDA. One head fits a joint 24-dimensional physical
path mixture, the other a finite-horizon event-time distribution. Both use the
same source-held-out likelihood selector. Readouts are diagnostic and stay local.

The stratified parent sample covers strict-clear liquid, visible interfaces and
crystalline centers. Natural parent-population weights and source bootstrap
uncertainty are retained. Event fitting excludes currently crystalline centers.
No atom-specific evaluation rows are dropped for a particular model. The Al64
contract identifies frozen encoder pretraining; this historical shooting assay
does not alter the fixed all64/legacy16 benchmark or its source roles.

The original outer validation sources are a reused historical test, never a
new prospective confirmation. Three source-seed35863 runs are selection within
the historical training population. Descriptor transformations use fitting rows.
The original and nested campaigns overlap and have incompatible historical
roles; their data are never pooled for fitting. Per-parent diagnostic exposure
is explicit. Four-shot nested/Ta results cannot establish precise individual
probabilities or independent-root transfer.

Primary evidence is held-out likelihood, energy score and event calibration,
particularly in crystal-free liquid. AP remains diagnostic. Split-shot analysis
quantifies finite-target noise. Atlas retrieval is secondary and uses disjoint
shots to choose and score its empirical oracle. Failure of a fitted readout is
not proof of zero physical information.

The independent diagnostic queue scores the thermostat top-up, nested Al and
Ta on the same structural target definitions, without fitting to these campaigns.
It does not add Ta event labels or reinterpret censored nested first passages.
History is a subsequent matched intervention: archived source history must be
located and its exact common physical offsets verified before adding a history
arm. The first submitted release is explicitly snapshot-only.

[Protocol recipe](../../configs/shooting_laws/al480_20260930.json) ·
[Metric definitions](../../docs/metrics/shooting_laws.md) ·
[Execution](../../docs/shooting_laws.md).
