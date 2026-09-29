# CIV-MACE128: distance and direction to an interface atom layer

This separate target protocol reuses the CDV-MACE128 architecture, initialization,
proper predictive losses and diagnostics in [crystal_vector.md](crystal_vector.md).
Its target is **not** distance to the entire crystal set. Never compare absolute
NLL/error numbers across these two different target definitions as a model ranking.

Let C be the existing current >=64-atom, past-confirmed PTM crystal reference set.
On its complement, construct the periodic 3.6-A neighbor graph used by the Al
lineage producer. Keep non-crystal components of at least 64 atoms; call their
union L. The crystal-side interface layer is

`I = {i in C : min_{j in L} periodic_distance(i,j) <= 3.6 A}`.

The target is the unsigned periodic distance to the nearest atom in I. A layer
atom has distance zero. A crystal interior atom has positive distance to the
layer; an exterior atom also has positive distance. The target vector points
toward the nearest layer atom, from either side. This is an atom-layer convention,
not a fitted mesh, signed distance, grain-orientation boundary, or Gibbs surface.
The one-neighbor-shell thickness gives an atom-scale offset relative to a continuum
surface. The component threshold removes small isolated PTM defects; large internal
disordered pockets and connected disordered grain boundaries may still define
interfaces. Small unconfirmed crystals belong to the complement of C. The threshold
is an explicit label-resolution choice, not an estimate of a critical nucleus size.

With no interface anywhere in the periodic cell, the target is +infinity and the
distance likelihood is right-censored at 64 A. This includes fully liquid and fully
crystalline cells. No artificial zero/boundary at the simulation-box faces is added.
Directions are masked for zero distance, d>=64, or nearest-distance ties within
1e-5 A. At a crystal's medial axis the tied direction is undefined, even though its
distance remains a valid target. Phase, interface membership, visibility, timestamps
and component labels are never supplied to either encoder or predictor.

All 126,545 fixed rows and 22,872 original scan rows are retained with their original
geometry and identities. Original train/selection uniform centers are reused.
Because the fixed-at-risk benchmark contains no crystal-interior rows, add 16
uniform atoms without replacement per retained calibration/test frame, using
`default_rng(20260926 + source_id)` over ascending frames. Selection is independent
of labels. These new queries are a separately named `uniform` evaluation track;
they do not replace fixed rows or change source roles. 90/15/15/30 train/selection/
calibration/test ancestors remain fixed. Geometry of original queries is checked
against the sealed parent, and the new layer's distances cannot be below distance
to the full crystal set. Original crystal distances and visibility remain in metadata.

Train with independent replacement draws: half fixed-at-risk and half uniform
centers, equal source mass in each half; batch/microbatch 256. Every row has unit
loss weight. No phase/distance labels set sampling probabilities, quotas or batch
acceptance. The preparation audit reports actual expected near-interface and
crystal-interior counts, plus binomial empty-batch probabilities. `zero_distance`
means interface membership, whereas `inside_crystal` includes layer and interior.

Three arms: distance only; distance plus direction; distance plus direction with
VCReg. Each starts from the same original completed snapshot CD-MACE128 parent,
fresh optimizer, one matched seed, width/export 128, vector export 16. 16 nominal
epochs of 248 updates; selection among epochs 12–16 by the arm's predictive
likelihood objective, excluding VCReg. Distance is zero-inflated lognormal, censored
at 64; the joint mixture uses normalized von Mises-Fisher directions with true-distance
concentration `8/(1+(d/16)^2)`. Proximity log losses at 8/12/20/32 A now refer to
being near the interface from either side. They are predictive Bernoulli losses,
not AP optimization. VCReg, online training tracking, and geometry-only inputs are
unchanged. Known labels/phase are used solely for supervision and stratified reports.

`distance.csv`, `direction.csv`, and `reliability.csv` retain the definitions of the
CDV diagnostic calculations but use interface targets/visibility. Export separate
`*_by_phase.csv` tables for crystal interior (C minus I), interface layer I, outside
C, and no-interface cells. The last category overlaps the phase categories and is
not a partition. All scores remain equal-source within the named population and
phase. Capped mean RMSE/median MAE and marginal NLL are comparable across the three
arms; directional likelihood/error is reported only for directional arms. Empty
groups are not given zero error. Interior performance should be read on uniform
held-out rows, not inferred from the fixed liquid-only benchmark.

The preserved scan paths were designed to approach crystal from liquid. For this
protocol evaluate only the exterior prefix before first crystal entry, never an
exit-side alarm. Require two consecutive probabilities strictly >0.5/0.75/0.95;
report misses, warning distances to I, all-path recall at 12/20 A and interface
visibility at alarm. Paths with fewer than two exterior observations count as
misses and are also reported explicitly. `away_alarm_rate` describes alarms on the
historical away paths; it is not proof that a near-interface prediction is false
under the new target. No interior-traversal warning claim is made by these paths.

Physical-information ridge readouts, scalar/vector ranks, normalized noise response
and independent-snapshot 0.75-ps stability retain their existing definitions. These
are diagnostic evaluations, not reconstruction pretraining or temporal inputs.
Scientific training alone creates online W&B runs; evaluation updates their recorded
IDs. Frozen metric documents and implementation hashes accompany all exported CSVs.
