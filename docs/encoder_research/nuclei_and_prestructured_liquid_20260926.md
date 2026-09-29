# Nuclei and prestructured liquid: methods and current limits

Literature and implementation review, 26 September 2026. No label definitions,
running simulations or historical results changed by this review.

For the reusable literature analysis, candidate contributions and later evidence
checks, see [nucleation literature and open questions](nucleation_literature_and_open_questions.md).

Our current birth detector is a reasonable operational detector of persistent
crystal establishment. It is not a validated critical-nucleus detector. The
current birth harvest measures several liquid-order descriptors but does not
yet supply a calibrated, spatially resolved prestructured-liquid definition.

## Distinguish the physical questions

| Quantity | Meaning | Evidence needed |
| --- | --- | --- |
| Locally crystalline atom | Its immediate environment resembles a crystal motif | Structural classifier and uncertainty/calibration |
| Ordered or prestructured liquid region | Correlated order differs from reference liquid without necessarily forming a bulk-like crystal | Local order, inter-atom coherence, spatial extent and reference distributions |
| Crystalline embryo/cluster | Connected collection of crystal-like atoms, potentially transient | Declared connectivity and temporal tracking |
| Operational establishment | A cluster crosses a declared size/persistence condition | Reproducible event rule, including sensitivity analysis |
| Critical nucleus | Configuration near the transition between dissolution and growth | Dynamical commitment or a validated reaction-coordinate/free-energy analysis |

These do not impose a universal sequence of distinct phases. Boundaries can be
diffuse and mechanisms depend on material, potential, undercooling and observation.
An individual PTM-positive atom is not itself a nucleus; a subthreshold cluster
is not automatically liquid in a structural sense.

## Established approaches

**Template and local-topology classifiers.** Common-neighbor analysis and
polyhedral template matching identify local crystalline motifs. PTM is designed
to be more robust to thermal disorder than distance-threshold CNA and can also
recover lattice orientation. Its output describes structure, not future
commitment. [Larsen, Schmidt & Schiøtz (2016)](https://arxiv.org/abs/1603.05143).

PTM's RMSD cutoff remains a scientific choice: relaxing it trades fewer missed
identifications for more false positives. OVITO recommends 0.1 as useful for
defect identification in crystalline solids, not as a calibrated universal
nucleation threshold. [OVITO method documentation](https://www.ovito.org/manual/reference/pipelines/modifiers/polyhedral_template_matching.html).

**Bond-orientational order and correlated bonds.** Steinhardt coefficients
describe the directions of bonds around each atom. Their scalar contractions
q4/q6 and w4/w6 describe local symmetry. Normalized q6-vector correlations between
neighboring atoms measure whether local environments share orientational order.
Counting coherent neighbors helps distinguish extended ordering from a single
ordered coordination shell. The bond and neighbor-count thresholds require
calibration for the chosen system and neighbor definition. The local solid-bond
construction is described in the introduction of
[Lechner & Dellago (2008)](https://arxiv.org/pdf/0806.3345).

Their averaged coefficients combine each atom's complex coefficients with its
neighbors before taking invariant contractions. This helps discriminate crystal
structures and includes information from beyond one coordination shell. It is
not equivalent to averaging already scalarized q6 values. Keep both local and
averaged measures because averaging also broadens spatial support.

**Prestructured regions surrounding a crystal core.** In Ni, Díaz Leines and
Rogal identified coherently ordered particles not assigned a bulk crystal
structure and found that including this surrounding region improved the
description of nucleation compared with crystalline core size alone. This is
direct motivation for inspecting our spatial context, not evidence that Al
has the identical pathway or transferable thresholds.
[Primary study](https://arxiv.org/abs/1810.04782).

**Topological descriptions of liquid motifs.** Persistent-homology descriptors,
Voronoi environments and related motif analyses can reveal structure beyond
ideal crystal templates. Becker et al. used topological learning on metal
inherent structures and reported emergence in regions with low fivefold symmetry,
with positional and orientational ordering developing together. This differs
from assuming that every material has a distinct orientational precursor stage.
Their use of minimized coordinates must remain explicit in comparisons.
[Becker et al. (2022)](https://doi.org/10.1038/s41598-022-06963-5).

**Dynamical commitment.** Define a liquid basin A and a growth basin B, then
estimate the probability of reaching B before A under a declared dynamical
ensemble. Values near one half identify a transition-state ensemble; simply
remaining ordered for 1.5 ps does not establish this condition. Structural
information beyond cluster size can matter: Ni3Al work identifies size,
crystallinity and chemical order as relevant variables.
[Liang et al.](https://arxiv.org/abs/2004.01473).
The alloy result does not authorize adding species channels to our encoders.

## What the current implementation actually does

Read producers: `crystallization_origin/extract.py`, `ancestry.py`, `harvest.py`,
and the reused `geoframe_evolution/reference.py:bond_descriptors`.
The [frozen metric definition](../metrics/crystallization_origin.md) gives the
complete contract:

1. Full periodic original MD frames, native saved precision; PTM at RMSD 0.1.
2. FCC/HCP/BCC atoms connected by distances no greater than 3.6 Å in Al.
   Connectivity does not use bond coherence or lattice orientation.
3. Components of at least four atoms enter tracking. Strong ancestry requires
   four shared identities and overlap of at least 25% of the smaller component.
   One missing frame can be bridged under the documented restrictions.
4. Establishment requires 64 atoms over three 0.75 ps observations: a 1.5 ps span.
   Confirmation is later than the reported first crossing. Future confirmation
   is used for labels only; origin-time eligibility is causally gated.
5. Existing-lineage contact, nearby interfaces, weak ancestry, merges and
   ambiguous periodic extents are treated separately. An 8 Å isolation rule
   is operational and does not prove absence of influence from a distant crystal.

These choices preserve useful protections: original trajectories define outcomes,
small precursors may remain eligible, disappearing embryos can remain controls,
and arrival is not automatically relabeled as a new birth. Keeping uncertain
ancestry explicit is preferable to assigning it a confident origin.

The [harvest descriptors](../metrics/nucleus_harvest.md) are sampled at candidate
and control centers, using 12 nearest neighbors and their own neighborhoods:
q4, q6, normalized w4/w6, averaged q6, mean q6 coherence, local neighbor density,
and coherent-bond counts at 0.65/0.70/0.75. They do not currently change which
components are accepted as births or construct a full precursor shell. Averaged
q4, PTM RMSD/orientation outputs and precursor-region tracking are not present
in this birth-label pathway. Earlier reference analyses contain heuristic
ordered-liquid classes; they do not constitute validation of this birth detector.

## Ambiguity already demonstrated in our data

Existing full-cell isolated-establishment counts from the **90 training sources**:

| Size criterion | Required observed span | Candidate births |
| --- | ---: | ---: |
| 32 atoms | 1.5 ps | 253 |
| 64 atoms | 1.5 ps | 206 |
| 128 atoms | 1.5 ps | 167 |
| 64 atoms | 3 ps | 201 |

Source: the completed origin audit's `tables/label-counts.csv`, filtered to
`role=train`, `unit=distinct_clusters`, `track=full_cell`,
`label=isolated_establishment`. These are existing metrics with their preserved
definitions/hashes, not a new fit or threshold selection. Count similarity alone
does not establish event-identity or birth-time agreement.

Outstanding ambiguities:

- **Order versus commitment:** a long-lived cluster can still dissolve. A true
  transition may happen before or after our size crossing.
- **Core versus surrounding region:** PTM can miss a relevant disordered but
  coherent shell; its reported cluster size need not equal a bond-order size.
- **Polymorph and grain identity:** geometrically touching ordered atoms can
  span different orientations. Conversely, one twinned nucleus can contain
  multiple orientations, so a grain boundary is not automatically a new birth.
  HCP environments in Al may represent stacking faults rather than a separate
  HCP nucleus. Retain structural composition rather than only the union count.
- **Ancestry:** an ordered front can flicker out of PTM visibility and reappear;
  rapid member exchange, splits and merges challenge identity-overlap tracking.
- **Sampling and precision:** 0.75 ps observations cannot resolve sub-frame
  establishment or remelting. Float64 arithmetic does not recover information
  lost in float16 coordinates. The new paired precision trajectories allow this
  particular uncertainty to be measured.
- **Prestructure is not necessarily productive:** high local order can belong
  to competing motifs. A precursor criterion must also be evaluated on regions
  that never establish a crystal, not defined only by looking backward from births.

## Recommended validation before a definitive birth benchmark

Keep the current detector as versioned baseline and label its counts
**isolated establishment candidates**. Add a parallel structural description
of crystal core, coherently ordered surrounding region, ordinary liquid and
competing fivefold order. Prefer continuous scores plus uncertain assignments
over forcing every interfacial atom into a sharp phase label.

Calibrate structure discrimination on training-side reference liquid and crystal
states spanning the intended conditions. Declare neighbor definitions, inspect
q4/q6 and averaged q4/q6 jointly, and retain PTM RMSD/orientation. Do not import
hard-sphere or Ni numerical thresholds unchanged. Temperature can stratify an
audit; it remains excluded from model inputs.

Compare PTM and bond-coherence catalogues using matched events, count differences,
membership overlap, birth-time shifts, near-interface disagreements and
growth/remelting outcomes. Extend sensitivity beyond size/persistence to PTM
cutoff, connectivity and temporal matching. Do not select definitions by AP.

Use the new 0.15 ps trajectories both at native cadence and downsampled to
0.75 ps, with identical physical persistence durations. A 1.5 ps span requires
11 native observations, not three. Compare float32 and float16 observations of
the same frames. Relaxation may provide a separate structural diagnostic but
must not silently redefine original-MD outcomes.

For a smaller representative set, a separately specified commitment study can
test growth versus dissolution. With deterministic MD, copies of an identical
full phase-space state do not generate independent outcomes. Resampling momenta
defines a position-conditioned ensemble and must be recorded; it is not the
same task as forecasting from the actual measured velocity/history. Define
competing basins and external-front outcomes before launching this analysis.

This review does not show that the running generation campaign is invalid.
Its uniform full-cell observations can support alternative structural labels;
the 10% stopping threshold limits late-time coverage and is not itself a nucleus
label. The appropriate claim today is prediction of declared establishment
events, with physical criticality and precursor-region validation still open.
