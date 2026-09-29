# Crystallization-origin availability counts (v2)

Producer: `src/research/crystallization_origin/{extract,ancestry,audit}.py`.
This audit fits no model and does not redefine historical outcomes or AP scores.

## Population and units

The immutable Al64 release supplies 150 independent sources, 64 atom identities
per source, the 90/15/15/30 train/selection/calibration/test roles, original PTM
labels and exact prediction sample IDs. The full original periodic cell is
processed at every 0.75 ps observation, including the start of each trajectory.
No relaxed coordinates, temperature, time covariates or learned encoder enter
the classifier. Timestamps organize labels only.

Tables distinguish:

- `distinct_clusters`: one operational establishment per tracked crystal lineage,
  across the entire observed cell and trajectory. These are **not** confirmed
  physical critical nuclei. Frame-zero establishments are left-censored existing
  crystals, never new births.
- `distinct_clusters_covered`: establishments whose initial cluster centroid is
  within 8 Å of at least one tracked center at the birth frame. Each cluster is
  counted once regardless of how many centers cover it. All64 and legacy16 are
  different spatial coverage sets.
- `onset_windows`: the exact fixed, currently-liquid prediction population.
  Each positive window has one origin-relative ancestry label. Negative windows
  are retained as `no_onset_within_horizon`; totals match the fixed population.
- `distinct_onsets_represented`: unique source/atom first-onset events represented
  by positive windows, separately within each label and horizon. The same onset
  can have different ancestry labels from different forecast origins, so these
  counts **must not be summed across labels** as independent events.
- `regional_birth_windows`: fixed at-risk windows with an isolated establishment
  inside the moving center's 8 Å region during `(origin, origin+horizon]`.
  This target does not require the center atom itself to crystallize. It is
  separate from first-onset attribution; multiple births in one window count once.

`sources_with_label` counts independent source IDs contributing at least one
row/event. `role=all_completed` aggregates only finished sources and is not a
population estimate until the queue is complete. Horizon zero marks full-
trajectory cluster counts, not a zero-lead forecasting task.

## Operational crystal and ancestry definition

Original MD positions are read at their stored precision and converted to
float64 for periodic geometry. Full-cell OVITO PTM uses RMSD cutoff 0.1 and FCC,
HCP or BCC as crystalline, matching the established benchmark classifier.
Crystalline atoms separated by at most 3.6 Å form an undirected periodic graph.
Only components containing at least four atoms enter temporal tracking; all
structural types remain available in the compressed PTM cache.

Adjacent-frame overlap is strong when at least four atom identities overlap and
the intersection is at least 25% of the smaller component. One missing frame
can be bridged only for components without a strong immediate predecessor.
Any weak overlap with an established ancestry is retained and marked uncertain,
rather than falsely creating a new birth. Splits inherit ancestry; merges keep
all established ancestors. Possible ancestry and strongly supported ancestry
are tracked separately. A component with only weak ancestry is unresolved; a
weak peripheral branch does not erase an independently supported strong path.
Possible extra ancestors are still included when comparing origin classes, so
disagreement between them remains unresolved.

The reference establishment is at least 64 crystalline atoms for three adjacent
observations: 1.5 ps spans the first and third frames. Sensitivity definitions
are 32 and 128 atoms for three frames, and 64 atoms for five frames (3 ps span).
These are predeclared engineering criteria, not fitted critical sizes.
Each continuous size-qualified streak is confirmed at its final required frame
and backfilled to its first frame for label construction. Backfill also updates
observed side branches, preventing the same streak from being counted twice.
Missing-frame links preserve ancestry but never extend a persistence streak.

Birth position is the periodic mean obtained by unwrapping the birth component
around its first atom. Components spanning more than half a box length on an
axis are marked `unresolved_periodic_extent`; their centroid is not trusted as
a nucleus location. Merging multiple size-qualified precursor paths during
establishment is `unresolved_establishment_merge`. A birth within 8 Å (minimum
atom-to-atom distance) of another crystal already confirmed by that birth frame
is `interface_associated`, not assigned to isolated nucleation. The remainder
are `isolated_establishment`. No claim of thermodynamic commitment follows.

The catalogue retains peak component size and final observation before merging
while ancestry is unique, plus `ever_merged`. These are growth/lifetime audit
fields, not additional hidden acceptance filters. Once clusters merge their
combined size is not attributed separately to each birth.

## First-onset attribution

Historical local onset remains the first run of three FCC/HCP/BCC frames.
Ancestry must be available for the tracked atom throughout that run. A full-cell
PTM disagreement with the fixed classifier during that run marks the outcome
`unresolved_ptm_disagreement`; historical labels are never overwritten.

For each positive fixed window and responsible lineage:

| Label | Rule |
| --- | --- |
| `existing_crystal_arrival` | Establishment was confirmed at or before the forecast origin. |
| `local_nucleus_establishment` | Isolated birth after the origin and no later than local onset, within 8 Å of the center at birth. |
| `external_new_crystal_arrival` | Same temporal rule, but birth outside the local 8 Å region. |
| `formation_in_progress` | Size-qualified streak began by the origin but was not yet confirmed. |
| `interface_associated_new_cluster` | A new lineage formed near an already established crystal. |
| `unresolved_unestablished` | No established ancestry supports all three onset frames, or establishment begins after local onset. |
| `unresolved_ancestry` | Weak-only ancestry, unsuitable birth geometry/merge, or multiple roots imply different origin classes. |

Several established ancestors can still unambiguously imply existing-crystal
arrival when all were confirmed by the origin. If ancestor classes disagree,
the window is unresolved. This is ancestry-based capture, not a measured front
velocity or proof of a particular atomistic growth mechanism.

The 3 and 6 ps comparisons include fully observed confirmation frames beyond
the endpoint, as supplied by the fixed cohort contract. Future information is
used only for outcomes. Test/calibration results never select these definitions.

## Interpretation limits

Counts depend on PTM cutoff, connectivity, atom overlap, persistence, spatial
definition and the 0.75 ps observation cadence. A gap longer than the one-frame
bridge can split an apparent lineage; sub-frame births, merges or remelting
cannot be resolved. A disconnected interface-associated birth is not proof of
heterogeneous nucleation. PTM connectivity does not preserve grain orientation.
The four threshold variants quantify size/persistence sensitivity only; they do
not validate connectivity or classifier robustness. Spatial review and a bond-
coherence/orientation comparison should precede using these labels as ground
truth for new training. The first five training-source audits are prioritized;
the same frozen rules apply to the remaining sources.
