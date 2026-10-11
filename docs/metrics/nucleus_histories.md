# Birth-centred crystalline histories

Execution optimization, October 7: output/cache locations are resolved once per
worker; unrelated dataset aliases are no longer expanded on every lookup. The
scientific calculations are unchanged. Each newly completed packet includes
`execution.json` with its actual implementation hashes, machine and Slurm job,
including helpers processing a previously frozen scientific plan. Helpers claim
only unlocked packets and never replace completed ones. See the
[performance audit](../nuclei_search_performance.md).

The descriptor revision computes only the three exported l=6 quantities from
the existing nearest80 producer, retaining its operation order exactly. The
unused l=2/4/8 fields, Wigner invariants and quantiles are omitted. Atom-level PTM
may be reused from checksum-verified extraction chunks only with matching RMSD
cutoff and PTM producer hash; old minimum-four-atom component labels are never
used to reconstruct missing atom types. Thread counts are execution metadata.

The master population is the pinned `al-nuclei-birth-010` PTM established-birth
selection. Each original event ID remains a nucleus record. Small connected
components and local appearance observations become history nodes or supporting
observations, never new nucleus events. Original times, establishment criteria,
roles and ancestry remain unchanged. Histories include geometry-only sources
without velocities; this does not make them eligible for motion prediction.

Before each catalogue first-eight-atom frame, up to 12 ps of exact 0.1 ps original
MD observations are classified with full-periodic-cell PTM, RMSD cutoff 0.1.
FCC/HCP/BCC atoms connect at 3.6 A. Components of sizes one, two and three are
retained in the new history graph. The anchor component must reproduce the
catalogue's recorded first-member atom IDs exactly. Full-cell labels are cached
with source/frame/producer hashes and reused across events. No cropped-patch
PTM, interpolated coordinates, quenches or new MD are used.

Backward edges require shared atom identities. A direct overlap has at least
min(4,parent_size,child_size) shared atoms and shared/max-fraction at least 0.5
(equivalently max(shared/parent_size, shared/child_size) >= 0.5). Both overlap
fractions and shared counts are retained. Strong-overlap edges across one missing
frame are marked `gap`; weaker nonzero overlaps are `weak_overlap`. Weak-only
nodes provide context but are not recursively expanded. A direct path to the
anchor has no gaps or weak edges. Gaps do not imply continuously observed
crystallinity; small-cluster identity links are retrospective associations, not
proof of causation. Splits/mergers retain all qualifying edges. Histories that
reach the lookback boundary are left-censored, not declared first-ever onsets.

The original saved family-size trace supplies the full recorded embryo/growth
history, including its gaps and fragmentation counts. Existing established
families encountered by precursor nodes are flagged as connections. The new
history graph does not silently change an established birth's origin or merge
catalogue events. Exact duplicate-view grouping uses ancestor, physical onset
time, founder IDs, their periodic positions and the box; source-view event IDs
remain available as aliases and do not increase independent-ancestor counts.

Atom movies cover the early interval [-12,+2] ps relative to the first-eight-atom
frame and [-3,+2] ps relative to establishment confirmation. Their union contains
only actual saved observations. Disjoint excerpts remain explicitly separated;
no footage is fabricated through long waiting intervals. A 14 A sphere follows
the founder atom nearest the recorded embryo centroid. Atom IDs, physical
coordinates, original box, PTM types, original largest-family membership and
retrospective precursor/founder membership are stored separately. Movies are
local crops; components extending outside the crop can be larger than the visible
count. Coordinates are retained as float32 local visualization values derived
from already stored original positions, not new simulation/restart states.

Measured movie properties are local crystal fraction, centre q6, neighbour-
averaged qbar6, q6 neighbour coherence and mean first-shell spacing from the
existing nearest80 descriptor producer. Radius of gyration is sqrt(sum of
covariance eigenvalues) of the visible precursor/original-family atoms; anisotropy
is (largest-smallest eigenvalue)/sum. Empty selections are undefined, not zero.
Crystal-type counts separate FCC/HCP/BCC. First-member retention, graph size,
weak/gap edges and lookback censoring describe this reconstruction, not predictive
accuracy. Histories, future membership and atom colors are retrospective viewing
metadata; they are not passed to encoders or birth predictors.

Three illustrative movies use the shortest recorded embryo-to-establishment
intervals on different <=100k-atom ancestors. They are not a random sample or
evidence of general predictive performance. The full catalogue is processed
separately; pending and completed histories remain explicit. Older cadences and
unsearched Al data are outside this exact-0.1-ps history view.


## Parent dataset cadence contract (October 7, 2026)

The shared catalogue now restricts `al-nuclei-birth-030` to uniformly saved
native 0.30 ps inputs, checking raw timestep arrays. The `010` source/event
population used here is unchanged. New preparations pin the current parent
fingerprint; frozen runs keep their original pins and metric definitions.
Source-bound 0.10 ps label/history supplements retain their original provenance.


## Shared frame execution and resumption

The frame cache is bound to the frozen history plan, source and executing code.
It reuses the identical periodic nearest-80 queries for every declared anchor and
full-cell precursor components, including one-to-three-atom clusters omitted by
minimum-four-atom tracking labels. Source-frame nodes keep their original IDs,
ordering and established-family associations. Event-specific observations and
masks are saved after each frame. Graph linking and shape calculations retain
the original per-event rules and use those exact saved observations. Atomic,
checksum-bound checkpoints preserve numeric dtypes and float values; completed
birth packets keep their original identities and execution records. Interrupted
frames are recalculated, while verified completed frames are reused.
