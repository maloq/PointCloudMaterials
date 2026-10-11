# Al nucleus births with crystalline precursor tails

Execution and cooperative-worker details: [nuclei search performance](../nuclei_search_performance.md).

`al-nuclei-birth-010-histories-v1` is a birth-centred supplement to the pinned
`al-nuclei-birth-010` catalogue, fingerprint
`ebb3c0ea03a86db3ce86ced6a4e12251743d6d2e4a134f7344a4d6e9a67f51b9`.
It contains the same **295 PTM established-birth records in 110 source views**.
An original birth remains the top-level entity. Earlier crystalline atoms,
small clusters, branches and sampled appearance observations are attached to it;
they never become extra nucleus births. Source-view records retain their event
IDs; exact physical onset aliases and original independent ancestry remain explicit.

The parent `030` dataset was subsequently restricted to native 0.30 ps raw
trajectories. This changes the cumulative parent's fingerprint, but preserves the
`010` sources, event IDs and labels used by this frozen collection. Its recorded
parent fingerprint above remains the one used at launch. Readers validate the
actual source payloads when attaching histories to a later parent publication.
This supplement is for `010` only; it does not provide 0.30 ps precursor tails.

The hierarchy is:

```
original established birth / physical-onset alias group
  original source, ancestor and role; first-eight and establishment milestones
  precursor graph (1–7 atom components included; direct / gap / weak links)
  original embryo and growth trace (largest component size and fragmentation)
  appearance observations that share onset atoms with a reconstructed component
  saved-frame atom excerpts, properties and provenance
```

This is retrospective structural annotation, not new MD or new prediction inputs.
Existing birth labels are not filtered or rewritten. No neighborhood screening
is introduced. A local appearance that belongs to an existing growing family is
attached to that birth instead of being counted as another birth. If it matches
several birth histories, all links and the ambiguity count are retained. If it
cannot be linked, its status is `unassigned_within_reconstructed_coverage`;
this is **not evidence of dissolution or absence of a later nucleus**.

## Exact observations and limits

Reconstruction goes back up to 12 ps before the first-eight-atom catalogue frame,
using native 0.1 ps saved observations. Full-periodic-cell PTM uses RMSD cutoff
0.1; FCC/HCP/BCC atoms connect at 3.6 Å. The first component must exactly reproduce
the catalogue's founder atom IDs. Backward links retain shared atom counts and
both overlap fractions. Direct links have at least `min(4,n_parent,n_child)`
shared atoms and at least half of either component in common. One missing
observation can be bridged by an explicitly uncertain gap edge; weaker overlaps
are contextual branches and are not recursively expanded. A tail reaching the
lookback boundary is left-censored. A gap link never establishes uninterrupted
crystallinity. All qualifying branches are retained, including splits and mergers.

The original saved family trace supplies the history from first embryo onward;
its largest-component membership does not reconstruct every detached fragment.
Original fragmentation counts remain available in `family-history.json`.
Detailed atom movies show [-12,+2] ps around the first embryo and [-3,+2] ps
around establishment confirmation. The full original size history covers the
intervening interval, but a missing atom excerpt is not interpolated. Two GIFs
per illustrated birth show early appearance and establishment separately.

The 14 Å movie crop follows the founder atom nearest the original embryo centroid.
It shows current PTM type, founder atoms, reconstructed precursor atoms and
original largest-family atoms separately. Shape properties describe the visible
crop; a component may extend beyond it. Bond order is measured with the existing
nearest80 descriptor producer. See [metric definitions](../metrics/nucleus_histories.md).
Raw coordinates are already quantized saved MD, not exact integration restart states.

The three illustrated examples use the shortest embryo-to-establishment intervals
on different <=100k-atom ancestors. They are illustrative, not a random cohort
or evidence of model performance. All 295 catalogue records are scheduled for
history extraction, including the positions-only million-atom source. Inclusion
in geometry annotations does not grant eligibility for motion learning.

## Publication and access

Run output: `${storage:training_storage}/nucleus_histories/birth-tails-010-20261007-v1`.
Permanent dataset: `${dataset:al-nuclei-birth}/histories/birth-tails-010-v1`.
The derived dataset manifest and index are published **once**, after all histories
and the separate first-crystal observation search have completed. The publisher
does not wait for model evaluation. Working example views are explicitly partial
analysis outputs and do not incrementally change the dataset catalogue.

`histories.sqlite` has one `births` row per original event, `observations` for
positive appearance/persistence onsets, and a many-to-many `links` table.
Onsets join by source, exact frame and atom ID. Before the first embryo, the
precursor graph supplies relations; afterward, original saved component labels
and family traces supply them. Weak/gap relations remain distinguishable.
The raw first-crystal supplement retains all candidates and horizon labels,
including negatives and observations outside its risk set.

```python
from src.postprocessing.datasets.births import NucleiBirthDataset
from src.postprocessing.datasets.histories import NucleiBirthHistories

with NucleiBirthDataset('010') as data, NucleiBirthHistories(data) as histories:
    births = list(histories.births())  # original event rows; explicit aliases
    history = histories.history(births[0]['birth_event_id'])
    # history['precursors'], history['family_trace'], history['observations']
    # history['movie_frames'], history['movie_path']
```

The parent catalogue remains unchanged. Both supplements have their own content
fingerprints and durable provenance. Reader access validates source identities
and saved packet hashes. Partial work is inspected in the analysis directory,
not silently exposed as a complete dataset.

## Reproduction

Use conda `pointnet-torch214`; do not repeat `launch` after a queue receipt exists.

```bash
python -m src.postprocessing histories prepare --config configs/data/nuclei_birth_histories_20261007.json
python -m src.postprocessing publish launch --config configs/data/nuclei_birth_histories_20261007.json
python -m src.postprocessing publish examples --config configs/data/nuclei_birth_histories_20261007.json
```

The queue freezes source code, configuration and metric definitions and records
job IDs under `technical/queue.json`. CPU workers share a verified full-cell PTM
cache; this work does not need a GPU. Existing completed histories are verified
and reused on resume. The finalization job depends on successful collection.
Failures leave the dataset unsealed and the failing lane receipt available.
Six logical lanes divide the original event records. Five run in parallel in
12-hour CPU allocations; a dependent 24-hour completion job runs the remaining
lane, resumes any incomplete packets, and publishes once. This uses six submitted
job slots under the cluster's per-user limit. Completed packets and PTM frames
are reused. The initial six-worker submission exceeded the remaining submit
capacity when adding the completion job; its sixth worker was deferred after
ownership was verified, and its verified cache is retained in the same protocol.
