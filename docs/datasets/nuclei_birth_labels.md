# Additional first-crystal labels for Al nucleus births

The versioned supplement **`first-crystal-010-v1`** belongs to
`al-nuclei-birth-010` and is also registered as
`al-nuclei-birth-010-first-crystal-v1`. Permanent storage is
`${dataset:al-nuclei-birth}/additional_labels/first-crystal-010-v1` on STORE.
The parent SQLite event catalogue, birth definitions, cadence manifests and
fingerprint are unchanged. The supplement has a separate content fingerprint.

These are labels of **local observations**, keyed by source, original global
frame and center atom ID. They are not one label per established nucleus and do
not require an eventual established nucleus. The sampled sites were chosen
before examining their future. Source IDs join the existing catalogue; copied
producer provenance retains original source manifests, frame mappings, ancestry,
roles and generating-potential metadata, including unknown values.

For a birth-centred view, use the separate
[nucleus histories](nuclei_birth_histories.md) supplement. It attaches matching
appearance observations and precursor branches to existing birth IDs. Raw
observation rows in this label supplement never create additional nucleus events.

## Targets and arrays

| Target | Meaning |
|---|---|
| `appearance` | First future FCC/HCP/BCC PTM classification of any consumed present atom |
| `persistent` | First of three consecutive saved frames where the same atom is PTM-crystalline |

Both targets are saved at **0.1, 0.5, 1, 3 and 6 ps**. The experiment currently
fits the persistence sensitivity only at 3 ps, but its saved onset permits all
five horizons. Transient structures count for `appearance`; neither an 8-atom
embryo nor 64-atom nucleus threshold applies. Positions are unrelaxed and PTM is
computed in the original full periodic cell at RMSD cutoff 0.1. Member atom IDs
are fixed at the present, restricted to the consumed nearest80/radius8 support.
See the frozen producer's full scientific definition in
`provenance/label-definitions.md`.

Each immutable `windows/<window_id>.npz` contains sampled center atom IDs,
80 member IDs, the consumed mask, local per-atom PTM labels at every saved
frame, exact source frame indices and times, the present-frame index, and the
producer's present-clear/history-clear/onset arrays. Future follow-up extends
to 6.2 ps to confirm three-frame persistence for onsets through 6 ps. Past
0.5 ps labels support a history-clear diagnostic. No raw coordinates or
velocities are duplicated in the supplement; input hashes and the copied
source/frame provenance identify the existing Q,V data.

`labels.sqlite` provides indexed sources, windows, observations and horizon
labels. Observations retain `center_atom_id`, source/frame/time joins, ancestry,
original role, Q,V input hash and:

- `at_risk`: no present crystalline atom in the consumed patch;
- `history_clear`: no crystalline atom in the recorded past 0.5 ps, for audit;
- first and persistent onset steps, source frames and `(lower, upper]` time
  bounds at the 0.1 ps observation cadence;
- final `cohort_included`, `canonical_id` and `cohort_weight`, once the producer
  seals its exact-duplicate policy and equal-ancestor evaluation weights.

The SQL horizon label is **null for a present-crystalline observation**, since
it is outside this appearance risk set. It is not converted to a negative.
For an at-risk observation, zero means no qualifying onset by that horizon
under complete recorded follow-up. Missing windows are absent and the supplement
remains unpublished until the collection is sealed. The NPZ packet retains every sampled candidate,
including those outside the risk set. Past history is not a primary filter.

These label fields and metadata are targets/audit information, not encoder or
predictor inputs. Different horizons, member atoms and source views do not
create independent melt ancestors. The supplement records the next first
observed local appearance after a clear present, not first-ever crystallinity
throughout an atom's unobserved history.

## Read and pin

```python
from src.postprocessing.datasets.births import NucleiBirthDataset
from src.postprocessing.datasets.labels import NucleiBirthLabels, label_sets

with NucleiBirthDataset("010") as births:
    print(label_sets(births))  # Includes state, fingerprint and coverage.
    with NucleiBirthLabels(births, "first-crystal-010-v1") as labels:
        labels.pin("output/my_experiment/technical/appearance-labels.json")
        rows = list(labels.observations(target="appearance", horizon_ps=1.0))
        # Default: the complete, deduplicated, at-risk experimental cohort.
        packet = labels.window(rows[0]["window_id"])
        source = births.source(rows[0]["source_id"])
```

Pass `expected_fingerprint="RECORDED_SUPPLEMENT_FINGERPRINT"` to require an
exact label release. A later cumulative parent catalogue can open this older
supplement only when all of its original source records still match. No extra
labels are implied for newly added sources. Cadence 0.30 ps readers do not list
the 0.10 ps supplement.

`cohort_only=False` exposes observations excluded by final duplicate handling.
`at_risk_only=False` also exposes present-crystalline candidates with null targets.
Readers reject incomplete manifests by default; the current publisher emits only
the completed supplement. `status` reports `waiting_for_sealed_collection` before
the first publication.

## Publication and continuation

```bash
python -m src.postprocessing labels launch \
  --config configs/data/nuclei_birth_first_crystal_20261007.json
python -m src.postprocessing labels status \
  --config configs/data/nuclei_birth_first_crystal_20261007.json
python -m src.postprocessing labels verify \
  --config configs/data/nuclei_birth_first_crystal_20261007.json
```

The launch schedules **one export after label collection is sealed**. The waiting
process checks only for the producer's completion receipt; it does not repeatedly
copy windows or update the dataset. At completion it validates upstream file
checksums and independently enumerates onsets from the already computed PTM
arrays. It does not run PTM, simulation, training or change the active experiment.
The publisher's source snapshot, ownership, process receipt and progress live in
the appearance run's `technical/additional-label-*` files. Use its frozen
`watch` command to resume the same publication contract after interruption.

Once the producer seals all 456 windows and its final observation identities,
the publisher copies the labels, cohort and alias manifests, verifies their joins
to the durable observations, publishes the complete supplement atomically and exits. This occurs
after label collection; it does not wait for neural evaluation. Immutable label
packets and copied provenance survive deletion of the experiment's disposable
cache. No additional MD or independent ancestry is generated.

`additional_labels/index.json` discovers supplements without modifying the
parent catalogue. Each publication atomically replaces the supplemental SQLite
index and manifest under a separate lock; open readers retain their snapshot.
Checksums cover the database, label packets and copied provenance. The regular
birth `refresh` workflow preserves the additional-label directory.

The initial attempt to copy windows during collection was stopped on the user's
request before a manifest or readable label release was published. Its copied
packets can be reused after checksum/content verification; its incomplete SQLite
staging files and stop receipt remain with the experiment's technical provenance.
