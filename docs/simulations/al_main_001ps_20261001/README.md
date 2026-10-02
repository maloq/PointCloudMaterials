# Two main Al descendants saved every 0.01 ps — 1 October 2026

**Latest queued protocol:** both additions were still unstarted and move to
[confirmed-halfway stopping with peer caps](../al_main_half_stop_20261001/README.md).
They retain exact 0.01 ps observations, velocities, ancestry and priority.
The original fixed 600-ps preparation and its estimates are preserved below.

User-authorized additional sampling for two existing main-Al ancestors. Each run
has **70,304 atoms**, **15 ps equilibration**, **600 ps measurement**, and saves
positions and velocities every **0.01 ps**: **60,001 measurement frames** including
both endpoints. LAMMPS still integrates with **2 fs** steps; output occurs every
five steps. This is a sampling change, with the retained Lee2003 Al MEAM potential,
zero-pressure NPT, 0.3 ps thermostat, 3 ps barostat and 0.3 ps drift removal.

| Source ID | Temperature | Frozen role | Ancestry |
| --- | ---: | --- | --- |
| 886 | 400 K | train | independent_melt_235751553 |
| 1004 | 520 K | train | independent_melt_76354933 |

Selection uses the first unassigned train source at the lower and upper campaign
temperatures in source-ID order, without inspecting crystallization outcomes.
Each uses the same native prepared liquid and velocity seed as its planned
0.1-ps descendant. They represent **two existing melt ancestors**, retaining their
frozen roles; the combined queue still has **150 independent ancestors**, with
152 daughter trajectories. Future labels must be measured on the new trajectories.
The 0.1-ps sources, historical trajectories, and frozen scientific manifests remain.

## Execution

[Recipe](../../../configs/simulation/al_main_001ps_20261001.json) ·
[Queue producer](../../../src/simulation/campaigns/dense_al_additions.py) ·
[Durable execution status](/store/PERSO/vmorozov/simulation-launches/al-main-010ps-plus001ps-20261001/status.json)

Original replacement controller **1017489** was queued after active worker array **1017389**.
It replaces only pending controller 1017390 and gives these two runs priority in
the next wave. [Submission](submission.json), [prepared inputs](preparation.json),
[startup observation](startup.json) and [receipt hashes](provenance.json) are copied
locally. The original 150 records are exactly unchanged in the new execution manifest.
The [metadata correction](metadata_correction.json) scopes `trajectory_roles` in
the separate two-source collection to its two train trajectories. The original
collection view is retained, and the queued 152-record input manifest is unchanged.

```bash
python -m src.simulation.campaigns.dense_al add-dense-sources \
  --config configs/simulation/al_main_001ps_20261001.json
```

Run once: preparation creates and freezes a new execution manifest containing the
150 main records followed by the two additional records, and references the prior
manifest SHA256. Only this campaign's pending next-wave controller is replaced;
the three running workers retain their original code, inputs and allocation.
The new controller waits for the active wave, checks its publication, then gives
the additional sources priority in the next wave. A wave containing 0.01-ps sources
assigns only one source per worker, with 48 MPI ranks, 64 GiB RAM and 48 hours.
Subsequent ordinary waves return to two sources per worker. Any failure is
checksum-published with native restart state and stops the queue for inspection.

SCRATCH sources are under:
`/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/al-main-010ps-20260927/al-main-001ps-20261001/runs/`.
Verified completed sources publish to
`/store/PERSO/vmorozov/simulations/al-main-001ps-20261001-source0886/` and
`/store/PERSO/vmorozov/simulations/al-main-001ps-20261001-source1004/`.
The separate collection `al-main-001ps-20261001` is registered in the dataset catalog.

## Precision, conversion and storage

Canonical `trajectory_binary_float16/` contains positions and velocities in
**float16**, boxes in **float32**, IDs/types as exact integers, and exact integer
timesteps. Estimated vector payload: **50.62 GB per run**, **101.24 GB total**,
plus a few MB of timelines, identities and native restart/preparation files.
The existing 150-source 0.1-ps target remains separate. Native integration and
restart precision remain unchanged.

```bash
python scripts/convert_trajectory.py dense-al-001ps RUN_DIR --delete-source
```

The 60,001-frame source exceeds the RAM budget of the historical all-frame parser.
The [streaming converter](../../../src/data/conversion/streaming_pair.py) uses the
established float32 consumer parser as a generator and verifies the entire timeline,
IDs, types and periodic box contract. It writes NumPy arrays in 32-frame chunks,
reads each chunk back, checks every float16 rounding value, and records coordinate
and velocity maximum/RMS errors against the float32 consumer. Persisted array-byte
SHA256 values are verified with bounded reads. Float32 semantic hashes are retained;
there is no full float32 trajectory copy. The source ASCII dump is deleted only
after the canonical consumer loads successfully and conversion provenance,
quantization errors and checksums have been saved.

At this cadence, transient ASCII output can reach roughly **0.5 TB per run**.
Preparation requires at least 1.5 TB free in SCRATCH and STORE; final canonical
storage is much smaller. Conversion remains local scientific provenance and does
not create a W&B run.

[Conversion validation](conversion_validation.json): a 108-atom infrastructure
fixture produced the full 60,001-frame native timeline. Every exported position
and velocity equals the maintained float32 converter rounded to float16; IDs,
types, timestamps and boxes match exactly, and consumer checksums pass. Endpoint
and restart-boundary parsing also matches the previous frozen parser exactly.
This fixture is excluded from scientific sources; its input and diagnostics are
kept in the campaign's `technical/dense001-conversion-check/` directory.
