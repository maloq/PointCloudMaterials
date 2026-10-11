# Four saved Ni birth examples

Current code and commands: [shared post-simulation module](postprocessing.md).
Dated execution receipts below retain their recorded implementations.

The October 7 request is four formation visualizations, without building a Ni
catalogue or generating more MD. The analysis selects one confirmed birth from
each completed independent 1100 K source `ni-T1100-melt00`, `melt01`, `melt02`,
and extension `melt04`. Source roles, ancestry, original potential, array hashes,
and exact saved timelines remain in `technical/search-plan.json`.

Configuration: [ni_four_births_20261007.json](../configs/data/ni_four_births_20261007.json).
Implementation: `src.research.crystallization_origin.ni_birth_histories`.
Scientific definitions: [Ni birth histories](metrics/ni_birth_histories.md).
Simulation provenance: [Ni campaign](simulations/ni_births_010_20261007.md).

The full periodic cell is classified at every native 0.1 ps frame from the
validated liquid start. A bounded prefix ends 20 ps after the coarse monitor
first crosses 128 atoms. That monitor selects the search bound only. Exact
births come from the unchanged Ni descendant-family tracker: connected PTM
FCC/HCP/BCC atoms within 3.2 Å, an eight-atom embryo, then a connected descendant
of at least 64 atoms maintained for 1.5 ps. Each example is the earliest confirmed
liquid-origin root in its source. This deliberate selection cannot estimate a
population nucleation rate. Time zero is the saved liquid start; cooling ends
at 20 ps.

## Outputs

The permanent run is `${storage:training_storage}/nucleus_histories/ni-four-births-20261007-v1`.
The repository alias is `output/nucleus_histories/ni-four-births-20261007-v1`.

- `index.html`: four examples and their links.
- `analyses/births/<id>/viewer.html`: rotating atoms and a native-frame slider.
- `plots/properties.png` and `.pdf`: six panels of size, crystal types, bond
  order, shape, membership, crystallinity, and spacing.
- `plots/precursor-graph.png`: retrospective shared-atom precursor branches.
- `plots/full-history.png`: family sizes over the searched prefix.
- `plots/early-appearance.gif` and `plots/establishment.gif`: two movies per birth.
- `movie.npz`, `movie-frames.json`, `birth.json`, `graph.json`: underlying data.
- `tables/`: exported measurements with pinned metric definitions and hashes.
- `technical/`: input identity, launch ownership, progress, and tracker outputs.

The atom movie is continuous from 12 ps before embryo appearance through 2 ps
after confirmation. It omits no intermediate native frames. GIFs use a declared
0.2 ps playback stride; the full viewer keeps 0.1 ps. The 14 Å crop follows a
founder atom. Shapes and visible membership refer to that crop; tracked family
size refers to the original full-cell connected component. Graph gaps mean
uncertain identity connections, not missing source coordinates.

The desktop export provides an offline ZIP of all four examples. Extract it and
open `index.html` in a browser. The in-chat establishment view retains every
crystalline atom while sampling liquid context; the complete crop is available
in the offline viewer. Display rounding does not alter scientific measurements.

## Reproduction and execution

Use conda `pointnet-torch214`. The initial launch uses one CPU thread per worker.
Initial preparation and launch (launch refuses a duplicate receipt):

```bash
python -m src.research.crystallization_origin.ni_birth_histories prepare --config configs/data/ni_four_births_20261007.json
python -m src.research.crystallization_origin.ni_birth_histories launch --config configs/data/ni_four_births_20261007.json
```

The launch freezes implementation and metric definitions. Four CPU workers
classify the four prefixes independently, then track births and render plots.
Current-chat job ownership is recorded in `technical/launch.json`. The separate
Ni MD campaign and its population-search jobs are not controlled by this run.
This analysis does not publish new catalogue entries or change prediction labels.

The four-example job was resumed as `1027681` with four OVITO threads per source
and 16 allocated CPUs. BLAS, OpenMP, and freud remain single-threaded. The frozen
scientific implementation and verified classification chunks are reused.
`technical/thread-audit.json` records exact agreement with the initial classifier
on all 32,000 labels of a saved frame; `cpu-resume-plan.json` and
`cpu-resume-launch.json` preserve the original worker records and ownership.

Once all four examples are complete, `src.research.crystallization_origin.ni_birth_export`
copies their scientific packets and frozen tables into an offline gallery, embeds
the full viewer data without browser decompression requirements, and prepares the
small in-chat excerpt. It accepts `--root`, `--inline-template`, and
`--inline-output`; the template is the existing in-chat Al atom viewer. Publication
records file checksums in `analyses/desktop-export/manifest.json`. It does not
recalculate labels or measurements.

## Completed examples

All four packets passed the final checksum, cadence, and persistence audit in
`technical/delivery-audit.json`. The 718 detailed movie frames are contiguous
within their four respective source intervals. Each confirmation has 16 native
observations spanning 1.5 ps, all above the 64-atom threshold.

| Independent source | Tracked embryo onset (ps) | Confirmation (ps) | Minimum atoms during confirmation interval |
| --- | ---: | ---: | ---: |
| ni-T1100-melt00 | 62.7 | 64.4 | 73 |
| ni-T1100-melt01 | 44.2 | 53.6 | 65 |
| ni-T1100-melt02 | 81.5 | 83.8 | 66 |
| ni-T1100-melt04 | 74.6 | 76.6 | 115 |

The tracked onset is a family-identity label; nearby order and earlier crystalline
atoms can precede it. Their positions remain visible in the movies. Weak or
absent precursor links do not prove an absence of prior structure.

The offline ZIP contains all four complete viewers, eight GIFs, scientific
packets, plots, and exported tables. The in-chat view contains 51 consecutive
establishment frames per example (204 total), every crystalline atom within the
crop, and at most 48 sampled liquid-context atoms per frame. The completed
fragment is 662,773 bytes. The standalone browser check caught and corrected a
presentation-only rename of `catalogued_family`; original generated pages are
preserved in `analyses/desktop-export/source-viewers/`. No numerical packets or
frozen metric definitions were changed. The corrected offline and in-chat views
were rendered in headless Chromium and visually inspected.

## Extended history and clearer surrounding atoms

The `history-context` publication profile retains all 718 saved frames in the
in-chat view: 50.7–66.4, 32.2–55.6, 69.5–85.8, and 62.6–78.6 ps for melts 00,
01, 02, and 04 respectively. Each begins 12 ps before its tracked embryo onset,
with the native 0.1 ps spacing. It reuses the completed packets without running
MD or changing classifications, measurements, or birth definitions.

Grey atoms are larger and have adjustable opacity, initially 60%. Their legend
says **Other / unclassified**: PTM does not establish that every such atom is
liquid. The compact in-chat view keeps every crystalline and linked atom, every
atom inside the visible linked atoms' convex envelope, and a stable atom-ID
sample of 12% of the remaining context. The envelope controls display sampling
only; it is not an inferred crystalline phase boundary. Display coordinates use
a 0.02 Å grid, leaving scientific coordinates and measurements unchanged. The
offline viewer retains every atom in the 14 Å crop and has the same visibility
control. Its copied historical GIFs remain unchanged.

```bash
python -m src.postprocessing.visualization.export \
  --root output/nucleus_histories/ni-four-births-20261007-v1 \
  --profile history-context --inline-output /absolute/visualization/ni-nucleus-history.html
```

The updated offline gallery, ZIP, and provenance manifest are under
`analyses/history-context/`. The earlier `analyses/desktop-export/` publication
and original scientific viewers remain preserved.
`browser-audit.json` records successful four-case decoding, native frame counts,
playback, opacity changes, and the narrow-screen layout; preview PNGs accompany
it. The updated fragment is 812,072 bytes. At melt00, 65.9 ps, it shows 185 grey
atoms instead of the earlier 48 while preserving all 118 visible crystalline
atoms.
