# Ta nucleus movies and local trajectories

## Separate 12 ps appearance windows

The requested12ps before and12ps after appearance is exported separately under
`${storage:analysis}/nucleus_crops/ta-appearance-20A-12ps-20261003`.
It contains the eight ambiguous Ta appearance candidates, one directory each,
and a combined `ta_appearance_20A_12ps.zip` for transfer. The20Å radius and0.1ps
observation spacing are unchanged. Earlier18ps appearance/disappearance exports
remain available independently.

The existing24ps branches do not provide full±12ps coverage for any of these
eight event times. Each movie contains the intersection of the requested window
with available observations. Its metadata and index record missing prehistory or
follow-up. The anchor is the start of the confirmed64-atom establishment streak;
these events remain ambiguous candidates, not confirmed independent nuclei.

```bash
python -m src.research.crystallization_origin.event_crop_windows \
  --config configs/analysis/ta_appearance_crops_12ps_20261003.json
```

This operation slices verified existing local files without reprocessing full
cells or relabeling events. Every resulting XYZ frame is read back and compared
with its NPZ counterpart. [Window definitions](metrics/ta_event_crop_windows.md).

Completed: eight crops and1418 independently validated movie frames. Seven
contain12ps of prehistory and2–8.5ps of follow-up; the remaining candidate has
11.5ps before and12ps after. All missing durations remain explicit. The ZIP
contains eight XYZ movies and eight NPZ arrays, each with metadata and checksums.

## Earlier 18 ps appearance and disappearance windows

Export the eight ambiguous appearance candidates and28 verified dissolution
episodes from the24 Ta shooting branches. Keep a20 Angstrom radius around each
event at native0.1ps spacing. Each event gets its own folder with
`trajectory.xyz.gz`, `trajectory.npz`, `metadata.json` and a completion receipt.
Appearance and disappearance folders are separate. Original labels are retained,
including unreliable lineage and interface-associated origins among dissolutions.

The requested18ps before and18ps after spans36ps, but the source branches last
only24ps. Export only available observations and record exact missing durations.
No new MD or invented prehistory. Appearance anchors use establishment onset;
disappearance anchors use the first verified post-cluster audit observation.

Open a `trajectory.xyz.gz` directly in OVITO. All coordinates are unwrapped around
the local moving reference, and crop boundaries are nonperiodic. Native atom IDs
allow tracking as atoms enter or leave the sphere. `ptm_type` carries full-cell
labels every0.5ps; −1 at intermediate frames means unavailable. The
`tracked_center` field marks the reference atom. The NPZ contains the same
centered geometry, frame offsets, original boxes and absolute centers.

```bash
python -m src.research.crystallization_origin.event_crops submit \
  --config configs/analysis/ta_nucleus_crops_20261003.json
```

Use conda `pointnet-torch214`. Four CPU lanes export sources with bounded memory;
a dependent collector validates completion and publishes the index. Frozen code,
configuration and Slurm receipts live under `technical/`. Completed event exports
are checksum-verified and reusable on resume. No GPU or online training log.

Output: `${storage:analysis}/nucleus_crops/ta-20A-18ps-20261003`.
Coverage: `analyses/coverage-v1/tables/crop-index.csv`.
[Exact definitions](metrics/ta_event_crops.md).

## Completed October 3 export

The36 crops are complete: eight ambiguous appearances and28 verified
disappearances, from16 of the new shooting sources. All8251 exported frames
passed independent XYZ read-back checks. Saved time spans range from18.5 to24ps;
none can provide the full requested36ps. The event files occupy approximately
275MB including both array and movie representations. Slurm worker array1019887
and collector1019888 recorded execution under the output's `technical/` folder.
