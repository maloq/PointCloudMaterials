# Original 0.75 ps Al trajectory retirement

On October 7, 2026, the user requested deletion of the original native 0.75 ps Al
dataset to simplify future temporal protocols. All 150 production trajectories
across the three collections below were retired. Exactly 1,050 manifest-listed
NPY arrays were deleted, releasing **99.624 GiB allocated** (94.524 GiB apparent).
This receipt covers local WORK/STORE storage; it does not claim deletion of
independently transferred copies on other servers.

| Collection | Trajectories | Allocated GiB released |
| --- | ---: | ---: |
| `al_meam_independent_sources_70304_400-500K_30perT_float16_20260902` | 90 | 56.708 |
| `al_meam_independent_sources_70304_510-520K_30perT_float16_20260903` | 36 | 22.683 |
| `al_520K_remaining24_20260913T204114Z` | 24 | 20.233 |

Removed arrays contain production positions, velocities, boxes, atom IDs/types
and timesteps. Each file was matched to its declared manifest, allowlisted
trajectory directory and recorded device/inode/size/modification time before
unlinking. All had a single hard link. The original manifests remain unchanged;
each retired trajectory directory now has a separate `retirement.json` notice.

Prepared liquid configurations, native melt restarts, input decks, logs,
single-frame melt validation, potentials and scientific results remain. The
initial array deletion preserved 900 files. At the user's subsequent request,
all **150 production `final.restart.bin` files were also deleted**, releasing
another **0.713 GiB**. The remaining 750 protected files were verified unchanged,
including all 150 `melt_final.restart.bin` files and 150 prepared liquids.
Total allocation released by these two removals is **100.337 GiB**.
Descendants keep their original
source IDs and split ancestry. Derived historical model inputs and analysis
artifacts were not included in this raw-trajectory removal.

The dependency audit checked all 971 source records in the consolidated birth
catalogue and 110 sources used by the active birth-history collection: neither
references the deleted trajectory directories. Prepared inputs for the original
40-branch V100 collection and its 33-branch birth-focused handoff passed their
recorded checksum checks. No Slurm jobs were changed. The 0.10/0.30-ps birth
datasets, dense descendants, shooting and million-atom trajectories remain.

All three catalogue entries now have state `payload_deleted`, classification
`retired` and role `provenance_only`. Their IDs remain available for provenance
and starting-state dependencies. A targeted live registry refresh confirms zero
usable production histories in these collections; the remaining 150 binary
records are **one-frame preparation QC only**. Unrelated inventory observations
retain their previous scan dates.

New birth experiments use separate exact 0.10- and 0.30-ps protocols. The old
Al64 source/split contract remains historical ancestry evidence; it is no longer
an available raw-history benchmark. The saved 0.75-ps audit still describes
historical results but cannot be rerun without the deleted raw inputs.

- [Deletion receipt](receipt.json)
- [Exact array inventory](files.json.gz) and [deletion log](deleted.jsonl)
- [Dependency plan](plan.json) and [protected-file inventory](protected.json.gz)
- [Post-removal checks](verification.json)
- [Targeted registry refresh](registry-refresh.json)
- [Production restart removal receipt](final-restarts-receipt.json)
- [Exact removed restart inventory](final-restarts-inventory.json.gz)
- [Production restart deletion log](final-restarts-deleted.jsonl)
