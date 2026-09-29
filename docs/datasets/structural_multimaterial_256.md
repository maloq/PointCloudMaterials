# Expanded native-unit structural pretraining

User-requested expansion, 25 September 2026. Preparation completed: 22,397 shards. Counts below describe the sealed release;
training consumption is recorded separately in each run.
Recipe: [multimaterial_256_20260925.json](../../configs/structural_pretraining/multimaterial_256_20260925.json).

The fixed Al64 crystallization benchmark, its source roles and all64/legacy16
rows remain unchanged. Running fits keep their recorded release. This is a
separate structural-pretraining release.

## Sampling and population

- Native Al training trajectories: **256 tracked centers**, including their
  original 64, sampled without outcome labels. All 90 train trajectories retain
  201 frames at 3 ps spacing: **4,631,040 neighborhoods**.
- External trajectories: **1% of atoms per frame**, rounded to the nearest
  integer, with deterministic tracked identities: 10,000 in a 1,000,000-atom
  cell, 10,486 in a 1,048,576-atom cell, 1,000 in a 100,000-atom Ti cell, and
  100,004 in a 10,000,422-atom Ta cell. Actual saved timestamps select frames at
  least 3 ps apart; timestamps never enter the model.
- Million-atom Al melt/measurement histories: **2,340,000 neighborhoods**. The
  shared physical boundary is retained only in the melt. Inspection found a
  1.33e-7 Å minimum-image RMS difference between the archived boundary frames;
  they are not bitwise identical. Both phases are one preparation lineage.
- Available Mg, Ta and Zr static configurations also use 1% of their total atom
  counts, sampled from centers at least 8 Å inside the coordinate bounds. These
  inputs have no invented periodic box or dynamics.
- Native selection keeps its original 64 centers: **192,960** current
  observations and **36,480** relaxed pairs. Calibration/test sources are
  excluded. No event, PTM, temperature or explicit time feature chooses centers
  or enters the model.

| Material | Dynamic training neighborhoods | Static training neighborhoods | Total |
| --- | ---: | ---: | ---: |
| Al | 7,537,284 | 0 | 7,537,284 |
| Mg | 566,244 | 62,916 | 629,160 |
| Ti | 728,000 | 0 | 728,000 |
| Ta | 4,592,340 | 510,260 | 5,102,600 |
| Zr | 0 | 61,440 | 61,440 |
| **Total** | **13,423,868** | **634,616** | **14,058,484** |

The 15 existing matched relaxed frames per native training source additionally
give **345,600 observed–relaxed pairs**, with observed neighbor identities
retained. Other materials have no claimed paired relaxed teacher. Physical
reconstruction uses the full mixed pool; paired VICReg/Epi use the expanded Al
pairs. These are different pretraining populations.

The older dynamic structural release supplies raw-source provenance only. Its
rescaled patches, temporal inputs and TDA labels are not reused. External Al EAM,
Mg, Ti and Ta branches are train-only and can share parents. Static generating
potentials remain unknown. Selection is Al-only, so this release does not
establish cross-material generalization. Shooting collections and precision
trajectories are not included in this first expansion.

## Input and model contract

Each patch contains the center and 79 nearest observed atoms, in native Å, with
an 8 Å support mask and no halo. **256 counts sampled centers, not atoms inside
a patch.** The encoder uses 5 Å edges and two spatial blocks.

The encoder is **geometry-only**, with one constant Al channel for every input,
width 128, export dimension 128, and **634,496 parameters**. There is no element
embedding, species one-hot input, material ID or scale feature in the encoder or
decoder. The briefly prepared five-species extension was removed at the user's
request before any training used it. Raw extraction continues unchanged: its
material and atomic-number records are audit metadata, not model tensors.

This follows the earlier
[`PretrainedMACEGeometryEncoder`](../../src/models/encoders/pretrained_mace.py)
approach: divide coordinates by a fixed source/material cutoff and multiply by
one common reference length. Here the existing training-calibrated scales from
the source catalog are expressed relative to Al, preserving native Al inputs
exactly: `x_model = x_A * scale_Al / scale_material`.

| Material | Fixed cutoff, Å | Coordinate multiplier |
| --- | ---: | ---: |
| Al | 9.1213891395 | 1.0000000000 |
| Mg | 10.1489976174 | 0.8987477861 |
| Ti | 9.3098646886 | 0.9797552859 |
| Ta | 9.3873371327 | 0.9716694959 |
| Zr | 10.3530089879 | 0.8810374984 |

These scales came from training-only nearest-160 calibration; the loader checks
their recorded sources against held-out ancestry. They are constant by material,
not refitted per neighborhood or test trajectory. Radius 8, cutoff 5, and
physical-pretraining targets operate on these Al-equivalent coordinates. Raw
Angstrom observations remain available in the cache. The sampled centers and
nearest-80 identities are unchanged. All four new initialization recipes use
the same original architecture. Some older structural MACE variants did include
species and scale inputs; their historical artifacts retain their actual inputs.

The physical targets remain 24 radial counts, two tapered counts and six angular
powers. The streaming trainer estimates target and initial-pool normalization
from 8,192 fixed training rows after coordinate normalization. Targets are computed per batch; the Epi reference
is frozen at initialization. Structural checkpoints are fixed epoch 12, with no
onset selector. Supervised continuation uses predictive likelihood; AP3/AP6 are
evaluation diagnostics.

## Execution and resumption

Cache: `${storage:scratch}/training-cache/structural-pretraining/multimaterial-256-20260925`.
Launch/logs: `${storage:analysis}/structural_pretraining/multimaterial-256-20260925/technical`.
Release identity:
`0d605bdbde89f27ee40135292633325a8ffa52ba3fede61542545a4d8b7441a3`.

```bash
python -m src.data.structural_pretraining.native plan --config configs/structural_pretraining/multimaterial_256_20260925.json
python -m src.data.structural_pretraining.native prepare --config configs/structural_pretraining/multimaterial_256_20260925.json --workers 4
```

Preparation is CPU-only and writes 22,397 resumable shards, including selection
and paired-only frames. Existing receipts are verified before reuse. All shards
must complete before the final manifest is published. Resume using the frozen
source in the launch receipt. No simulation, relaxation, quota gate or archive
operation is launched.

`NativeStructuralDataset` memory-maps at most 32 shards. It shuffles blocks of
16 shards and their rows, retains partial batches, and reproduces epoch/update
resumption. Every row appears once per epoch; this is not material-balanced
sampling. Large cells contribute proportionally more rows as requested. GPU
inputs and targets are batch-bounded, rather than loading the corpus into VRAM.

Prepared continuation recipes:
[`multimaterial256_20260925/campaign.json`](../../configs/encoder_context/multimaterial256_20260925/campaign.json).
They retain 12 pretraining epochs, 24 supervised/context epochs, batch/microbatch
256, one seed and online W&B. Vector messages and harmonic hierarchy both use
the new encoders and the fixed evaluation. These recipes require the completed
release; preparation does not relaunch the older campaign. See
[campaign commands](../encoder_context.md).
