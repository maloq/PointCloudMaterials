# CIV-MACE128 execution

**Complete as of 28 September:** all three 16-epoch fits and their associated
evaluations finished. [Results](../experiments/crystal_interface_20260928/RESULTS.md).

Use conda `pointnet-torch214` and the [interface recipe](../configs/crystal_vector/interface_al64_20260928.json).
This is separate from the running CDV-MACE128 crystal-set experiment.

Submitted 28 September 2026: CPU preparation array **1012053**, sealing **1012054**,
and one-GPU training array **1012055** (three treatments, at most two concurrent).
Continuation array **1012072** was accepted after completed preparation tasks
freed the per-user submission limit; no other experiment was canceled.
The local compiled batch-256 backward check passed with finite gradients through
the encoder and vector export. A periodic slab check verified positive interior
distances, layer zeros, periodic wrapping, tied-direction masking and empty-cell
censoring. Real-source checks covered one training and one held-out source; all
original rows/geometry were preserved and uniform held-out interiors were present.

```bash
python -m src.research.crystal_vector.interface_queue submit \
  --config configs/crystal_vector/interface_al64_20260928.json
```

Submission freezes source/config/metric contracts, prepares interface labels in a
CPU array, seals all 150 sources, then starts the three treatments with one GPU
each and at most two concurrent GPUs. Jobs use Slurm and remain detached. Training
has an eight-hour allocation and one dependent continuation for a clean deadline
checkpoint. Failures retain a traceback and do not silently retry. Online W&B is
mandatory; local evaluations update the existing training run.

Results: `${storage:analysis}/crystal_interface/al64-random-20260928`.
Cache: `${storage:cache}/crystal-interface/al64-covering-20260928`.
Each variant exports `analyses/localization-v1/tables/`, including separate phase
tables for interior, layer and exterior. `technical/launch.json` records job IDs;
`technical/variant-N.json` records training/evaluation status. Dataset `state.json`
records population counts and expected random-batch coverage.

Original train/selection coordinate banks are linked to the sealed CDV dataset;
held-out banks add uniformly sampled contexts to support interior evaluation.
Do not remove the parent dataset while this dependent collection is in use.
All caches remain outside the repository. Source identities, original query
geometry, phase reconstruction, parent checksums and interface constraints are
validated by preparation; no new MD simulation is needed.

The [scientific protocol](../experiments/crystal_interface_20260928/README.md) and
[metric contract](metrics/crystal_interface.md) define the atom-layer target.
