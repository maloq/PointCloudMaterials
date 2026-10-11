# Local80 finite-displacement forward campaign and matched fits

The [scientific protocol](../../experiments/local_response_20261003/README.md)
reuses the completed local80 bank with hashes verified before every stage.
New simulations use only four declared train parents from that collection.
Al parent physics is Lee2003 MEAM; the new forward teacher remains
MACE-MPA-0 medium (SHA25675428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638).
No new independent ancestry or downstream crystallization labels are claimed.

Use conda `pointnet-torch214` and the established trusted-checkpoint setting:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m src.research.local_response_followup.queue prepare --config configs/simulation/local_response_followup_20261003.json
python -m src.research.local_response_followup.queue launch --config configs/simulation/local_response_followup_20261003.json
```

Launch requires one allocated idle GPU with at least five hours remaining.
It freezes source/metric definitions, starts scientific training detached in that
allocation, and submits a separate one-GPU forward worker on L40S/RTX6000PRO.
The36Å AD reference needs most of an L40S, so it must not share the training GPU.
The forward job requests eight hours and can requeue at allocation expiry at
most twice. Other errors halt with stage receipts; no silent numerical fallback.
Training saves model/optimizer/RNG every update and can resume from the frozen
bundle with `train --config config.json` in a later GPU allocation if needed.
Both workers hold stage locks. Training is online in teshbek/PointCloudMaterials;
simulation/numerical checks remain local and final evaluation updates existing
training run summaries through their recorded IDs.

Locations use machine.local.yaml aliases:

- Output: `${storage:training_storage}/response_atlas/local80-followup-20261003`.
- New simulation work: `${storage:simulation_runs}/local80-finite-response-20261003`.
- Verified archive: `${storage:archive}/simulations/local80-finite-response-20261003`.
- Original training bank: the unchanged cache recorded by the reference run.

Each branch saves features and scalar physical diagnostics to SCRATCH, then
copies and checksum-verifies STORE before writing a completion receipt.
No trajectory-position arrays are written; input parents retain their original
provenance and precision. Interrupted completed branches are already durable.
Execution logs, launch IDs and progress live in `technical/`; comparison and
forward figures/tables live in `analyses/comparison-v1` and `analyses/forward-v1`.
Use the frozen `summarize` stage for a partial forward report or `evaluate` after
all15 fits. Every CSV carries its frozen metric definitions and producer hashes.
