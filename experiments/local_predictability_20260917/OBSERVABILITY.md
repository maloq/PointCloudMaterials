# Packet observability diagnostics

These ten fits complete the packet portion of the planned observability audit.
They ask whether the physical packet can describe crystal state and sustained
onset when the relevant observations are available. The raw-atom current-state
classifier is the separate follow-on diagnostic specified below.

| Task | Packet observations | Target | Population |
| --- | --- | --- | --- |
| Current state | Current frame | Current PTM crystal label | All states |
| Future state, 9 and 48 ps | True endpoint frame | Endpoint PTM crystal label | All states |
| Future sequence onset, 9 and 48 ps | Every true future frame through the endpoint plus two confirmation frames | First three-frame sustained onset in (anchor, endpoint] | Frozen primary at-risk set |

All-state recognition includes crystalline frames. The sequence includes the
two subsequent frames needed to confirm an onset starting exactly at the
endpoint. These future-input models are diagnostic oracles, never forecasts.
Endpoint state and first-onset targets are different tasks; their scores cannot
be compared as if they shared a target. Failed fits do not establish an
information-theoretic limit of the packet.

Use the already frozen 150 sources, 16 centers/source and 16 native anchors/center.
Preserve the 90/15/15/30 train/selection/calibration/test source folds. No outcome
filtering is applied to the all-state tasks; onset uses the same first-event and
three-negative-frame risk rule as the core study. Each observation also includes
the seven existing temperature/time condition features at the anchor.

Fit linear logistic and 256/128 SiLU MLP classifiers for each of the five cases.
The sequence is flattened in chronological order without summary pooling.
Use only seed 20260919 and the existing descriptor optimizer, regularization
grid and 2,000-update maximum. Standardization uses training rows; loss weights
give each source equal total weight. Choose regularization/checkpoints by
selection-source binary log likelihood only. A one-bin hazard is exactly
binary logistic likelihood: saved `event_bin=0` means positive, `1` negative,
and `probability[:,0]` is the positive probability.

Save checkpoints, split-tagged predictions, task metadata, source-release and
implementation hashes. Scientific analysis, calibration, uncertainty estimates
and metric CSV export are deferred until requested. This queue does not modify
the native training code or targets.

Reproduce with `python -m src.research.local_predictability.observability --config
configs/local_predictability/rtx6000_observability.json` using a new output and
an explicit valid deadline. See [execution record](../../docs/local_predictability_16h.md).

## Raw-atom current-state diagnostic

Train one snapshot NativeEncoder (width 16, two blocks, z128) from the fixed seed
20260919, with a linear binary classifier on z plus the seven known conditions.
Use positions and velocities at the current frame only. Train on all 23,040
training rows, including already crystalline environments, with uniform source
then uniform row sampling and effective batch eight. The budget is 8,192 AdamW
updates at learning rate 0.0003, weight decay 0.0001 and gradient clipping 5.
This equals the total parent-plus-child update count of each native onset model,
but is a separate objective and population, not a matched onset ablation.

Select by source-weighted binary NLL every 256 updates on 64 outcome-independent
windows from each of the 15 selection sources. Export predictions on all rows in
each fold, including separate calibration and test folds. Save exact-resume
optimizer, source sampler and RNG states. The binary encoding remains zero for
crystalline and one for noncrystalline. No future frame enters the observation.

This uses the existing 17 A native context; the physical packet summarizes 7 A.
Thus input extent and model capacity both differ from the packet classifier.
A failure by this particular encoder cannot establish that raw atomic input
contains no label information. Interpret the assay alongside the positive
PTM consistency check and the packet diagnostics.

## Frozen onset-state readouts

After all native continuations finish, freeze each selected checkpoint and export
its z128 on every eligible native row, including training rows. For snapshot,
history12 and repeat12 separately, train fresh linear and 256/128 SiLU MLP
six-bin hazard readouts on z plus the same seven conditions. Reuse seed 20260919,
source-equal likelihood, train-only normalization and the original descriptor
regularization grid/2,000-update maximum. Readout selection uses the full
selection fold; retain the native head's predictions on the exact same rows.

These six fits test accessibility of event information in the fixed state. They
do not add original history or claim state sufficiency. Interpret improvements
over the native head with the changed decoder fitting procedure in mind.
Keep checkpoints, row identities and predictions; defer scientific interpretation.
