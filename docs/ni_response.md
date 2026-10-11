# Ni response workflow

`src.research.ni_response.workflow` owns preparation, GPU response banks,
1600-update fits, resumable workers and dependent CPU frozen-probe evaluation.
The recipe is `configs/ni_response/esen_unrelaxed_20261009.json`.

Sixty completed native0.10ps Ni sources retain original ancestry and roles.
Catalogues: ni-20261007-v1, ni-20261007-extension01-v1 and
ni-lowT-20261008-gpu-full-v1. Existing records contain117 persistent32-atom and
70 persistent64-atom liquid-origin roots before lead-availability checks.
The completed unrelaxed checkpoint hash begins4e4439d46a97; the ongoing relaxed
training is a different experiment and is not used here.

Physical response banks are written to SCRATCH then checksummed into STORE.
Observation/target caches use IDS. GPU workers share one6-entry disposable
encoder feature pool with process leases; checkpoints and metrics are preserved.
The two requested GPU workers each accept RTX6000PRO/H100/A100/L40S/A40 as
alternatives. Only this campaign's receipt-bound jobs may requeue themselves.
Training uses online W&B. A CPU dependency runs frozen prediction diagnostics.
The old Al workflow and all historical definitions are unchanged.

[Scientific protocol](../experiments/ni_response_20261009/README.md) and
[exact metric definitions](metrics/ni_esen_response.md).

October9 launch: CPU preparation job1030776 completed; GPU workers1030777
(node32/A40) and1030778(node37/A40) are generating the physical response bank.
CPU frozen-probe job1030779 depends on both workers. Each GPU worker has24h
and requeues its own receipt-bound job if continuation is required. The
pre-existing relaxed spatial fit on node58 remains separate.

Prepared counts:5120 motion-fit rows,1280 training-ancestor selection rows,
and12800 pooled held-out rows. Appearance eligibility retains3825 original
training and8027 held-out observations. At each1/3/6ps nucleus lead, the32-atom
assay has37 training and80 held-out root events;64 has21 training and49 held-out.
An event has four founder observations and sixteen matched controls; those rows
are not independent nuclei. All declared leads were available for these events.

The local FP32 eSEN velocity secant/JVP discrepancy was0.181% on the audited
Ni inputs; finite full-batch gradients passed. Two complete256-row derivative
training backward passes took7.52/7.36s on a shared RTX PRO6000, without optimizer
updates or W&B runs. These timings do not establish the A40 training ETA.
