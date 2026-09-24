# Running the encoder parameter search

The [scientific protocol](../experiments/encoder_parameters_20260923/README.md)
declares28fits and fixed budgets. Use conda `pointnet-torch214` on the allocated
two-GPU node. Source checkpoints and analysis from other running queues are
preserved. Preparation copies the existing fixed numerical and8×spatial input
caches once; raw inputs and dataset ancestry remain unchanged.

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True
python -m src.research.encoder_parameter_search.preflight --config configs/encoder_parameter_search/campaign.json
python -m src.research.encoder_parameter_search.queue launch --config configs/encoder_parameter_search/campaign.json
```

Preflight is separate: tests, exact within-seed initial encoder identity, one
disposable complete pass for both manual/automatic GeoFrame optimizers, and three
production-size updates of each MACE arm. These diagnostics are not scientific
fits. Launch checks source/config hashes, freezes code and starts two detached
advisory-lock workers inside the current Slurm allocation. Training, inference,
probe fitting and plots run in separate child processes. Independent task failures
are recorded and do not silently change settings or block other treatments.

Results: `output/encoder_research/parameter-search-20260923/index.html`.
`technical/launch.json` records commands, PIDs, GPU assignments and deadline.
`technical/tasks/NAME/` contains stage logs and completion/failure receipts.
GeoFrame checkpoints: `technical/fits/NAME/technical/training/`.
Native MACE checkpoints: `technical/mace-sSEED/technical/fits/ARM/`.
`technical/evaluations/NAME-MILESTONE/` retains native embeddings and original
metrics; `technical/supplements/` retains the additional liquid/interface assay.
Dense final panels preserve the requested8×sample count and half-diameter markers.

Workers stop near allocation expiry. GeoFrame saves complete-pass periodic
checkpoints; MACE saves optimizer/sampler/RNG state. To continue, use the recorded
frozen `worker --config technical/campaign.json --lane NAME` command under a new
allocation, one process per GPU. Locks prevent duplicate fits. Failed tasks need
diagnosis and receipt archival before an intentional retry. Do not mutate frozen
code or launch a second copy through the active-config launch command.

For a CPU report refresh, use the frozen source:
`python -m src.research.encoder_parameter_search.report --config ABSOLUTE-RUN/technical/campaign.json`.
The earlier native snapshot queue and its historical metric files remain intact.

## Launch,23September2026

Running inside allocation1005857 on node61, two RTX PRO6000 GPUs. Detached
worker PIDs971485/971486 and immutable commands are in the launch receipt.
The first paired GeoFrame control fits are advancing through complete passes.
22 tests and three disposable full-pass trainer checks passed; the native MACE
initial encoder tensors also match exactly across its four treatments (fitted
normalization buffers retain native CUDA reduction noise).
