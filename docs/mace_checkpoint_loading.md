# MACE checkpoint loading on a non-default GPU

During the September 14 single-frame ridge audit on node57, the process had two
L40S GPUs visible and requested `cuda:1` while the current device was `cuda:0`.
The accelerated MACE constructor allocates on the current CUDA device. The old
checkpoint helper constructed that model, loaded a CUDA:1 state dictionary into
it, and finally moved it to CUDA:1. Exact tensor comparisons detected corrupted
backbone buffers and parameters before any prediction was accepted. A CPU-state
reload after construction restored those tensors exactly.

`src/utils/model_utils.py` now constructs under the requested CUDA device context
and loads saved tensors through CPU before the final device placement. A regression
test with GPU-allocated constructor buffers plus CPU head parameters verifies exact
state on a non-default GPU and restores the caller's prior current device.

cuEquivariance kernel/autotuning calls also require the current device to match
the tensors. The audit explicitly sets its current device before construction,
forward passes and backward checks. Its first attempt without that setting passed
state checks after the loader fix, but failed inside the CUDA kernel before inference
results were produced. Both failed-attempt logs are retained in the audit output.

These failures were encountered while setting up the new audit. After the fix,
all six models passed exact state checks, and the historical ridge scores were
reproduced within 1.6e-6 balanced MSE under matching precision. The device issue
therefore does not explain the near equality of the historical VICReg/TDA scores.

Reproduction and scientific results are in the
[ridge audit](../experiments/mace_tda_ridge_audit_20260914/README.md).
