# CPU integration example

This is a software integration fixture, not a crystallization experiment.
Independent synthetic point clouds pass through the native Q/V converter and
binary reader. The MLP predicts each cloud's mean squared distance from its
centroid. Coordinates are centered per cloud and divided by one scalar fitted
on training coordinates only. There are only train and test rows.

`test_mse` is the arithmetic mean of squared prediction errors over the held-out
rows, computed after the configured fixed update budget. No test-based selection,
calibration or early stopping occurs. `training_updates`, `train_rows` and
`test_rows` are execution metadata. All computation is CPU float32 after reading
the converter's float16 stored observations.

The example compares uninterrupted training with checkpointed/resumed training,
including parameters, optimizer states, random states, batches and losses.
Exact agreement is required within the same environment. Scores are not expected
to be bit-identical across PyTorch versions or hardware.
