# MACE context pilot

Matched 5,760 anchors: 18 training, 6 validation and 6 test simulations. One encoder initialization; exploratory cohort already used in earlier diagnostics.

| Method | Hot TDA MSE | Relaxed TDA MSE | Boundary / 0.75 ps energy |
|---|---:|---:|---:|
| frozen-mean80 | 0.017148 | 0.035753 | 0.0367 |
| frozen-halo_mean80 | 0.042502 | 0.031095 | 0.00645 |
| frozen-halo_inner | 0.041357 | 0.030925 | 1.43e-09 |
| frozen-halo_center | 0.077556 | 0.049863 | 1.49e-10 |
| trained-mean80 | 0.017150 | 0.035616 | 0.0367 |
| trained-halo_inner | 0.041476 | 0.030984 | 1.41e-09 |
| trained-halo_center | 0.078912 | 0.046926 | 5.4e-10 |

![Context comparison](plots/context-comparison.png)

Readouts are float64 SVD ridge regressions, with alpha chosen on validation sources only. TDA labels still describe the original hard 80-atom support. A smooth embedding need not reproduce every discontinuity of that label definition.

Training is a matched eight-epoch warm-start pilot at learning rate 1e-4 and batch 256, using the original spatial/temporal VICReg objective, projector, mirror and jitter settings. Encoder gradients are replayed in graph microbatches; the complete batch enters VICReg and projector BatchNorm. Validation VICReg loss selects the checkpoint. No TDA labels train the encoder.

The halo uses complete native two-hop 5 A neighborhoods. Inner pooling has unit weight to 5 A and a quintic taper to zero at 7 A. The center variant returns the tracked center node. The 18 A candidate crop includes a checked augmentation margin. Feature width remains 256.

Training continuation and feature extraction status are retained separately under technical/. No forecasting model has been retrained in this pilot.
