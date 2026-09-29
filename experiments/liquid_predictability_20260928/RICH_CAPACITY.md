# Rich-descriptor capacity and duration experiment

Question: does limited capacity or short optimization explain weak recovery of
known geometric information by the current MACE training setup?

RD-MACE256-L3-Z256 is trained from scratch on the complete existing raw cohort,
using the same 3,536 descriptor targets. The changes are width 128→256, spatial
depth 2→3, exported state 128→256, and duration to 60 exact full data passes.
The vector-context module grows to width 256; the final decoder hidden width is
512. The user additionally requested the largest practical batch and maximum
learning rate 0.004 with warmup/cosine. This combined intervention cannot isolate
the independent contribution of width, depth, batch size, LR or duration.

Primary endpoint: held-out equal-family standardized feature MSE. Publish every
feature's native-unit RMSE and R² alongside the training-mean baseline and all
four family summaries. Report both validation-selected and final training epoch
identities. No crystal-distance labels enter the optimization or selector.
Strong reconstruction would establish descriptor recoverability in this model,
not necessarily useful distance-to-crystal information. Weak reconstruction
remains ambiguous between representation, optimization and target conditioning.

The earlier full-raw MACE128 fit is the relevant cohort comparison. The smaller
paired raw/relaxed fits have a different training population. Changes in data
visits and optimizer updates must remain explicit. One seed; no seed-uncertainty
claim. Source roles and evaluation rows never change.

[Recipe](../../configs/liquid_predictability/rich_mace256_20260929.json) ·
[Metrics](../../docs/metrics/rich_descriptor_encoder.md) ·
[Execution](../../docs/rich_descriptor_encoder.md).
