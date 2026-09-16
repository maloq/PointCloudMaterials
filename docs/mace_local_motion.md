# Discarded consecutive local-motion map experiment

This frozen-feature map workflow was discarded on 16 September 2026. Its active
code and recipes were removed; see the [retirement record](discarded_frozen_encoder_maps.md)
for scope, exact source and reproduction details. Embedding forecasting and native
encoder training remain active.

All 44 fits and evaluation completed. No candidate passed the information gate
or the joint 0.10 jump requirement. See [results](../experiments/mace_local_motion_20260916/RESULTS.md),
[protocol and historical recipes](../experiments/mace_local_motion_20260916/README.md)
and [metric definitions](metrics/mace_local_motion.md).

The main output remains `output/mace_local_motion/sequences-20260916/`, linked to
`/store/PERSO/vmorozov/analysis/mace_local_motion/sequences-20260916/`.
The smoke run remains `output/mace_local_motion/smoke-20260916/`.
Checkpoints, histories, exact launch records, exported tables and IDS sequence
caches remain unchanged. The sequence cache is still used for labels/provenance
when preparing the separate native encoder data-amount study.
