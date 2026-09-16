# Discarded frozen local-state smoothness sweep

This frozen-feature map workflow was discarded on 16 September 2026. Its active
code and recipes were removed; see the [retirement record](discarded_frozen_encoder_maps.md)
for scope, exact source and reproduction details. Embedding forecasting and native
encoder training remain active.

The first run is `output/mace_local_smooth/velocity-frozen-20260915/`, linked to
WORK analysis. Its 32 fits and evaluation completed; no candidate passed the
information gate. See [findings](../experiments/mace_local_smooth_20260915/RESULTS.md).

The capacity follow-up is `output/mace_local_smooth/velocity-frozen-capacity-20260915/`.
Its 22 fits are saved; final evaluation completion was not established. This is no
longer an active submission. Exact submission commands and batch scripts remain
under each run's `technical/` directory.

Both outputs, IDS feature caches, checkpoints, optimizer states and immutable
provenance remain in place. Historical recipes are retained with the
[scientific record](../experiments/mace_local_smooth_20260915/README.md).
