# Transfer from established births to failed embryos

The target is now **any local crystal appearance**. Failed clusters count as
positive, as explicitly requested. The 184 failed candidates include the 28
strong candidates; the remaining disjoint group has 156 episodes. No duplicate
events enter training. Original train/source roles and merged held-out roles stay
fixed, with unchanged old histories and uniformly checked new input eligibility.

Run these requested comparisons with matched liquid controls:

1. Frozen original-birth predictor → held-out strong failed embryos.
2. Frozen original-birth predictor → all held-out failed embryos, also separating
   the 156 weaker candidates from the strong subset.
3. Combined-pool predictor → the same held-out strong failed embryos.
4. Combined-pool predictor → combined held-out established/failed appearances.

Also report established-only performance and original-model combined-test
performance to expose tradeoffs. Training-source scores remain separate.

Use logistic and gradient-boosted readouts of the existing 442 physical
descriptors, current/eight-frame histories, and original/full-cell-relaxed inputs.
Reuse eight existing models and fit eight combined counterparts. Single seed;
five training-source folds select by NLL. AP is diagnostic. No encoder fitting,
new MD, temperature/time input, source resplit or calibration mapping.

[Definitions](../../docs/metrics/birth_appearance.md) ·
[Configuration](../../configs/birth_prediction/appearance_transfer_20261002.json) ·
[Detached execution](../../docs/birth_appearance.md).

```bash
python -m src.research.birth_prediction.appearance submit \
  --config configs/birth_prediction/appearance_transfer_20261002.json
```

Output: `${storage:training_storage}/birth_prediction/appearance-transfer-20261002/`.

Input eligibility retained 133 failed events (18 strong): 76 train-source events
(12 strong) and 57 merged-test events (six strong). Combined sample support is
2,245 training rows and 1,425 held-out rows, each retaining the 1:4 case/control
ratio. All 184 candidates remain in the eligibility audit, with explicit reasons
for the 51 excluded events. The strong held-out result therefore concerns six
events across five sources, not all 28 catalogue candidates.
