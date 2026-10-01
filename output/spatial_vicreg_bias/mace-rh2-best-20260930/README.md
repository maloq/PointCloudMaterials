# MACE RH2: selected checkpoint comparison

Frozen best checkpoint of the latest normalized residual-head run: **epoch 14,
update 1806**, selected by Al validation descriptor Gaussian NLL
**1.0348901381502593**. The immutable checkpoint and full selector record are
in `technical/checkpoint-selection.json`.

- [Held-out Al MD comparison](heldout-al.html)
- [Six static Al snapshots, 166–240 ps](al-static.html)

Both compare the new neural embeddings with unchanged all-training TDA/bond-order/CNA
clusters and descriptor PaCMAP. Dense MD, samples, PTM overlays, color correspondence
and both travel modes use the new checkpoint. The interface toggle only changes
opacity. No neural training or descriptor recomputation is launched.

[Held-out metrics](analyses/matched/tables/metrics.csv) · [definitions](analyses/matched/tables/METRICS.md)

[Static metrics](analyses/static/tables/metrics.csv) · [definitions](analyses/static/tables/METRICS.md)

Recipe: `configs/analysis/mace_rich_current.json` in the repository.
This replaces the previous step-1088 MACE analysis; its copied checkpoint/data
are removed after verification, recorded in `technical/previous-analysis-removal.json`.
The original scientific training artifacts are preserved separately.
