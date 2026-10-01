# General neural / descriptor comparison

Descriptor clustering uses all 74,880 uniform training observations from 90 sources.
The existing primary-seed-17 all-training fits replace the interface-adjacent fits;
neural models, neural clusters and neural PaCMAP are unchanged.

- [GeoFormer · held-out MD](../matched-al64-20260929/analyses/interface-pacmap-v1/index.html)
- [GeoFormer · six static Al snapshots](../static-al-six-20260929/analyses/interface-v1/index.html)
- [MACE · held-out MD](../mace-rh2-best-20260930/analyses/matched/index.html)
- [MACE · six static Al snapshots](../mace-rh2-best-20260930/analyses/static/index.html)

[Held-out metrics](analyses/matched/tables/metrics.csv) · [definitions](analyses/matched/tables/METRICS.md)

[Static metrics](analyses/static/tables/metrics.csv) · [definitions](analyses/static/tables/METRICS.md)

Interface-only controls and layouts are retired from the active viewer. Highlight
interface layers changes opacity in PaCMAP only; all observations and correspondence
counts remain. Original historical analyses retain their original definitions.

Recipe: `configs/analysis/general_descriptors_20260930.json` in the repository.
Build provenance and exact fit hashes: each analysis's `technical/build.json`.
The 13 held-out MD frames contain 913,952 centers; the six static frames contain
684,723 centers. All four descriptor families have updated dense labels, samples
and ideal-lattice fits. Displayed PaCMAP populations remain 24,960 and 24,000.
