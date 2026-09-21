# Short-horizon crystallization information diagnostic

Submitted detached CPU job **1001428**, four parallel lanes. Five frozen encoders; 148 hazard readouts and 11 physical decoders. Horizons .75,3,6,9,12 ps. No GPU training interrupted.

[Scientific protocol](../../../experiments/crystallization_information_20260920/README.md), [results](RESULTS.md). Five focused tests and a real-data four-update smoke on all 191,688 rows passed. Cached inputs and checkpoint ancestry/checksums were verified. This new short-horizon likelihood is not numerically comparable to older .75–96 ps NLL values.

Regenerate a partial report with `python -m src.research.crystallization_information --config configs/analysis/crystallization_information_short.json --stage report`. Full report is generated automatically on completion.
