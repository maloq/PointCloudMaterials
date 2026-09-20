# Expanded neighborhood JEPA on node53

Paired-GPU MACE arm E, width64 (657,299 encoder parameters), global anchor batch 1024.
32,768 native Al training anchors / 90 lineages; 480 held-out selection anchors.
Existing Lee 2003 MEAM dynamics, instantaneous TDA and physical anchors, fixed
angular moments and sample-normalized SIGReg. Compiled BF16, one seed.

Target: approximately three training hours, plus preflight and frozen crystallization probes.
The separate preflight freezes at least 40 sampled epoch equivalents. Exact budget:
`technical/resolved-config.json`; status: `technical/launcher-status.json`.
The coordinator waits for the previous comparison queue to finish on this node.

Results: [combined analysis](../large-20260920/RESULTS.md) and [frozen crystallization evaluation](CRYSTALLIZATION.md). Training stopped at its three-hour limit: 2,692/3,040 updates (84.125 sampled epochs); saved-checkpoint evaluation completed separately.
Treat scale comparisons as exploratory: data, width, batch and budget change together.
