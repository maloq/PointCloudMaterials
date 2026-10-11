# Matched transformer response and training-horizon comparison

Question: does physical derivative supervision help when the transformer is
trained over the same six-picosecond duration used for free-rollout evaluation?
The previous transformer was trained only through one picosecond, and lacked
a matched transformer values-only arm.

Eight fits cross training horizons of 1 and 6 ps, physical derivatives off/on,
and seeds 20261013/20261014. All fits use 4,000 updates, batch/microbatch 256,
AdamW maximum learning rate 0.001, 200-update warmup and cosine decay to a 1%
floor. The transformer retains two width-64, four-head blocks, four history
slots and a 16-dimensional code. Neural integration advances by 5 fs. The
1-ps likelihood observes 0.1/0.5/1 ps; the 6-ps likelihood observes
0.1/0.5/1/3/6 ps with equal weights. Each horizon selects its checkpoint using
the corresponding label-free validation likelihood. Thus the horizon factor
changes both the training and selection horizon; comparisons of derivatives
within each horizon have identical selectors.

Every arm receives the same physical value targets at 5,10,...,100 fs. Only
the derivative factor adds the initial-position/velocity response term. The
completed short teacher bank, physical precision and fixed-box BAOAB protocol
are unchanged, and remain distinct from the long native NPT trajectories.

Data are the completed 14,400 fixed-central-atom local Ni sequences: 3,840
fitting rows, 960 selection rows and 9,600 held-out rows, with the existing
16/4/40 source-ancestor partition. Inputs retain frozen unrelaxed eSEN
geometry, actual-velocity features and descriptors. No new time, temperature,
species or material inputs, neighborhood filters or broad spatial replay are
introduced. This reuses the expanded original/liquid/front collection; it is
not the earlier proposed 36,608-row fitting collection.

Each model receives autonomous rollout/proper-score/stability evaluation to
6 ps, the held-out dense physical derivative assay, and the unchanged pooled
32/64-atom nucleus probes at 1/3/6-ps lead times. AP remains diagnostic only.
No outcomes determine the queue order or scientific checkpoint selection.

Recipe: `configs/ni_latent_sim/autonomous_overnight_20261011.json`.
The operational launch and optional review are documented in
`docs/ni_latent_sim.md`; this queue starts before any chat review can happen.
