# Frozen rich-MACE interface comparison

This evaluates an immutable MM-RD-MACE256-D3-C3-L3-Z256 checkpoint trained on
442 local rich descriptors with variance/covariance regularization. Agreement
with TDA, bond-order and CNA clusters is not an independent test of discovering
those features without supervision. No neural training or checkpoint selection
occurs here. This is not a forecast or precursor test.

## Representation and fitting

The exact training producer is imported; hashes of the encoder, graph builder
and typed export dependencies must match. Inference returns its 256-dimensional
scalar state after frozen scalar centering/scaling, not descriptor predictions
or a VICReg projector. Precision is recorded BF16 autocast with float32 exports;
inference uses chunks of 256. Inputs are centered nearest-80 coordinates, fixed
material normalization (factor 1 for Al), radius 8, edge cutoff 5, and a constant
atom channel. No species, temperature, age, time, history, motion, context
predictor or training teacher enters inference.

K=7 MiniBatchKMeans fits only the original 74,880 uniform training-source rows:
seed 20260929, batch 4096, n_init 3, max_iter 200, reassignment_ratio 0, raw
exported state without fitted feature normalization. Fixed Al64 release identity:
`e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d`.
Roles and exact display identities are inherited. No held-out/static data fit
neural centroids. Descriptor clusters and descriptor projections remain frozen.

## Correspondence

Contingencies have original neural IDs on rows and descriptor IDs on columns.
One-to-one assignment maximizes shared memberships, fixed per descriptor family
and dataset: 24,960 held-out display rows for matched Al, and all 684,723 dense
centers for the six static snapshots. These are descriptive display mappings,
not learned predictive readouts.

`matched_fraction` is assigned contingency mass divided by sample count. Pair
`intersection`, `union`, `iou`, `neural_fraction` and `descriptor_fraction` use
the two original membership sets; zero denominators produce null.
`adjusted_rand_index` is sklearn's chance-adjusted Rand statistic; an empty
population produces null. Every subset uses the same fixed color mapping.

Tables cover all displayed observations, displayed observations within 12 Å of
the existing interface, and each dense snapshot overall and within 12 Å. Interface
labels come from the original physical producer; no-interface snapshots have no
near-interface rows. Families are joint TDA + bond order + CNA, TDA alone, bond
order alone and CNA alone. Counts are atoms, not independent sources; these
scores carry no source-generalization confidence claims.

## Views and interpretation

Neural 3D PaCMAP uses the original versioned parameters and exact sample IDs;
all-observation and within-20-Å layouts are separate fits. Descriptor projections,
labels, physical fields and MD positions are reused. Thirteen held-out MD frames
and six static frames retain their full previous display populations. Static
inputs are relaxed; this model was trained on raw dynamic patches, an explicit
distribution change.

Examples use the established deterministic five-per-cluster sampler and sparse
geometry. PTM/lattice fits and full-vector travel retain the existing
[definitions](interactive_structure_paths.md), with 256 scalar coordinates.
Vector packing is lossless. Paths never interpolate or modify atomic structures.

Current publication refreshes archive separate interface20 HTML and redirect
those links to the full comparison. A within-12-Å highlight changes point opacity
without filtering observations or recomputing correspondence statistics. The
original separate layouts, exported subsets and scientific definitions remain
frozen. Explicit JSON sidecars carry viewer payloads through the shared template
writer; later publication stages do not recover inputs from HTML or generated JS.
