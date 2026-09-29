# Full-snapshot MD display assignments

This adds a visualization population, not a predictive metric or new fit.
The reference recipe selects one already held-out snapshot by maximum accepted
interface-layer atom count, independent of encoder clusters. All 70,304 atoms
are displayed; the PaCMAP evaluation rows and coordinates remain unchanged.

Full-source wrapped coordinates and PTM labels use the existing observed data
and frozen PTM extraction. New local patches are periodic nearest-80 geometry;
previously observed patches retain their saved nearest-neighbor ordering.
Rich descriptors use the original producer. Previously cached targets are
verified against recomputation (`rtol=1e-5, atol=1e-7`) then preserved exactly.

Classical cluster assignments use the frozen train-interface means, standard
deviations, active columns, square-root family balancing and primary-seed
K=7 centroids. Standardized columns cast to float32 before balancing, matching
the original clustering producer. Original sampled labels must agree exactly.
All four descriptor clusterings remain independent; no color permutation is fit.

Neural coloring uses frozen saved checkpoints and original model source, CUDA
float32 without TF32, centered geometry with recorded fixed length scale,
eval mode, disabled gradients and no optimizer. Saved K=7 centroids assign
raw encoder and projector vectors. The original observed rows are compared to
saved assignments with the existing at-most-two disagreement gate; their
original assignments are retained. New dense observations use 256-row batches;
this is a declared visualization inference, not the original A+B evaluation
batch replay. No conditions, motion, history, time, future labels or teachers
are added. Checkpoint and asset hashes and discrepancies are recorded.

The viewer uses identical tab10 entries for each cluster ID in PaCMAP and MD.
Matching an NN cluster's color across panels does not align it semantically
with an independently fitted descriptor cluster. Both MD panels show the same
coordinates and chosen z slab, with independent cameras/legends and no linked
atom events. No unknown labels are interpolated or painted onto nearby atoms.

Implementations: `dense_md.py`, `md_space.py`, `pacmap_md_view.html` in
`src/research/spatial_vicreg_bias/`. Existing historical metric definitions and
scientific exported tables are unchanged. Dense assignments and publication
receipts are separate artifacts.
