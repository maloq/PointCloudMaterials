# Six relaxed static Al snapshots: frozen interface-cluster transfer

The population is the existing static three-edge-layer grid at 166, 170, 174,
175, 177 and 240 ps: 117,649 / 117,649 / 110,592 / 110,592 / 110,592 /
117,649 centers, totaling 684,723. Every source contains 1,048,576 atoms.
Only grid coordinates/source rows are reused from the CD-MACE analysis;
its cluster assignments and PTM labels are not reused. Source hashes and exact
source-row coordinate equality are checked. This is exploratory transfer to
already-relaxed structures of incompletely recorded ancestry/potential, not an
independent six-source predictive evaluation. Snapshot time organizes displays
only; cross-snapshot atom identity is not asserted.

Each center uses its nearest 80 atoms from the full source, including itself,
centered without periodic wrapping. Missing periodic-cell metadata is never
inferred from coordinate extrema. The bounding outline shows coordinate extent.
Every retained neighborhood must fit inside these bounds. Full-source PTM uses
RMSD cutoff 0.1; types 1/2/3 (FCC/HCP/BCC) are crystalline. The 3.6 Å graph and
64-atom minimum connected components define the crystal-side interface next to
connected PTM-unclassified matter. Interface atoms must belong to a ≥64-atom
crystal component and be ≥8 Å from coordinate extrema, excluding artificial
outer-surface interfaces. PTM-unclassified matter need not be liquid.

Rich descriptors use the established nearest-80 producer: TDA, bond order,
CNA and their balanced combination. Frozen train-interface K=7 cluster models
from the preceding MD study supply columns, means, deviations, balancing and
centroids. Standardization casts to float32 before family balancing. Nothing
is fitted to static data. The two neural checkpoints (S1 seed17 epoch24,
S0 seed17 epoch4) use their original model implementation, CUDA float32 with
TF32 disabled, eval mode, disabled gradients and no optimizer. Their original
K=7 centroids assign raw encoder/projector vectors. Input is relaxed nearest-80
geometry divided by the saved Al length scale; no species, time, temperature,
history, motion, future labels or teachers. There is no predictor. The two
examples differ in epoch as well as neighbor alignment and are not a causal
single-factor comparison. Neural sample vectors stay in RAM, not a durable
feature cache; predictions, descriptor arrays and projections are retained.

For each snapshot / neural space / descriptor family / population, the metric
producer tabulates descriptor rows × neural columns. ARI and AMI use their
usual chance adjustments; mutual information and conditional entropies are
in natural-log units. Counts and contingency tables accompany the scores.
Populations are all grid centers; ≤12 Å interface context; crystal-side boundary;
connected disorder ≤12 Å; small disordered pockets ≤12 Å; connected disorder
≤20 Å with exactly zero PTM-crystal atoms in its 80-atom input; and connected
disorder in (0,3.6], (3.6,8], (8,12], (12,20] Å shells. Distances are unsigned
to the nearest accepted crystal-side boundary atom. Fewer than two observations
has `defined=false` and only a count, never a fabricated score. No qualifying
interface has infinite distance and contributes only to all-grid metrics.
`sources=1` describes one collection; no source/atom bootstrap or independence
claim is made. These are correspondence diagnostics, not future prediction,
conditional-information estimates, precursor labels or proof of physical states.

PaCMAP uses 4,000 uniform centers per snapshot, sampled without replacement with
seed 20260929 + the snapshot time label. All eight feature spaces use identical
rows. Separate all-static and ≤20 Å interface projections use the original
PaCMAP 0.9.1 settings. Axes/gaps are visualization geometry. Two dense MD panels
show all grid centers with NN and descriptor colors, exactly matching their
own cluster colors on PaCMAP. Independent neural/descriptor numeric cluster
IDs are not semantically aligned. Atom event linkage is disabled.

Numerical implementation: `static_md.py`, `data.py`, `correspondence.py`,
`crystal_vector/interface.py` and the rich-descriptor kernels recorded in the
contract. Tables export with frozen descriptions and implementation hashes.
