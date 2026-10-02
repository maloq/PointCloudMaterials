# Al64 response-transfer / MM-TDA / GeoFormer PaCMAP

This visualization uses the existing fixed Al64 structural assay identity
c580bd469e360b76d60a2d88fb3ed3b6c9eccec06eaefd22776d97c831e74d39,
derived from fixed release e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d.
All24,960 uniform test observations (30 sources,13 frames,64 centers) are shown
for every encoder. There is no source resplit or model-dependent row removal.
Eight uniformly sampled centers per train source/frame, chosen without labels
using seed20261002, give9,360 fitting landmarks across90 training sources.
Selection/calibration sources are excluded from projection fitting and display.

Every model consumes the same centered nearest80 unrelaxed atomic coordinates.
All selected atoms are verified within8A. There is no history, motion, relaxation,
temperature, species ID, absolute time, or other condition input. Frame IDs remain
metadata and hover labels. Predictor heads are unused. No models are trained.

Response-trained MACE128 uses the checkpoint with minimum recorded selection
feature NLL across its three seeds (20261002, epoch161). Its native training
support was a complete periodic256-atom cell. The explicit transfer adapter uses
open80-atom neighborhoods, cutoff5A, constant atom channel, zero center marker,
unit atom weights and population mean/variance scalar pooling. It retains the
original learned projection/readout and trained pool normalizer. No artificial
periodicity or padding is introduced. On the original256-cell geometry, the
generalized adapter must reproduce the native export within1e-5 relative L2 error;
absolute error is recorded too. Repeated CUDA inference uses the same relative
L2 tolerance, because float32 scatter roundoff near zero makes coordinate-wise
relative error misleading. It does not require bitwise-identical CUDA reductions.
This is an input-domain transfer diagnostic, not the validated full-cell task.

MM-TDA-BLOCK-DIRECT-FULL is the exact previously selected label-free epoch20
checkpoint, native local z256 before descriptor heads. Its archived inference
uses the fixed Al length factor, radial taper and center marker, bfloat16
autocast and batch128. GeoFormer is the existing primary matched spatial-VICReg
S1/seed17/epoch24 checkpoint: native forward_features z128 before its VICReg
projector, float32, batch256, original9.192189A coordinate divisor. Response
inference is float32/batch64. Frozen source trees own model imports. The same
input points do not imply matched architecture, pooling, pretraining or precision.
MM-TDA is the prior strongest reference in the shooting comparison, not a
checkpoint selected using these maps. GeoFormer retains the recorded fixed-Al
ancestry; MM-TDA retains its historical multimaterial ancestry limitations.

Coordinate-wise mean and population standard deviation (floor1e-5) are fitted
only to the9,360 landmarks, with equal rows and equal source/frame counts. This
external visualization normalization is separate from each encoder's retained
learned normalization. PaCMAP0.9.1 fits those training landmarks and transforms
all test embeddings against that basis. Parameters:15 neighbors, mid-near0.5,
far2, Euclidean distance, learning rate1, iterations100/100/250, internal PCA
enabled, PCA initialization, faiss neighbors, seed20261002. Independent map axes,
sizes and island separations are not comparable physical quantities. No latent
quality, clustering, prediction, or sufficiency score is inferred from this plot.

Colors reuse the existing PTM labels and nearest80 crystalline-support fraction.
A crystalline center has PTM FCC/HCP/BCC (codes1/2/3); other centers with positive
crystalline support form the mixed-neighborhood category; the remaining category
has no crystalline atom among the80 inputs. The continuous fraction counts
FCC/HCP/BCC atoms among those80 candidates. Source colors are an audit of source
effects. No physical color enters normalization, PaCMAP, or checkpoint selection.

`coverage.csv` reports unweighted embedding dimension, training-landmark count,
held-out row count and source counts. These are coverage records, not scores.
Coordinates, row identities and visualization-normalization parameters are saved
in `data/*-pacmap.npz`. Source hashes, inference checks, package versions, recipes,
PaCMAP models and six-entry leased embedding-cache pointers remain in technical/.
