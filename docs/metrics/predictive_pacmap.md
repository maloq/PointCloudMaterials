# PaCMAP visualization of the shooting follow-up embeddings

Recipe: `configs/analysis/predictive_pacmap_20261001.json`. These are exploratory
views of existing frozen embeddings, not new encoder training or predictive
performance measurements. The original Al480 observation/target identity and
11/3/6 training/selection/test source roles are preserved. Every encoder displays
all 2,285 test rows, including 768 clear-liquid, 768 interface and 749 crystalline
centers. The three frozen references plus all twelve newly trained MACE variants
and seeds are projected. The default new encoder is full/free, seed20261001,
selected by the recorded moment validation likelihood, never by map appearance.

For every encoder, coordinate mean and standard deviation are fitted using only
the original 4,224 training rows and source-normalized inclusion weights. Standard
deviations are floored at 1e-5. PaCMAP itself fits the standardized training rows
with equal row weights (its API does not take population weights). Selection and
test rows never fit normalization, PCA, neighbors among training rows, or the
training layout. The test coordinates use PaCMAP's out-of-sample transform against
the fixed training basis. They are not a joint train/test fit. The library warns
that transformed observations can differ from their fit-time positions.

PaCMAP0.9.1 uses Euclidean distance, PCA initialization, default internal PCA
preprocessing (100 components when embedding dimension exceeds100), 10 neighbors,
mid-near ratio0.5, far ratio2, learning rate1, iterations100/100/250, faiss neighbors
and seed20261001. Inputs are the stored z128 or z256 exports, never head outputs,
future targets, temperature or time. Coordinate standardization changes the raw
latent distance geometry and is explicitly part of this common visualization.
Each encoder gets its own map; absolute coordinates and island spacing cannot
be compared between encoders. No clustering or embedding-quality score is fitted.

`projection-coverage.csv` records encoder ID, exported and projected dimension,
number of fitting/displayed observations and number of their distinct sources.
Counts are unweighted, not uncertainty estimates. Coordinates and original row,
parent and atom IDs are in `data/*.npz`; complete coloring data and source labels
are in `data/observations.json`. Source checkpoints, embedding hashes, normalizer,
PaCMAP model, recipe and package versions are retained in `technical/` and `data/`.

Present structure reuses the original full-cell PTM stratum: clear liquid is a
noncrystalline center more than8Angstrom from any crystalline atom; interface is
a noncrystalline center within8Angstrom; crystalline center is FCC/HCP/BCC.
Future crystalline fraction is the PTM-crystalline fraction within8Angstrom among
the future nearest80 neighborhood, following the original target producer. Colors
are its arithmetic mean across12 shooting branches at3/6/12ps, or sample standard
deviation (ddof1) at6ps. Fractions are displayed in percent; SD in percentage
points. These are observed local fractions, not crystallization-event probabilities.
The extra q-bar-6 color is its12-shot mean at6ps. No noise correction applies to
these individual color values. No color enters the projection calculation.

Every point has equal size and the original stratified sample is retained without
resampling. Apparent density therefore does not represent population frequency.
Filtering does not refit maps or change their extents. Static all-environment
crystallinity uses0–100%; the liquid view uses0–its observed maximum. Interactive
continuous scales span the selected population from min(0,min) to max, identically
across all panels. These descriptive display limits do not select models.

All observations remain the historical local shooting assay, not Al64's fixed
window benchmark. MM-TDA retains its larger capacity, broader pretraining and
archived ancestry limitations. Interpret maps as visualization, not proof of
sufficient predictive information or of the absence of nonlinearly decoded signal.
