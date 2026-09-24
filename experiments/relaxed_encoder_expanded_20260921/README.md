# Expanded paired relaxation and conditional rank regularization

Question: does relaxed-input prediction improve on a larger cohort, and can the
snapshot export preserve more within-temperature information with direct SIGReg
or VICReg variance/covariance regularization?

Recipe: configs/analysis/relaxed_encoder_expanded.json. One seed; MACE width64;
all variants start from the same original development-selected parent, not from
an already low-rank pilot continuation. Calibration/test sources never train the
encoder. Source split is historical, not an untouched new test set.

Preliminary readouts of the first three completed runs use the earlier fixed
two-origin cohort (frames 64 and 368), via `relaxed_encoder_interim.json`.
This avoids selecting evaluation cells by relaxation completion speed. Source
splits, original MD onset labels, and matched linear/MLP readout settings are
preserved. Its 758 test windows contain eight positives by 12 ps; these results
are diagnostic and do not replace the larger assay or select hyperparameters.

## Data

90 training and15 development sources; encoder origins at frames64/224/368/512,
with next-frame labels at+0.75ps. 64 training centers and16 development centers
per cell:23,040 training/960 development anchors (9x/3x the pilot). All present,
neighbor and future labels follow the existing matched three-domain protocol.

All150 sources have15 predeclared assay origins:32,64,80,128,176,224,272,320,368,
416,464,512,560,608,656. No event-driven selection. Original-MD natural at-risk
population audit:10,825/2,034/1,650/4,262 train/development/calibration/test readout
windows;360/63/71/117 positive12ps windows. These overlap, not independent events.
All90 encoder-training sources are developmental and disjoint from held-out roles.

Full periodic fixed-box Lee2003 Al MEAM FIRE,0.01eV/Angstrom. Reuse verified older
full-precision local clouds when exact tracked center/query views are available;
otherwise quench the existing MD frame. No new MD integration. Radius8 support,
nearest80 tracked candidates and input/target-domain cropping remain matched.
FIRE iteration-limit retries reset its internal state at archived full-precision
coordinates, preserving force tolerance. Per-cell metadata records the retry.

Unlike the pilot, physical/TDA/order normalization and moment scales are fitted
on each target domain's training data. Hot/cold and cold/cold have identical cold
labels and normalizers. Never compare decoder MSE across target domains as an
encoder ranking. Frozen probes all predict the original-MD event with unchanged
labels and common train-only feature standardization.

## Fits

Nine2048-update runs, batch512, encoder/head peak LR5e-5/5e-4,10% warmup and cosine
decay, BF16, compiled MACE. Approximately45.5 sampled training passes, not full
shuffled epochs. All physical/TDA/order/moment/neighbor-future JEPA anchors remain.
The projector is identity: regularization acts directly on the exported128-vector.

| Input/target domain | Regularizer | Weight | Statistical population |
|---|---|---:|---|
|hot/hot|SIGReg|0.1|global control|
|cold/cold|SIGReg|0.1|global control|
|cold/cold|SIGReg|1|within temperature|
|cold/cold|SIGReg|3|within temperature|
|cold/cold|VICReg|0.01|within temperature|
|cold/cold|VICReg|0.1|within temperature|
|cold/cold|VICReg|0.1|global|
|hot/hot|VICReg|0.1|within temperature|
|hot/cold|VICReg|0.1|within temperature|

Conditional penalties average over temperatures represented in each batch; only
independent current-anchor slots enter the statistic, not their correlated views.
No phase labels partition the training regularizer. VICReg denotes variance and
covariance terms; JEPA supplies prediction, not forced temporal equality.

## Evaluation

Nine trained encoders plus unchanged parent on hot/cold inputs and four descriptor/
condition controls:30 frozen readouts (linear and MLP). Readout architecture,
source-balanced updates and budgets are matched. Report0.75/3/6/9/12ps NLL, AP,
calibration and timing with misses; source-bootstrap uncertainty excludes seed
uncertainty. Select the cold variant by development physical+.25TDA only, never
by test AP or maximum rank. Report covariance rank, correlation rank, channel
standard deviations and within-temperature noncrystalline covariance ranks.
A higher rank with worse physical/event information is not a success.

Timeout policy update (user requested): skip timed-out cells instead of stopping
or retrying the entire campaign. Remove affected training pairs and common assay
windows across all comparison arms; export exclusions and retained event counts.
Already recovered cells remain included. This introduces conditional-on-success
sampling and must accompany the scientific results.

[Static Al analysis](STATIC_AL.md) uses the selected cold-vic-temp01 step2048 embedding directly on the existing relaxed snapshots.
