# Native embedding coordinates across crystal and liquid

Descriptive frozen-checkpoint assay, not a new encoder fit or a causal test of
spatial VICReg. Input: the archived GeoFormerV2 epoch-34 checkpoint and existing
Al inherent snapshots (166/174/177 ps). The saved checkpoint disables neighbor
shifting and includes FactorVAE. Source hashes, full-cell physical reference and
checkpoint identity are checked. Raw encoder and once-applied projector outputs
use float32, deterministic evaluation grouping and nearest-80 geometry divided
by the recorded 9.192189 Å radius. No temperature/time/identity features enter
inference. Repeated inference must agree exactly.

Full-cell crystal means PTM FCC/HCP/BCC with RMSD <=0.10. Of the 14 nearest
neighbors excluding self, crystalline fraction >=0.8 defines a crystal core
(also requiring a crystalline center); <=0.1 defines bulk-like liquid with a
noncrystalline center. Reference populations exclude a one-radius external
margin. No periodic box is invented. The continuous spatial coordinate is
`d=(distance to nearest crystal-core atom - distance to nearest bulk-liquid
atom)/2`, in Å: negative toward crystal, positive toward liquid. This is a
distance contrast, **not an exact signed distance to a thermodynamic interface**.
It inherits the classical definitions and may include defects or grain-boundary
liquid-like regions. Core/liquid labels are evaluation metadata only.

Only the left fitting half of 174 ps selects coordinates: rank all 128 native
coordinates by absolute Spearman correlation with d, keep six for plots, and
freeze the order/sign across all snapshots. Report all coordinate scores. Means
and standard deviations come from that fitting half. Sign is chosen to increase
toward liquid. The right half of each snapshot supplies reported scores; the
original split has a spatial exclusion gap. These frames were in encoder
training: a held-out probe region is **not independent source generalization**.
No p-values or independent-atom confidence claims are made. Coordinate indices
are zero-based and have no invariant meaning across separately trained models.

For each coordinate, report fitting and held-region Spearman rho, plus rho on
noncrystalline centers with <=0.1 nearest-neighbor crystal fraction and on the
strict subset with zero crystalline atoms among all 80 encoder input atoms.
Fewer than three rows or constant values give undefined correlation, not zero.
The fixed bulk phase direction is `(mean_liquid - mean_crystal)` divided by its
squared norm; projecting `z-mean_crystal` gives training bulk means 0 and 1.
It is a descriptive label-derived direction, not an SSL coordinate.

Isotonic regression predicts each coordinate from d using fitting-half 174 ps
rows and the fitting correlation's sign; extrapolation is clipped. Evaluate
MSE divided by the fitting coordinate variance. Compare a two-level predictor
using the fitting mean coordinate of crystalline/noncrystalline centers and
the actual evaluation PTM class. `isotonic_minus_binary_nmse <0` favors the
distance profile over this particular abrupt two-level reference. The isotonic
curve is constrained to be monotone: its appearance cannot prove smoothness.
Raw scatter and unconstrained per-bin medians/IQRs are plotted instead. Bins
are fixed 2 Å intervals over [-25,25], with >=15 held-region rows. Bands are
the middle 50% of observations, not confidence intervals of a smooth mean.

Individual paths are selected from physical geometry alone: held-region core
anchors 12–28 Å from the nearest bulk-liquid atom, random fixed ordering,
>=25 Å between starts, at most eight per snapshot. Straight lines head toward
that bulk-liquid atom and extend 12 Å beyond it; requested 1 Å positions snap
to actual atoms. Duplicate atoms are removed, outputs are ordered by projected
physical position, and actual off-axis distances are recorded. Endpoints and
paths maintain two-radius exterior clearance and full nearest-80 support.
Paths can encounter multiple environments; no monotonic physical trajectory is
assumed. Plot actual individual outputs without smoothing/interpolation.

Path metrics: Spearman rho against actual projected position; total variation
`sum(abs(diff(y)))`; endpoint absolute change; their ratio (undefined for zero
endpoint change); median absolute adjacent jump, where y is signed and scaled
by fitting-half SD. These are path-sampling-dependent diagnostics, not estimates
of differentiability or universal correlation length. Black dashed curves show
the crystalline fraction among the actual 80 observed atoms. Nearby overlapping
patches and spatial selection do not provide independent replicates.

Smooth trends would establish a descriptive coordinate field. They do not show
that a separate liquid phase exists, identify a precursor, or establish the
neighbor-augmentation mechanism. The matched-view intervention is a separate
proposed experiment in `experiments/spatial_vicreg_bias_20260929/README.md`.

Revision 2 compares identical 256-example batches for exact repeatability and records the 64-versus-256 batch numerical error separately (relative tolerance 1e-4, absolute 1e-5). The first attempt failed a mismatched-batch bitwise check; no scientific measurements were published from it.


Table export: 2026-09-28T23:19:48.568341+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/spatial_vicreg_coordinates.json` relative to the analysis root.
