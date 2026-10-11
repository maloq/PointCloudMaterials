# Ni crystallization progress and retrospective order preview

This descriptive preview uses current native-coordinate 1-ps PTM monitoring
and separate archived float16 Q/V trajectories. It does not replace the planned
0.1-ps four-detector lineage analysis. Each monitor CSV is copied before reading
and hashed, so running sources are represented by an explicit observation prefix.
Crystal fraction counts FCC/HCP/BCC. Hold time subtracts the 20-ps cooling ramp.
Threshold crossings are first monitor observations, without persistence or
lineage confirmation; they are not birth times or critical nucleus estimates.

Only the first two completed 1100-K sources enter the retrospective preview.
For each, choose the first native monitor observation with largest PTM component
at least 128 atoms, then select the largest PTM component on its archived
float16 coordinates. Quantization can change the selected component size.
Save those exact atom IDs and inspect the same atoms every 1 ps from 30 ps before
this anchor through the anchor. This outcome-selected cohort is descriptive,
not a prediction split or a prospective detector.

PTM uses RMSD 0.10; crystal labels are FCC/HCP/BCC. All other labels are called
noncrystalline, which alone does not establish thermodynamic liquid behavior.
Ni neighbors lie within 3.2 Angstrom. Compute normalized l=6 bond correlations
using freud SolidLiquid; coherent degree counts neighbors with correlation >0.7.
A coherent atom has at least 7 such neighbors, as in the main search. For both
the fixed future-component cohort and all other atoms, report coherent and
noncrystalline atom counts and their ratio among noncrystalline members only.
A zero eligible denominator is missing, not zero. Also report PTM crystalline
fraction within each group and the largest whole-cell PTM component.

The comparator is the same frame's other atoms, not a matched independent
control. Spatial proximity to an already-present embryo, temporal dependence,
membership selection and detector thresholds can explain enrichment. Signals
preceding the 128-atom anchor are not automatically signals preceding a birth.
The first 8-atom monitor crossing anywhere is shown only as context and does
not establish ancestry. Establishing a prestructured-liquid pathway requires
lineage, persistence, geometry/precision sensitivity and additional sources.
There are two independent melts; atoms or frames are not independent replicates.


Table export: 2026-10-07T16:52:04.326700+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/ni_births_preview.json` relative to the analysis root.
