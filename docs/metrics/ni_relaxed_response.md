# Relaxed Ni observations with unchanged original MD

This experiment pairs every row, ancestor split, observation time, center and
first-appearance/32/64-atom event label with the unrelaxed Ni response campaign.
Its original definitions in [ni_esen_response.md](ni_esen_response.md) still govern
binary scores, readout selection, prevalence, lead times and bootstrap intervals.
No label is recomputed on relaxed positions. No new neighborhood filter applies.

Relaxed dump box bounds are read in float64 for the fixed-cell identity check
(absolute tolerance 1e-10 angstrom), before archival conversion to float32 boxes.
This preserves full-precision boxes of simulated endpoints; checking already
rounded float32 bounds against float64 metadata incorrectly rejected valid cells.
MD, FIRE, local float32 observations and final storage precision are unchanged.

The completed epoch12 relaxed Small eSEN (checkpoint09e1b31dddfdf0b7...) replaces
the unrelaxed spatial start. Every input and saved future geometry is a whole-cell
fixed-box FIRE minimum of its original Ni configuration. Original atom identities
and selected nearest80 membership are retained through each quench. Original
velocities stay paired with those atoms. Coordinates and velocities receive the
same fixed Ni length normalization as spatial training. No temperature, age,
time, material/species or relaxation energy is a predictor input. No broad
spatial arrays are loaded or replayed; only checkpoint and ancestry metadata.

For clarity, m=[E(R(Q)+epsilon*0.1ps*V)-E(R(Q)-epsilon*0.1ps*V)]/(2epsilon)
with epsilon0.01 is a **velocity-conditioned geometric probe at the relaxed
configuration**. It is not dE(R(Q(t)))/dt: velocities do not pass through the
minimizer. Neither the encoder nor prediction readouts get any future state.
This preserves the existing motion feature parameterization while exposing a
quenched structural landscape. Its usefulness is an experimental question.

Short physical simulations always start from ORIGINAL unrelaxed Q,V and its
locally perturbed copies, never R(Q),V. They use the original source temperature,
box, Ni EAM/FS potential and100fs BAOAB recipe. Force arithmetic isfloat32;
MD states, momenta and noise arefloat64. Two independent shots share exactly the
same noise across the five variants: base, Q-minus, Q-plus, V-minus, V-plus.
Underlying direction RMS scales remain0.01A and0.1A/ps, multiplied by the signed
response epsilon0.5. Actual plus/minus displacements are therefore0.005A and
0.05A/ps. Long NPT futures at0.1,0.5,3,12ps remain original saved trajectories;
their geometry is quenched only for the feature/descriptor targets.

Minimization can change basins and has no generally usable infinitesimal chain
rule here. Consequently the response target is an explicitly FINITE difference
[E(R(Q_T_plus))-E(R(Q_T_minus))]/(2epsilon), averaged over paired shots. Both
perturbed trajectories are simulated and their full endpoints quenched. Student
responses use the corresponding separately quenched INITIAL plus/minus states,
not a straight-through relaxation tangent or linearization of the relaxed input.
The same particle noise is used for both student signs. Frozen adapters cache
these separate initial codes. Values compare the base path to the base quenched
endpoint. This is a finite-amplitude basin-response experiment, not an analytic
Hessian-through-FIRE claim. CUDA float32 quench roundoff can select nearby minima;
quench convergence does not establish reproducibility of basin assignment.

The100 short initial states from original training ancestors are prepared
(80 fit and20 internal selection). The original held-out200 short banks were
unused by fitting or checkpoint selection and are not recomputed. Every19200
long-observation row and11220 birth-assay row is retained. There are1930 unique
native frames, of which310 had exact-recipe native index candidates at planning.
There are two additional initial-position quenches and ten endpoint quenches
per short state; velocity perturbations reuse the same initial minimum.

The audited GPU FIRE recipe is reused through the shared relaxation index:
float64 minimizer coordinates, float32 forces, max component0.01eV/A,20000
iterations and100000 evaluations. Pre-quantization float32 local patches are
preserved before verifiedfloat16 global archival. Native restart minima enter
the shared index. Signed MD integration checkpoints stayfloat64 so that resumed
quenches do not change response inputs through quantization.

Four1600-update arms, batch256/micro32, optimizer, long likelihood, short value
and relative-response weights match the parent. Actual finite perturbations
replace its analytic MD+neural-JVP short target. AP never selects training.
The same source-weighted NLL selection and local linear/CatBoost readouts apply.
The relaxed spatial start is its own frozen baseline.

The final matched comparison reports relaxed-minus-unrelaxed AP, AUROC, NLL and
Brier with paired whole-ancestor bootstrap intervals and checks exact row IDs,
labels, roles and weights. Both feature pipelines predict the same original MD
labels. Future descriptor scores are not directly subtracted: quenching changes
those targets and their fitted standardization. All scientific fitting remains
online W&B; diagnostic readouts and numerical checks stay local.
