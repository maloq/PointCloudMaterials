# Force-driven atomistic benchmark

`src.data_utils.synthetic.atomistic_generator` builds three aluminium environments for every
explicitly configured random seed:

1. finite-temperature FCC bulk equilibrated in NPT;
2. a liquid melted at 1600 K, quenched, and sampled as a finite-time supercooled state at 650 K;
3. a transient solid-liquid growth front made by melting a periodic slab, rapidly quenching it, and evolving it briefly at 650 K.

There is no density input. At fixed atom count, pressure, and temperature, the barostat changes the cell volume and the measured number density is `N / <V>`. This follows the constant-stress molecular-dynamics construction of [Parrinello and Rahman](https://doi.org/10.1063/1.328693) and the MTK NPT equations implemented by ASE ([Martyna, Tobias, and Klein](https://doi.org/10.1063/1.467468)). The solid, liquid, and interface therefore do not receive independently chosen packing densities.

The force model is an explicit local file below `datasets/`. Generation never downloads a checkpoint or substitutes a calculator. Every output records the model name, family, selected head, SHA-256 digest, source URL, license, MACE/ASE/Torch/CUDA versions, calculator settings, and a hash of the producer source. A changed Hamiltonian, runtime, configuration, or producer invalidates checkpoints. MACE's permissive unknown-head fallback is also forbidden: the configured head must exist in the loaded model and must be the head MACE actually selects.

The baseline recipe uses MACE-MPA-0 with cuEquivariance CUDA kernels. A local parity test on 256 perturbed Al atoms found a force RMSE of `8.2e-7 eV/A` and a maximum force difference of `2.8e-6 eV/A` relative to the original e3nn kernels. The calculator also retains a neighbor graph with a configured Verlet skin and rebuilds it from an exact atomic-displacement and cell-deformation bound. On 9,216 atoms, the selected `0.3 A` skin reduced measured 650 K NPT time from `0.468` to `0.213 s/step` and 1600 K NVT time from `0.411` to `0.242 s/step`. Cached-versus-fresh graph parity gave `7.3e-7 eV/A` force RMSE and `9.5e-9 eV/A^3` maximum stress difference. The underlying foundation-model scope and limitations are described by [Batatia et al.](https://arxiv.org/abs/2401.00096). This is still an approximate Hamiltonian, so numerical kernel parity and stable MD do not establish correct liquid thermodynamics or nucleation pathways.

### MACE-MH-1 candidate and qualification

`configs/simulation/atomistic/al/phase_context_70304_mh1.yaml` pins the newer
[MACE-MH-1](https://huggingface.co/mace-foundations/mace-mh-1) checkpoint and its
materials-relevant `omat_pbe` head. It records the official checkpoint digest and ASL license.
It is a candidate, not a silent upgrade: the foundation-model benchmarks do not specifically
validate undercooled Al liquid, solid-liquid interfaces, the melting point, or homogeneous
nucleation kinetics.

`src.data_utils.synthetic.atomistic_potential_benchmark` compares the MPA baseline and MH-1
candidate using:

- FCC/HCP/BCC equations of state and phase ordering;
- 0.5 fs versus 1.0 fs NVE energy-conservation runs at 650 K;
- energy, force, stress, and pressure disagreement on repository solid, liquid, and interface
  structures;
- per-state energy/force/stress errors on a provenance-complete DFT extxyz set spanning bulk
  solid, bulk liquid, interface, strained solid, nucleus, HCP, and BCC configurations; and
- replicated direct-coexistence scans whose confidence intervals robustly bracket zero
  interface velocity for the exact model SHA and head, reproduce the 933.45 K experimental
  reference within the configured tolerance, and agree across at least two cell-size/orientation
  protocols.

The supplied benchmark deliberately has no DFT or melting-scan reference yet. It therefore
writes `scientifically_qualified: false` for both models even if the numerical smoke checks pass.
`usage_mode: quantitative` is rejected unless the generator points to a qualification report for
the exact SHA/head, declares `report_type: al_crystallization_mlip_qualification`, and sets
`scientifically_qualified: true`. The report must also cover the requested
chemical symbol, pressure, timestep, calculator implementation/dtype/cuEquivariance/neighbor-skin
settings, and every solid, liquid, interface, strained-solid, nucleus, HCP, or BCC temperature
used by the selected protocol. In particular, the coexistence scope includes metastable solid
through 1000 K and the homogeneous scope includes 500 K interface/nucleus configurations. A
report for one thermodynamic window cannot authorize another. Both supplied generation recipes
remain visibly `exploratory`.

Qualification is also claim-specific. The base phase-context generator requires
`phase_context_structure`; direct coexistence requires `equilibrium_thermodynamics`; and the
homogeneous workflow requires `kinetics`. Every report records all three authorization values as
explicit booleans. This benchmark can authorize the first two after all evidence gates pass, but
always records `kinetics: false`: force accuracy, stable integration, and a melting-point bracket
do not validate diffusion, viscosity, attachment kinetics, or the MLIP time scale.

The DFT file is not accepted merely because it has ASE labels. Every required state needs at least
five independent periodic all-Al frames. Every frame must carry a unique configuration ID, state,
temperature, target pressure, electronic-structure code,
level of theory, pseudopotential, plane-wave cutoff, and k-point spacing that match the benchmark
declaration, together with finite energy, force, and stress results. Geometry hashes reject copied
frames with renamed configuration IDs, and each state's temperature and pressure samples must span
the declared scope. State-by-state thresholds
prevent a large easy subset from hiding a failed liquid, interface, or nucleus subset. The
coexistence reference is the NIST phase-transition temperature for elemental aluminium; a
resolved bracket alone is evidence about one MLIP protocol, not evidence that the Hamiltonian is
physically accurate.

Run the comparison after placing both explicitly selected checkpoints below
`datasets/potentials/`. Optional repository-generated application structures may be added to
the YAML for disagreement diagnostics, but they do not count as independent validation data:

```bash
conda run -n pointnet python \
  -m src.data_utils.synthetic.atomistic_potential_benchmark \
  --config configs/simulation/atomistic/al/potential_benchmark.yaml
```

## What the labels mean

`solid_bulk`, `liquid_bulk`, and `interface` are preparation/context labels:

- bulk endpoints are known from how the simulations were prepared;
- interface atoms lie within a declared distance of either prepared slab boundary;
- grain ancestry and the prepared boundary are preserved in metadata.

CNA, PTM, and bond-order parameters do not create the labels. PTM is run only after generation
as a falsification check: the solid endpoint must remain FCC, the liquid endpoint must lose
crystalline order, and the interface cell must contain both crystalline and non-crystalline
populations. Base endpoint checks use the same explicitly configured OVITO normalized RMSD
cutoff `0.10` as the derived transition and homogeneous analyses, and persist that cutoff in
diagnostics. A failed check aborts generation.

The implementation calibration used the configured MACE-MPA checkpoint on a deliberately small 288-atom cell. The constrained slab was 41.7% FCC after melting and 45.1% FCC after rapid quench/capture, so both sides of the front survived even under strong finite-size interaction. The production cells are longer along the interface normal and must independently pass the configured 20--80% crystalline acceptance interval.

At 650 K the solid-liquid interface is not an equilibrium coexistence interface; it is a moving growth front in an undercooled liquid. This matches the repository's nucleation setting. Its volume is constructed from the measured solid and liquid endpoint volumes in the same replica, weighted by the prepared slab fractions. The rapidly quenched front is then evolved at fixed volume and explicitly recorded rather than being falsely described as an equilibrium interface.
The prepared cell uses equal nominal solid and liquid thicknesses. In the approximately 105 Å
interface-normal cell this leaves about 52 Å for each phase, reducing premature phase exhaustion
in the derived 25 ps coexistence scans.

## Claims this benchmark can support

The benchmark can test whether a learned representation separates known bulk contexts and finds reproducible subdivisions enriched at a physically generated growth front. A candidate novel motif should additionally:

- recur across random seeds, cell sizes, and at least two validated potentials;
- be spatially coherent and temporally persistent in real MD trajectories;
- remain after controlling for density, temperature, strain, and distance to the interface;
- correlate with an independent physical observable such as local energy, stress, mobility, or future crystallization probability;
- not be merely a relabeling of CNA/PTM/BOP output.

The synthetic class labels alone cannot prove a new thermodynamic phase, determine a nucleation rate, or measure interfacial free energy. Those claims require free-energy/coexistence calculations and finite-size analysis. The large-scale aluminium nucleation methodology motivating the repository is described by [Jakse et al.](https://arxiv.org/abs/2201.01370), while the multi-material unsupervised analysis used in the repository datasets is described by [Becker et al.](https://doi.org/10.1038/s41598-022-06963-5).

## Running

```bash
conda run -n pointnet python -m src.data_utils.synthetic.atomistic_generator \
  --config configs/simulation/atomistic/al/phase_context_70304_mpa.yaml
```

The single production recipe fails if the selected potential is absent, an endpoint is
structurally wrong, pressure has not converged for a bulk branch, atoms overlap, or forces are
non-finite/excessive. The old density cache is deliberately not used: it lacks the potential,
temperature, pressure, and phase provenance required to serve as a scientific reference.

Each environment stores its final atom table and a `trajectory.npz` containing wrapped positions,
cell vectors, thermodynamic values, and exact MD step indices at the configured sampling interval.
For the production recipe this retains 51 solid frames, 31 target-temperature liquid frames,
and 11 growth-front frames per replica.

## Crystallization and melting trajectories

`src.data_utils.synthetic.atomistic_transition_generator` models phase change itself with
planar direct coexistence. It reads the final step of the repository-owned 1 ps prepared-interface
trajectory rather than its high-pressure step-0 transient. Every configured temperature and
replica starts from those exact coordinates, receives independent velocities, and undergoes a
separate MTK-NPT equilibration at zero pressure. Equilibration frames are saved in
`equilibration_trajectory.npz` but excluded from the production trajectory and velocity fit.
Equilibration and production are two slices of one continuous MTK trajectory, so the boundary
does not reset thermostat or barostat chain variables.

The production recipe runs four independent replicas at 650, 800, 850, 900, 950, and 1000 K.
The 650 K and 1000 K endpoints must respectively pass explicit growth and melting checks; the
interior temperatures are not assigned a direction in advance. This grid measures the selected
MLIP's zero-velocity bracket rather than borrowing the experimental melting point or that of a
different potential. Each run uses 5 ps equilibration followed by 20 ps production; only the
5--20 ps production interval enters the velocity fit.

This follows direct-coexistence simulations of aluminium, which construct explicit solid and
liquid slabs and infer growth or melting from interface motion. The method avoids the superheated
solid and rare homogeneous-nucleation waiting time of single-phase heating/cooling. It is used in
first-principles aluminium melting calculations ([Alfè](https://discovery.ucl.ac.uk/id/eprint/8715/))
and modern aluminium growth simulations with roughly 46,000--92,000 atoms
([Sun, 2024](https://www.nature.com/articles/s41467-024-50182-7)). MACE is the force provider;
the phase-change protocol is ordinary molecular dynamics. Official MACE documentation recommends
the ML-IAP interface for still larger or longer GPU runs, while supporting cuEquivariance and
atomic virials ([MACE ML-IAP documentation](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html)).

Every temperature/replica run stores 101 full production states (99 intermediate states plus
endpoints, every 200 fs) in `trajectory.npz`, plus `transition_progress.npz` with:

- PTM FCC/HCP/BCC/ICO/other fractions for each frame;
- crystalline fraction inside the initially liquid and initially solid spatial regions;
- raw and cyclically smoothed one-dimensional crystalline-fraction profiles along the interface
  normal;
- the two oriented crossings of a branch-temperature PTM threshold calibrated halfway between
  the post-equilibration solid-core and liquid-core profile baselines;
- each interface's signed advance and a linear velocity fit over the explicitly configured
  thermodynamically stationary production interval.

PTM uses the explicitly configured normalized RMSD cutoff `0.10`; the cutoff and the calibrated
per-run solid/liquid profile baselines are recorded rather than relying on OVITO defaults.

The interface coordinate is measured in fractional-cell space and converted with the initial
interface-normal cell height, so isotropic barostat strain is not misreported as front motion.
Positive velocity always means crystal growth. This replaces the former conversion from the
whole-cell PTM atom count, which could not distinguish interface motion from thermal changes in
bulk PTM classification. Generation fails if either phase disappears, the profile contrast is
unresolved, the fit is insufficiently linear, or the equilibration/production temperature and
pressure blocks are not stationary within the configured validation tolerances.

`velocity_summary.json` retains every replica's two-front velocity, fit interval, R², residual
RMS, and ordinary-least-squares slope standard error. The latter is a fit diagnostic, not an
independent uncertainty estimate because adjacent MD frames are correlated. For
each temperature it reports the replica mean, sample standard deviation, standard error, and 95%
Student-t confidence interval. A zero-velocity interpolation is emitted only for an adjacent
positive-to-negative pair whose two confidence intervals both exclude zero. A less precise point
elsewhere in the grid does not erase that local bracket, but is retained in the report. If no
unique confidence-resolved adjacent pair exists, or a higher-temperature point robustly reverses
from melting back to growth, the bracket is explicitly `null` with an
unresolved reason; no single noisy trajectory is promoted to a melting point. These are
independent-velocity trajectories conditional on one prepared
interface configuration; their intervals do not include independent preparation, cell-size,
orientation, order-coordinate, or MLIP uncertainty.

`phase_rdf.npz` stores a radial distribution function for every saved frame and each initial
prepared provenance (`solid_bulk`, `liquid_bulk`, and `interface`). The selected atoms are RDF
centers and all Al atoms are their possible neighbors; normalization uses the instantaneous
whole-cell number density. This center-conditioned definition measures how the local coordination
around each prepared population changes without incorrectly treating the finite interface slab as
an independent periodic bulk phase. `visualizations/phase_rdf.png` shows initial, midpoint, and
final curves together with the full time-resolved RDF heatmaps.
RDF accumulation uses OVITO's compiled partial-RDF modifier and combines the phase-pair curves
into the same center-conditioned total-Al definition. It does not materialize the full neighbor
pair list in Python.

Each run also contains `visualizations/structure_slice.png`: central-y atomic slices at the
initial, midpoint, and final frames. Atoms are colored by PTM structure and the dashed lines mark
the two originally prepared solid-liquid interfaces. `structure_slice_overview.png` places the
first replica's slices for every temperature together at the dataset root; every other replica
retains its own slice image.

PTM is an audit observable, not a training label. `phase_id` remains the initial preparation
provenance of each atom. This is important for testing whether learned representations discover
interfacial precursors, stacking faults, transient ordering, or other motifs without receiving
CNA/PTM/BOP classifications as targets.

Run the transition generator after the phase-context dataset exists:

```bash
conda run -n pointnet python -m src.data_utils.synthetic.atomistic_transition_generator \
  --config configs/simulation/atomistic/al/phase_transition_70304_mpa.yaml
```

After generating the corresponding MH-1 phase-context source, run the identical candidate scan
with:

```bash
conda run -n pointnet python -m src.data_utils.synthetic.atomistic_transition_generator \
  --config configs/simulation/atomistic/al/phase_transition_70304_mh1.yaml
```

The two recipes differ in source Hamiltonian/head and output path, while grid, seeds, durations,
PTM settings, and fit windows are identical. Both remain exploratory until the benchmark receives
the required DFT and independent finite-size/orientation coexistence evidence.

RDFs are also an explicit post-processing operation, so an already running or completed MD job
does not need to be repeated after changing RDF resolution:

```bash
conda run -n pointnet python -m src.data_utils.synthetic.atomistic_transition_rdf \
  --config configs/simulation/atomistic/al/phase_transition_70304_mpa.yaml
```

The temperature grid can demonstrate seeded growth/melting and estimate a finite-protocol
zero-velocity bracket for this Hamiltonian. It does not determine a homogeneous nucleation rate or
potential-independent experimental melting point. Quantitative kinetics still require interface
orientation and cell-length convergence and a potential validated on solid, liquid, and
interfacial DFT configurations.

## Homogeneous crystallization from supercooled liquid

The current 16,384-atom production workflow is documented in
`docs/optimized_homogeneous_campaign.md`. It starts from an immutable, model-specific
500 K liquid endpoint and runs independent MTK-NPT replicas to their configured physical
endpoints. Complete MTK checkpoints provide recovery after an external interruption; no
wall-clock deadline is part of the simulation.
