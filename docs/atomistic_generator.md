# Force-driven atomistic benchmark

`src.data_utils.synthetic.atomistic_generator` builds three aluminium environments for every
explicitly configured random seed:

1. finite-temperature FCC bulk equilibrated in NPT;
2. a liquid melted at 1600 K, quenched, and sampled as a finite-time supercooled state at 650 K;
3. a transient solid-liquid growth front made by melting a periodic slab, rapidly quenching it, and evolving it briefly at 650 K.

There is no density input. At fixed atom count, pressure, and temperature, the barostat changes the cell volume and the measured number density is `N / <V>`. This follows the constant-stress molecular-dynamics construction of [Parrinello and Rahman](https://doi.org/10.1063/1.328693) and the MTK NPT equations implemented by ASE ([Martyna, Tobias, and Klein](https://doi.org/10.1063/1.467468)). The solid, liquid, and interface therefore do not receive independently chosen packing densities.

The force model is an explicit local file below `datasets/`. Generation never downloads a checkpoint or substitutes a calculator. Every output records the model SHA-256 digest. The supplied recipe uses MACE-MPA with cuEquivariance CUDA kernels. A local parity test on 256 perturbed Al atoms found a force RMSE of `8.2e-7 eV/A` and a maximum force difference of `2.8e-6 eV/A` relative to the original e3nn kernels. The calculator also retains a neighbor graph with a configured Verlet skin and rebuilds it from an exact atomic-displacement and cell-deformation bound. On 9,216 atoms, the selected `0.3 A` skin reduced measured 650 K NPT time from `0.468` to `0.213 s/step` and 1600 K NVT time from `0.411` to `0.242 s/step`. Cached-versus-fresh graph parity gave `7.3e-7 eV/A` force RMSE and `9.5e-9 eV/A^3` maximum stress difference. The underlying foundation-model scope and limitations are described by [Batatia et al.](https://arxiv.org/abs/2401.00096). This is still an approximate Hamiltonian, so matching a density does not establish that every local motif is quantitatively correct.

## What the labels mean

`solid_bulk`, `liquid_bulk`, and `interface` are preparation/context labels:

- bulk endpoints are known from how the simulations were prepared;
- interface atoms lie within a declared distance of either prepared slab boundary;
- grain ancestry and the prepared boundary are preserved in metadata.

CNA, PTM, and bond-order parameters do not create the labels. PTM is run only after generation as a falsification check: the solid endpoint must remain FCC, the liquid endpoint must lose crystalline order, and the interface cell must contain both crystalline and non-crystalline populations. A failed check aborts generation.

The implementation calibration used the configured MACE-MPA checkpoint on a deliberately small 288-atom cell. The constrained slab was 41.7% FCC after melting and 45.1% FCC after rapid quench/capture, so both sides of the front survived even under strong finite-size interaction. The production cells are longer along the interface normal and must independently pass the configured 20--80% crystalline acceptance interval.

At 650 K the solid-liquid interface is not an equilibrium coexistence interface; it is a moving growth front in an undercooled liquid. This matches the repository's nucleation setting. Its volume is constructed from the measured solid and liquid endpoint volumes in the same replica, weighted by the prepared slab fractions. The rapidly quenched front is then evolved at fixed volume and explicitly recorded rather than being falsely described as an equilibrium interface.

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
  --config configs/data/atomistic_al_phase_context.yaml
```

The single production recipe fails if the selected potential is absent, an endpoint is structurally wrong, pressure has not converged for a bulk branch, atoms overlap, forces are non-finite/excessive, or simulated bulk densities disagree with the repository MD reference beyond the configured tolerance.

Each environment stores its final atom table and a `trajectory.npz` containing wrapped positions,
cell vectors, thermodynamic values, and exact MD step indices at the configured sampling interval.
For the production recipe this retains 51 solid frames, 31 target-temperature liquid frames,
and 11 growth-front frames per replica.

## Crystallization and melting trajectories

`src.data_utils.synthetic.atomistic_transition_generator` models phase change itself with
planar direct coexistence. It reads step 0 of the repository-owned prepared interface, assigns
independent velocities, and evolves two copies in MTK NPT at zero pressure:

- `crystallization`: 650 K, where the undercooled liquid slab should be consumed by the two
  periodic crystal fronts;
- `melting`: 1100 K, where the solid regions should be consumed by the liquid fronts.

This follows direct-coexistence simulations of aluminium, which construct explicit solid and
liquid slabs and infer growth or melting from interface motion. The method avoids the superheated
solid and rare homogeneous-nucleation waiting time of single-phase heating/cooling. It is used in
first-principles aluminium melting calculations ([Alfè](https://discovery.ucl.ac.uk/id/eprint/8715/))
and modern aluminium growth simulations with roughly 46,000--92,000 atoms
([Sun, 2024](https://www.nature.com/articles/s41467-024-50182-7)). MACE is the force provider;
the phase-change protocol is ordinary molecular dynamics. Official MACE documentation recommends
the ML-IAP interface for still larger or longer GPU runs, while supporting cuEquivariance and
atomic virials ([MACE ML-IAP documentation](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html)).

Every branch stores 101 full atomic states (99 intermediate states plus endpoints, every 50 fs)
in `trajectory.npz`, plus `transition_progress.npz` with:

- PTM FCC/HCP/BCC/ICO/other fractions for each frame;
- crystalline fraction inside the initially liquid and initially solid spatial regions;
- a one-dimensional crystalline-fraction profile along the interface normal;
- net displacement per periodic front and its finite-trajectory average velocity.

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

Each branch also contains `visualizations/structure_slice.png`: central-y atomic slices at the
initial, midpoint, and final frames. Atoms are colored by PTM structure and the dashed lines mark
the two originally prepared solid-liquid interfaces. `structure_slice_overview.png` places the
crystallization and melting slices together at the dataset root.

PTM is an audit observable, not a training label. `phase_id` remains the initial preparation
provenance of each atom. This is important for testing whether learned representations discover
interfacial precursors, stacking faults, transient ordering, or other motifs without receiving
CNA/PTM/BOP classifications as targets.

Run the transition generator after the phase-context dataset exists:

```bash
conda run -n pointnet python -m src.data_utils.synthetic.atomistic_transition_generator \
  --config configs/data/atomistic_al_phase_transition.yaml
```

RDFs are also an explicit post-processing operation, so an already running or completed MD job
does not need to be repeated after changing RDF resolution:

```bash
conda run -n pointnet python -m src.data_utils.synthetic.atomistic_transition_rdf \
  --config configs/data/atomistic_al_phase_transition.yaml
```

The two trajectories demonstrate seeded front growth and melting for this MACE Hamiltonian. They
do not determine a homogeneous nucleation rate or a thermodynamic melting point. The latter needs
a temperature bracket around zero interface velocity; quantitative kinetics should also be
repeated over seeds, interface orientations, cell lengths, and a fine-tuned potential validated on
solid, liquid, and interfacial DFT configurations.

## Homogeneous crystallization from supercooled liquid

`src.data_utils.synthetic.atomistic_homogeneous_crystallization` starts from the exact final
frame of the validated 70,304-atom bulk liquid rather than the prepared solid-liquid interface.
It assigns 500 K Maxwell-Boltzmann velocities and evolves the unseeded liquid for 10 ps in MTK
NPT at zero pressure. The temperature follows ML-Al simulations reporting spontaneous
crystallization at 500--540 K ([Tipeev and Zanotto, 2025](https://www.sciencedirect.com/science/article/abs/pii/S1359645425005324))
and is consistent with earlier ambient-pressure work on deeply supercooled aluminium
([Desgranges and Delhommelle, 2007](https://pubmed.ncbi.nlm.nih.gov/17935411/)).

The trajectory stores 101 full states, including both endpoints. For every saved state, OVITO's
compiled analysis computes the total Al RDF and PTM structure types. FCC, HCP, and BCC atoms are
then joined with a 3.5 Å neighbor cutoff; `crystallization_progress.npz` records the number of
connected crystalline clusters and the largest cluster size. The configured analysis event is
the first saved frame with a connected cluster of at least 100 atoms, matching the published
nucleation-time convention. The same study estimated a 20--30 atom critical size at 500 K using
a different, pair-entropy fingerprint. Our PTM threshold is therefore an operational trajectory
observable, not a computed critical nucleus size.

If the finite run never crosses the threshold, generation still succeeds and metadata records
`observed: false`. A single absence or occurrence in 10 ps is not a nucleation rate. Estimating a
rate requires independent trajectories, survival statistics, finite-size checks, and validation
of the potential's liquid/crystal free-energy difference.

Run locally or submit `scripts/job_al_homogeneous_crystallization.sh` to Slurm:

```bash
conda run -n pointnet python \
  -m src.data_utils.synthetic.atomistic_homogeneous_crystallization \
  --config configs/data/atomistic_al_homogeneous_crystallization.yaml
```

Ready outputs are written below
`output/synthetic_data/al_homogeneous_crystallization_70304/`, including the full trajectory,
cluster/RDF archives, endpoint metadata, PTM-colored initial/midpoint/final structure slices, and
`crystallization_overview.png`.
