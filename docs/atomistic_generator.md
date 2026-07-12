# Force-driven atomistic benchmark

`src.data_utils.synthetic.atomistic_generator` builds three aluminium environments:

1. finite-temperature FCC bulk equilibrated in NPT;
2. a liquid melted at 1600 K, quenched, and sampled as a finite-time supercooled state at 650 K;
3. a transient solid-liquid growth front made by melting a periodic slab, rapidly quenching it, and evolving it briefly at 650 K.

There is no density input. At fixed atom count, pressure, and temperature, the barostat changes the cell volume and the measured number density is `N / <V>`. This follows the constant-stress molecular-dynamics construction of [Parrinello and Rahman](https://doi.org/10.1063/1.328693) and the MTK NPT equations implemented by ASE ([Martyna, Tobias, and Klein](https://doi.org/10.1063/1.467468)). The solid, liquid, and interface therefore do not receive independently chosen packing densities.

The force model is an explicit local file below `datasets/`. Generation never downloads a checkpoint or substitutes a calculator. Every output records the model SHA-256 digest. The supplied recipes use MACE-MPA; the underlying foundation-model scope and limitations are described by [Batatia et al.](https://arxiv.org/abs/2401.00096). This is still an approximate Hamiltonian, so matching a density does not establish that every local motif is quantitatively correct.

## What the labels mean

`solid_bulk`, `liquid_bulk`, and `interface` are preparation/context labels:

- bulk endpoints are known from how the simulations were prepared;
- interface atoms lie within a declared distance of either prepared slab boundary;
- grain ancestry and the prepared boundary are preserved in metadata.

CNA, PTM, and bond-order parameters do not create the labels. PTM is run only after generation as a falsification check: the solid endpoint must remain FCC, the liquid endpoint must lose crystalline order, and the interface cell must contain both crystalline and non-crystalline populations. A failed check aborts generation.

The implementation calibration used the configured MACE-MPA checkpoint on a deliberately small 288-atom cell. The constrained slab was 41.7% FCC after melting and 45.1% FCC after rapid quench/capture, so both sides of the front survived even under strong finite-size interaction. The production cells are longer along the interface normal and must independently pass the configured 20--80% crystalline acceptance interval.

At 650 K the solid-liquid interface is not an equilibrium coexistence interface; it is a moving growth front in an undercooled liquid. This matches the repository's nucleation setting. The interface snapshot is therefore captured after a short, explicitly recorded evolution instead of being falsely described as equilibrated.

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
  --config configs/data/data_synth_polycrystalline_balanced_geometries.yaml
```

The ordinary recipe is intended for validation and iteration. The `_v2` recipe is the larger, slower study configuration. Both fail if the selected potential is absent, an endpoint is structurally wrong, pressure has not converged for a bulk branch, atoms overlap, forces are non-finite/excessive, or simulated bulk densities disagree with the repository MD reference beyond the configured tolerance.
