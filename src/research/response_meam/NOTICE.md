# Lee2003 Al PyTorch implementation

`potential.py` specializes the energy formulas, screening and pair-table
construction of LAMMPS `stable_22Jul2025` to the repository's SHA256-pinned
Lee2003 Al library and parameter files. It is a derived work distributed under
GPL-2.0-or-later. LAMMPS copyright (2003) Sandia Corporation; under Contract
DE-AC04-94AL85000 the U.S. Government retains certain rights in that software.
The license is included in `COPYING`.

Upstream: https://github.com/lammps/lammps/tree/stable_22Jul2025/src/MEAM
Relevant producers: `meam.h`, `meam_funcs.cpp`, `meam_setup_done.cpp`,
`meam_dens_init.cpp`, `meam_dens_final.cpp`, `meam_force.cpp`.
The benchmark stores the consulted upstream source files and their hashes.

This is the pure-Al specialization (`fcc` reference, `ibar=3`, `ialloy=2`,
`nn2=1`, `augt1=0`, `erose_form=2`, `zbl=0`). It rejects different parameter
files, does not implement alloys or periodic boundaries, and does not fit a
replacement potential. FCC is the potential's reference lattice; arbitrary
input geometries, including liquid environments, use the full angular density
and geometric screening equations.
