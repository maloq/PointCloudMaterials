# Source and license

The interpolation coefficients follow LAMMPS `PairEAM::interpolate` and its
EAM/FS evaluation, pinned source commit `8de817dd79bfe4525d5d39246a212d833e6dee07`.
The CUDA spatial-bin construction adapts the repository's GPL CUDA MEAM topology.
This implementation is distributed under GPL-2.0-or-later; see COPYING.
LAMMPS copyright Sandia Corporation; contributing EAM authors Stephen Foiles
and Murray Daw. https://www.lammps.org/

The physical potential is the unchanged author-supplied
`Ni1_Mendelev_2012.eam.fs`, archived from NIST. Its provenance and checksum are
recorded beside the file. This is pure Ni, not a Ni-Zr alloy implementation.
