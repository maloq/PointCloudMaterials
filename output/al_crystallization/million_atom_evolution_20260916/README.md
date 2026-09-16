# Million-atom Al crystal evolution

Six central-slice panels at 0, 120, 160, 200, 240 and 400 ps after the quench to
450 K. Each snapshot is classified with full-periodic-box PTM at RMSD cutoff 0.10
using the retained original text snapshot (not float16 coordinates). Full-box
structure fractions are checked against the simulation's saved assessments.
Only after classification is a central y slice of thickness 2.5% of the box selected
for display. Slices occupy the same fractional region, not the same tracked atoms.
X/Z coordinates are normalized to the original box aspect; these are spatial slices,
not grain-orientation maps. FCC/HCP/BCC denote matched local templates; Other can
include liquid-like environments, interfaces and defects.

The fraction plot reuses all saved 4 ps assessments. This is a visualization of
recorded producer measurements, not a recomputation of historical metric tables.

Reproduction (pointnet):
`QT_QPA_PLATFORM=offscreen OVITO_THREAD_COUNT=8 OMP_NUM_THREADS=8 python output/al_crystallization/million_atom_evolution_20260916/technical/render.py`.

Figures are in `plots/`; exact provenance and selected slice arrays are in
`technical/`. The render script is a disposable run diagnostic, not a new maintained
workflow. No simulation inputs or stored trajectories were changed.
