# Threefold motion collection with liquid persistence and front advance

The user requested14,400 local sequences from existing Ni MD, explicitly retaining
one central atom over time. This release retains all4,800 original outcome-blind
sequences and adds4,800 liquid-to-liquid and4,800 advancing-front sequences.
The60 original parent trajectories, source ancestry and roles remain unchanged;
local samples are not new independent MD runs. Every sequence has64 native0.1ps
frames, from0.3ps of past history through6ps of future. The fixed central atom ID
is passed unchanged to every frame extraction. Each observation is recentered on
that atom; its nearest80 neighbors can change. The center never switches to follow
the moving interface.

Each added regime contributes1,280 fitting,320 inner-selection and3,200 held-out
rows. Combined totals are3,840 /960 /9,600. Original birth32/64 probe rows and
labels are reused unchanged, without new neighborhood screening.

PTM FCC/HCP/BCC define the operational crystalline indicator, using the source's
full-cell RMSD0.1 classifications. Local fractions use nearest80 support within
normalized radius8. The added labels select/audit samples only; they do not enter
encoder inputs, neural losses, or nucleus targets.

* Liquid-to-liquid: the central atom is noncrystalline and its local crystalline
  fraction is at most5% at every native observation from0 through6ps.
* Advancing front: initially noncrystalline center, local crystalline fraction
  20–80%, with both a crystalline interior and a disordered reference center
  within normalized distance8. Six ps later the local fraction has increased by
  at least20 percentage points, the center is crystalline, and it is crystalline
  in at least80% of the11 observations covering the final1ps. This follows an
  initially liquid-side atom as the front reaches it.

Candidate anchor screening uses the existing exact0.1ps component histories,
starting after the20ps cooling phase. Liquid windows have at most320 tracked
crystalline atoms globally throughout6ps. Front starts have320 to85% of all atoms
tracked, a largest component of at least128 atoms, and at least64 more tracked
atoms6ps later. These coarse screens propose windows; atom-level PTM supplies
the actual local criteria. Existing atom labels are reused with checksums;
missing historical atom labels are explicitly recomputed with the same PTM cutoff.

Two liquid and up to16 front anchors per source span the eligible windows.
Up to512 liquid or2,048 front centers are proposed per anchor using seeded draws
without replacement. Selection interleaves sources and anchor frames within
each declared role/regime; exact source/frame/central-atom duplicates, including
original rows, are excluded. Failure to meet a quota stops preparation rather
than weakening a criterion or moving ancestors between roles. The initial8-anchor
preflight found only191 front candidates in the single eligible inner-selection
source; revision2 increased search coverage and found1,089 without changing
the scientific definition. The earlier preflight is retained, never trained.

The additions are deliberately selected using future behavior. The combined
population does not represent natural phase occupancy or unconditional nucleation
rates. Original, liquid-persistent and advancing-front held-out scores are reported
separately, with source counts and ancestor-bootstrap intervals. Only one of the
four inner-selection ancestors has qualifying front evolution; this limitation
remains explicit rather than changing the established split.

The same16-run derivative factorial is repeated: physical-response matching on/off,
learned-map Jacobian penalty on/off,1/6ps training horizon, two seeds,3,200 updates,
batch256. The unrelaxed eSEN remains frozen, the original training-only feature
mean/scale remain fixed for common metric units, and the existing300 short
physical-teacher groups are reused. No new physical MD or broad spatial dataset
is loaded. Checkpoint selection remains predictive likelihood on inner-selection
sources; birth AP remains a diagnostic.

Recipe: `configs/ni_latent_sim/regimes3x_20261010.json`.
[Dataset and execution](../../docs/ni_latent_sim.md),
[exact metrics](../../docs/metrics/ni_latent_sim.md).
