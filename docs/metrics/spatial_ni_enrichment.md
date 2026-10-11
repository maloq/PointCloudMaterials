# Ni crystal and interface spatial enrichment

The parent is the completed 1,320,960-row spatial fitting set with 264,192 Ni
rows. The addition is `round(0.20 * 264192) = 52,838` observations, split equally
into 26,419 crystal interiors and 26,419 interfaces. All parent training and
held-out raw observations/targets are retained in order, with ancestry and
original row mappings. The new training total is 1,373,798; Ni contributes
317,030 (23.0769%). Original selection and test are merged into **evaluation**,
retaining their original-role metadata. Their 8,192 Ni rows receive 1,638
additional observations (819 per category), yielding 9,830 Ni evaluation rows
and 248,081 evaluation rows overall. The separate 16,848 Al calibration rows
remain preserved, outside the merged pretraining evaluation set and training.

The parent's completed Ni sources are eligible within their original ancestry
groups. Training and evaluation additions are sampled separately. Actual saved
steps join the original 3-ps-spaced candidate frames to native 1-ps PTM monitoring,
without interpolation. Frames with at least 10% global PTM crystallinity are
eligible. At most 16 evenly spaced frame indices per source cover those eligible
frames. This is an explicit phase-enriched spatial sample, not a representative
MD population or a nucleus-prediction cohort.

Fresh full-cell periodic PTM classifies the saved, unrelaxed coordinates, with
RMSD cutoff 0.1. FCC, HCP and BCC labels count as crystalline. For every atom,
the local crystalline fraction is computed over its nearest 80 atoms (including
itself), restricted to normalized distance <8. Ni uses the unchanged parent
length normalization; no source- or phase-specific rescaling is fitted.

- Crystal-interior candidates have a crystalline central atom and local
  crystalline fraction >=0.80.
- Disordered reference centers have local crystalline fraction <=0.05.
- Interface candidates have local crystalline fraction >=0.20 and <0.80,
  and lie within normalized distance <=8 of both a crystal-interior center and
  a disordered reference center, using periodic distances.

The interface rule identifies structural crystal/disordered boundaries, and is
not an independent thermodynamic liquid-phase measurement. Thermal PTM misses,
grain structure and the finite observation radius can affect this operational
classification. It does not assert event origin, growth rate or nucleus fate.

All atom centers are considered in selected frames. Exact source/frame/atom
identities already in the parent's corresponding role are excluded. Within each
category and combined role, a seeded uniform draw without replacement selects
the requested rows from eligible candidates. No held-out source contributes
training candidates or fitted thresholds. Categories and PTM are sampling/audit metadata,
not encoder inputs or additional decoder targets.

The same 198 targets (40 bond-order, 45 CNA, 113 TDA) are calculated for added
rows with safe native alpha persistence. Existing raw target values are copied
unchanged. Target mean, variance, scale floor 1e-4 and active mask SD >=1e-4 are
refitted on combined training rows only. The spatial loss retains equal thirds
for the three families, with TDA split equally over its 18 groups and then
their active coordinates. Held-out rows receive this updated training transform.

`population.csv` counts observations by combined role and material, not
independent trajectories. Frame classification packets, candidate inventories,
selected category arrays, manifests, code and normalization have bound hashes.
The workflow prepares data only and launches no encoder training or W&B run.
