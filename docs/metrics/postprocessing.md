# Shared post-simulation implementation

The October 7 consolidation moves descriptor extraction, lineage tracking,
catalogue/supplement publication and birth-history visualization into
`src/postprocessing`. Existing scientific thresholds, native frame selection,
event identities, fates and retrospective graph rules are unchanged. Each
export's contract includes its new implementation dependencies. Frozen exports
retain their original paths, hashes and descriptions.

Al and Ni use the same atom-overlap precursor graph kernel, periodic local
geometry, local l=6 descriptors and shape contractions. Their observation windows,
physical distance cutoffs, source selections and output interpretations remain
explicit in their protocol adapters. Precursor graph nodes remain annotations;
they are not additional birth events.

The audited PTM file-hash equivalence declaration permits reuse of already
verified labels after relocation. New extraction receipts also record the actual
classifier file hash. Unknown implementations receive a different cache identity.
See [module guide and verification](../postprocessing.md).

The subsequent acceleration pass replaces sparse component construction and
per-node sorted overlap calculations with integer union-find and count kernels.
Component numbering, sorted successor ordering, thresholds and floating-point
descriptor reductions remain unchanged. New extraction chunks store separate
detector arrays (`detector_arrays_v1`); older combined arrays remain readable.
Repacking is a verified copy into a distinct cache and plan, never a relabelling.
Raw atomic PTM is loaded lazily with its checksum and producer checks retained.
Scoped path snapshots and a single reference scan preserve source selection and
native timelines. Audited complete execution dependency sets may retain the
canonical scientific producer identity; actual execution hashes are still saved.
