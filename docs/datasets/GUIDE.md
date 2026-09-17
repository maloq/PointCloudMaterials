# Dataset registry: maintenance and interpretation

[Browser](index.html) · [Cards](README.md) · [Registration source](../../configs/datasets.json)

## Registration and refresh

`configs/datasets.json` is the authoritative list of stable IDs, relative
locations, dependencies and aliases. Per-entry `metadata` stores reviewed
descriptions and restrictions. Its `potential_registry` defines named interaction
models and exact file hashes; `registry_discovery_roots` declares shallow discovery
scopes; `remote_holdings` retains attributed reports from other machines. Machine
mount points remain in ignored `machine.local.yaml`.

Run `python scripts/project.py datasets --refresh` in `pointnet`. It updates
the JSON, CSV, offline HTML browser and individual HTML/Markdown cards here.
`records/` contains structured producer metadata, exact JSON field paths, current
binary array schemas and manifest SHA-256. `--output DIRECTORY` writes a separate
observation. The older `project.py simulations` command retains its distinct
producer-outcome export.

A registration entry looks like:

```json
{
  "root": "simulations",
  "path": "my_campaign",
  "kind": "simulation",
  "dependencies": ["potential-ti-kavousi2019"],
  "aliases": [],
  "metadata": {
    "title": "Ti example campaign",
    "materials": ["Ti"],
    "role": "raw_dynamics",
    "classification": "review_required",
    "description": "Describe the observations actually produced.",
    "potential_ids": ["ti-kavousi2019-meam"],
    "evidence": ["${dataset:my-campaign}/config.json"],
    "lineage": "Describe shared parents and independent preparations.",
    "limitations": ["Record protocol restrictions or missing information."]
  }
}
```

Classifications include `research`, `derived`, `building`, `duplicate`, `prepared`,
`incomplete_or_rejected`, `fixture`, `administrative`, `reference`, `archive`,
`mixed` and `review_required`. These are use notes, not automatic assertions of
scientific validity or current scheduler state. Materials describe actual species:
pure-Ti simulations are not Ni/Ti alloys just because their potential supports Ni.

For a potential, record its name, family, supported elements, exact LAMMPS mapping,
citation/source evidence and files as `{path, sha256}`. Use explicit dataset/storage
tokens. Refresh verifies available potential-file hashes and fails if a registered
file changed; it never silently accepts a replacement. Original static datasets
with undocumented potentials remain unknown. A later continuation's potential is
not proof of the original dataset's generating potential.

## What the fields establish

- **Available:** the registered directory exists on this machine.
- **Complete binary records:** recognized temporal/shooting manifests declaring
  completion, required arrays, matching NPY headers and a consistent timestep
  array. These can include source histories, branches, parent histories, converted
  copies and relaxed single frames. They are not independent-source counts or
  automatic training eligibility.
- **Precision and fields:** current array headers. Older protocol fields can
  retain float32 declarations after verified float16 conversion. Both statements
  remain visible with their producer context. Positions-only data do not acquire
  velocities through inference.
- **Potential identity:** reviewed annotations or checksums in potential-specific
  producer fields. An encoder checkpoint hash does not identify the physical
  generating potential.
- **Duplicates:** equality of current binary array descriptions and their
  producer-declared hashes. The refresh does not rehash terabytes of coordinates.
  Equal copies are not independent data; shared ancestry can also exist when
  hashes differ.
- **Recorded fields:** exact source JSON paths. Temperature, duration, frame
  count and states can describe different protocol stages or nested products;
  their union is not a common experimental condition.
- **Config/code references:** maintained references, not proof that a training
  run consumed those data. Consult the immutable run/cohort manifest and data-use
  audits for actual use.
- **Storage:** apparent and allocated bytes of owned regular files. Registered
  child roots are excluded from parents. Directory symlinks and tracking source
  copies are not traversed. These are neither unique payload totals across copies
  nor quota measurements.

Missing files, corrupt metadata and header disagreements are explicit integrity
issues. Rejected historical preparations may intentionally lack payloads; keep
their classifications and evidence. The observation is not atomic during active
generation. Producer state does not establish current scheduler liveness.

## Discovery and other machines

Refresh lists unregistered top-level directories in declared scopes. Review and
register useful collections; launch directories, transfer staging and empty
preparations are not silently counted as data. Do not move or remove them during
registry maintenance. Register useful child collections separately when a parent
contains multiple materials or protocols.

H200 holdings are explicitly **user-reported and not remotely verified**. No
remote path or checksum correspondence is invented, and copies are not additional
independent data. Run the same refresh on H200 with its machine roots, into a
separate output, to retain observations from both machines.

## Export definitions and testing

`datasets.csv` is an operational inventory, not a research metric table. Its
`complete_binary_records`, `allocated_bytes`, `available`, `potential_ids`,
`missing_metadata` and `integrity_issues` follow the definitions above; lists use
semicolons. `registry.json` records catalog/implementation hashes and the timestamp.

Tests cover stale precision, missing arrays, positions-only data, separation of
encoder and generating potential identities, nested ownership, duplicate groups,
missing locations, discovery, changed potential files and HTML escaping.
