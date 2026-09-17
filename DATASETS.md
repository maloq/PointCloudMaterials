# Datasets: start here

**[Open the searchable dataset registry](docs/datasets/index.html)** ·
[Browse the Markdown cards](docs/datasets/README.md) ·
[Download CSV](docs/datasets/datasets.csv) · [Registry JSON](docs/datasets/registry.json)

**[Potential files, hashes and provenance](docs/datasets/potentials.md)**

The registry covers **Al, Mg, Ti, Ta, Zr and Al–Ni structures**: raw dynamics,
static structures, synthetic geometry, training caches, physical targets and
potential files. Filter by material, potential or use classification. Each card
links to producer records, array schemas, hashes and known provenance.

Read the classification before using a dataset. A complete binary is not
necessarily an independent trajectory, an accepted protocol or a training input.
Duplicate exports, incomplete preparations, removed Zr dynamics, shared ancestry,
metadata gaps and reported H200 copies remain visible.

Refresh from the repository root in `pointnet`:

```bash
python scripts/project.py datasets --refresh
```

This reads metadata, potential files, array headers and filesystem sizes. It does
not alter datasets, load full trajectories, submit jobs or fit models. Pages show
an observation at the displayed timestamp, not live job status. Unregistered
directories are listed for review.

**Register new data in [configs/datasets.json](configs/datasets.json)**, including
storage role, material, generating potential, provenance and ancestry when known.
Keep unknown values explicit. See [the registry guide](docs/datasets/GUIDE.md).
