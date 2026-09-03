from __future__ import annotations

import numpy as np

from src.temporal_vamp.nested_committor import select_parent_region_atom_ids


def test_nested_region_selection_separates_cluster_interface_and_bulk() -> None:
    side = 6
    grid = np.stack(
        np.meshgrid(
            np.arange(side, dtype=np.float32),
            np.arange(side, dtype=np.float32),
            np.arange(side, dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    atom_ids = np.arange(1, grid.shape[0] + 1, dtype=np.int64)
    cluster = np.zeros(grid.shape[0], dtype=np.int64)
    cluster[:8] = 1

    nucleus, interface, background = select_parent_region_atom_ids(
        all_atom_ids=atom_ids,
        positions=grid,
        box_lengths=np.full(3, side, dtype=np.float32),
        cluster_labels=cluster,
        nucleus_center_count=4,
        interface_center_count=12,
        background_center_count=16,
        seed=7,
    )

    assert nucleus.size == 4
    assert interface.size == 12
    assert background.size == 16
    assert set(nucleus).issubset(set(atom_ids[:8]))
    assert set(nucleus).isdisjoint(interface)
    assert set(nucleus).isdisjoint(background)
    assert set(interface).isdisjoint(background)
    repeated = select_parent_region_atom_ids(
        all_atom_ids=atom_ids,
        positions=grid,
        box_lengths=np.full(3, side, dtype=np.float32),
        cluster_labels=cluster,
        nucleus_center_count=4,
        interface_center_count=12,
        background_center_count=16,
        seed=7,
    )
    assert all(
        np.array_equal(first, second)
        for first, second in zip(repeated, (nucleus, interface, background))
    )
