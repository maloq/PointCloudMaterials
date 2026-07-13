from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from mace import data as mace_data
from mace.calculators import MACECalculator
from mace.tools import torch_geometric


class VerletSkinMACECalculator(MACECalculator):
    """MACE calculator with an exact, displacement-triggered neighbor-graph cache.

    The graph is built to ``r_max + neighbor_skin_A``. MACE's own radial cutoff
    remains ``r_max``, so the additional edges have zero energy and force. The
    cached graph is discarded before atomic motion or cell contraction could
    bring an omitted edge inside ``r_max``.
    """

    def __init__(self, *args: object, neighbor_skin_A: float, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.neighbor_skin_A = float(neighbor_skin_A)
        self.graph_rebuild_count = 0
        self.graph_reuse_count = 0
        self._cached_batch = None
        self._reference_cell_A: np.ndarray | None = None
        self._reference_scaled_positions: np.ndarray | None = None

    def _atoms_to_batch(self, atoms: Atoms):
        positions_A = np.asarray(atoms.positions, dtype=np.float64)
        cell_A = np.asarray(atoms.cell.array, dtype=np.float64)
        if self._cached_batch is None or not self._graph_is_valid(positions_A, cell_A):
            return self._rebuild_graph(atoms, positions_A, cell_A)

        self.graph_reuse_count += 1
        batch = self._cached_batch
        batch["positions"].copy_(
            torch.as_tensor(
                positions_A,
                dtype=batch["positions"].dtype,
                device=batch["positions"].device,
            )
        )
        batch["cell"].copy_(
            torch.as_tensor(
                cell_A,
                dtype=batch["cell"].dtype,
                device=batch["cell"].device,
            )
        )
        batch["shifts"].copy_(torch.matmul(batch["unit_shifts"], batch["cell"]))
        return batch

    def _graph_is_valid(self, positions_A: np.ndarray, cell_A: np.ndarray) -> bool:
        reference_cell_A = self._reference_cell_A
        reference_scaled_positions = self._reference_scaled_positions
        if reference_cell_A is None or reference_scaled_positions is None:
            return False

        deformation = np.linalg.solve(reference_cell_A, cell_A)
        minimum_stretch = float(np.linalg.svd(deformation, compute_uv=False)[-1])
        affine_reference_positions_A = reference_scaled_positions @ cell_A
        maximum_nonaffine_displacement_A = float(
            np.linalg.norm(positions_A - affine_reference_positions_A, axis=1).max()
        )
        shortest_omitted_distance_A = minimum_stretch * (
            self.r_max + self.neighbor_skin_A
        ) - 2.0 * maximum_nonaffine_displacement_A
        return shortest_omitted_distance_A > self.r_max

    def _rebuild_graph(
        self,
        atoms: Atoms,
        positions_A: np.ndarray,
        cell_A: np.ndarray,
    ):
        self.arrays_keys.update({self.charges_key: "charges"})
        keyspec = mace_data.KeySpecification(
            info_keys=self.info_keys,
            arrays_keys=self.arrays_keys,
        )
        configuration = mace_data.config_from_atoms(
            atoms,
            key_specification=keyspec,
            head_name=self.head,
        )
        atomic_data = mace_data.AtomicData.from_config(
            configuration,
            z_table=self.z_table,
            cutoff=self.r_max + self.neighbor_skin_A,
            heads=self.available_heads,
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        self._cached_batch = next(iter(data_loader)).to(self.device)
        self._reference_cell_A = cell_A.copy()
        self._reference_scaled_positions = np.linalg.solve(
            cell_A.T,
            positions_A.T,
        ).T
        self.graph_rebuild_count += 1
        return self._cached_batch
