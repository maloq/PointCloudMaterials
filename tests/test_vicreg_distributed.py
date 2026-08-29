from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from src.training_methods.contrastive_learning.vicreg import VICRegLoss


_WEIGHT = torch.tensor(
    [
        [0.20, -0.10, 0.30],
        [-0.40, 0.50, 0.10],
        [0.70, -0.20, -0.30],
        [0.10, 0.60, -0.50],
    ],
    dtype=torch.float32,
)

_VIEW_A_BY_RANK = (
    torch.tensor(
        [
            [0.10, 0.20, 0.30],
            [0.40, -0.50, 0.60],
        ],
        dtype=torch.float32,
    ),
    torch.tensor(
        [
            [-0.30, 0.80, 0.20],
            [0.90, -0.10, -0.70],
            [0.50, 0.40, -0.20],
        ],
        dtype=torch.float32,
    ),
)

_VIEW_B_BY_RANK = (
    torch.tensor(
        [
            [0.20, 0.10, 0.35],
            [0.35, -0.45, 0.70],
        ],
        dtype=torch.float32,
    ),
    torch.tensor(
        [
            [-0.25, 0.75, 0.30],
            [0.85, -0.05, -0.65],
            [0.55, 0.35, -0.15],
        ],
        dtype=torch.float32,
    ),
)


def _vicreg_loss() -> VICRegLoss:
    cfg = OmegaConf.create(
        {
            "vicreg_enabled": True,
            "vicreg_weight": 1.0,
            "vicreg_embed_dim": 4,
            "vicreg_projector_mode": "identity",
            "vicreg_sim_coeff": 25.0,
            "vicreg_std_coeff": 25.0,
            "vicreg_cov_coeff": 1.0,
        }
    )
    return VICRegLoss.from_config(cfg, input_dim=4)


def _linear_model() -> torch.nn.Linear:
    model = torch.nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(_WEIGHT)
    return model


def _ddp_vicreg_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    result_directory: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = DistributedDataParallel(_linear_model())
        local_views = torch.cat((_VIEW_A_BY_RANK[rank], _VIEW_B_BY_RANK[rank]), dim=0)
        local_embeddings = model(local_views)
        local_batch_size = _VIEW_A_BY_RANK[rank].shape[0]
        z_a, z_b = local_embeddings.split(local_batch_size, dim=0)

        loss, _ = _vicreg_loss()._loss(z_a, z_b)
        loss.backward()

        torch.save(
            {
                "loss": loss.detach(),
                "gradient": model.module.weight.grad.detach(),
            },
            Path(result_directory) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_process_ddp_gradient_matches_single_process_global_batch(tmp_path: Path) -> None:
    world_size = 2
    rendezvous_path = tmp_path / "distributed_init"
    result_directory = tmp_path / "results"
    result_directory.mkdir()

    mp.spawn(
        _ddp_vicreg_worker,
        args=(world_size, str(rendezvous_path), str(result_directory)),
        nprocs=world_size,
        join=True,
    )

    global_model = _linear_model()
    global_a = torch.cat(_VIEW_A_BY_RANK, dim=0)
    global_b = torch.cat(_VIEW_B_BY_RANK, dim=0)
    global_embeddings = global_model(torch.cat((global_a, global_b), dim=0))
    z_a, z_b = global_embeddings.split(global_a.shape[0], dim=0)
    reference_loss, _ = _vicreg_loss()._loss(z_a, z_b)
    reference_loss.backward()

    for rank in range(world_size):
        rank_result = torch.load(
            result_directory / f"rank_{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        torch.testing.assert_close(rank_result["loss"], reference_loss.detach())
        torch.testing.assert_close(
            rank_result["gradient"],
            global_model.weight.grad,
            rtol=1e-5,
            atol=1e-6,
        )
