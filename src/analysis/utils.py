from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.data_utils.data_load import PointCloudDataset
from src.utils.evaluation_metrics import random_rotation_matrix


@runtime_checkable
class AnalyzableModel(Protocol):
    """Protocol for models that can be used with the analysis pipeline.

    Any model whose ``__call__`` returns
    ``(invariant_latent, model_latent, equivariant_latent | None)``
    satisfies this protocol.  ``VICRegModule`` already conforms.
    """

    def __call__(
        self, pc: torch.Tensor
    ) -> tuple[torch.Tensor | None, Any, torch.Tensor | None]: ...

    def eval(self) -> "AnalyzableModel": ...

    def to(self, device: str, **kwargs: Any) -> "AnalyzableModel": ...


def cap_cluster_labels(
    labels: np.ndarray,
    *,
    max_clusters: int,
    other_label: int = -1,
) -> np.ndarray:
    """Keep the largest `max_clusters` labels; collapse the rest to `other_label`."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) <= max_clusters:
        return labels
    order = np.argsort(counts)[::-1]
    keep = set(unique[order[:max_clusters]].tolist())
    capped = labels.copy()
    mask = ~np.isin(capped, list(keep))
    capped[mask] = other_label
    return capped


def _extract_pc_phase_coords(
    batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Extract point cloud, phase (class_id), coords, and instance_id from a batch.

    Returns (pc, phase, coords, instance_id).
    """
    pc = batch["points"]
    phase = batch.get("class_id", None)
    coords = batch.get("coords", None)
    instance_id = batch.get("instance_id", None)
    return pc, phase, coords, instance_id


def _extract_pc_and_phase(batch: Any) -> Tuple[torch.Tensor, torch.Tensor | None]:
    pc, phase, _, _ = _extract_pc_phase_coords(batch)
    return pc, phase


def _extract_optional_flat_tensor(batch: Any, key: str) -> torch.Tensor | None:
    value = batch.get(key, None)
    if value is None:
        return None
    return value.detach().view(-1).cpu()


def _prepare_pointcloud_batch_for_model(
    pc: torch.Tensor,
    *,
    temporal_sequence_mode: str = "static_anchor",
    temporal_static_frame_index: int | None = 0,
) -> torch.Tensor:
    if pc.ndim == 3:
        if pc.shape[-1] != 3:
            raise ValueError(
                "Static point-cloud batches must have shape (B, N, 3), "
                f"got {tuple(pc.shape)}."
            )
        return pc
    if pc.ndim != 4 or pc.shape[-1] != 3:
        raise ValueError(
            "Analysis inference supports point-cloud batches with shape "
            "(B, N, 3) or temporal batches with shape (B, T, N, 3), "
            f"got {tuple(pc.shape)}."
        )

    mode = str(temporal_sequence_mode).strip().lower()
    if mode == "static_anchor":
        frame_index = _resolve_temporal_frame_index(
            sequence_length=int(pc.shape[1]),
            frame_index=temporal_static_frame_index,
            field_name="temporal_static_frame_index",
        )
        return pc[:, frame_index, :, :]
    if mode == "temporal":
        return pc
    raise ValueError(
        "temporal_sequence_mode must be one of ['static_anchor', 'temporal'], "
        f"got {temporal_sequence_mode!r}."
    )


def _resolve_temporal_frame_index(
    *,
    sequence_length: int,
    frame_index: int | None,
    field_name: str,
) -> int:
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}.")
    resolved = 0 if frame_index is None else int(frame_index)
    if resolved < 0:
        resolved += int(sequence_length)
    if resolved < 0 or resolved >= int(sequence_length):
        raise ValueError(
            f"{field_name} is out of range for temporal sequence length {sequence_length}. "
            f"Got {frame_index!r}, resolved_index={resolved}."
        )
    return resolved


def _prepare_coords_batch_for_model(
    batch: Any,
    coords: torch.Tensor | None,
    *,
    temporal_sequence_mode: str = "static_anchor",
    temporal_static_frame_index: int | None = 0,
) -> torch.Tensor | None:
    if str(temporal_sequence_mode).strip().lower() != "static_anchor":
        return coords
    center_positions = batch.get("center_positions", None)
    if center_positions is None:
        return coords
    if center_positions.ndim != 3 or center_positions.shape[-1] != 3:
        raise ValueError(
            "Temporal analysis expects batch['center_positions'] with shape (B, T, 3), "
            f"got {tuple(center_positions.shape)}."
        )
    frame_index = _resolve_temporal_frame_index(
        sequence_length=int(center_positions.shape[1]),
        frame_index=temporal_static_frame_index,
        field_name="temporal_static_frame_index",
    )
    return center_positions[:, frame_index, :]


def _unwrap_dataset(dataset: Any) -> Any:
    """Unwrap nested dataset wrappers (Subset, etc.) to get the underlying dataset."""
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _unwrap_subset_indices(dataset: Any) -> Tuple[Any, list[int] | None]:
    indices: list[int] | None = None
    while isinstance(dataset, torch.utils.data.Subset):
        if indices is None:
            indices = list(dataset.indices)
        else:
            indices = [indices[i] for i in dataset.indices]
        dataset = dataset.dataset
    return dataset, indices


def _try_reuse_full_pointcloud_dataset(dm: Any) -> PointCloudDataset | None:
    train_dataset = getattr(dm, "train_dataset", None)
    val_dataset = getattr(dm, "val_dataset", None)
    if train_dataset is None or val_dataset is None:
        return None

    train_base, _ = _unwrap_subset_indices(train_dataset)
    val_base, _ = _unwrap_subset_indices(val_dataset)
    if train_base is not val_base:
        return None
    if not isinstance(train_base, PointCloudDataset):
        return None
    return train_base


def build_static_coords_dataloader(
    cfg: DictConfig,
    dm: Any,
    use_train_data: bool,
    use_full_dataset: bool = False,
    prefer_existing_full_dataset: bool = False,
    batch_size: int | None = None,
) -> torch.utils.data.DataLoader:
    data_cfg = cfg.data
    auto_cutoff_cfg = getattr(data_cfg, "auto_cutoff", None)
    if isinstance(auto_cutoff_cfg, (DictConfig, ListConfig)):
        auto_cutoff_cfg = OmegaConf.to_container(auto_cutoff_cfg, resolve=True)

    dataset_kwargs = dict(
        radius=getattr(data_cfg, "radius", 8),
        sample_type=getattr(data_cfg, "sample_type", "regular"),
        overlap_fraction=getattr(data_cfg, "overlap_fraction", 0.0),
        n_samples=getattr(data_cfg, "n_samples", 1000),
        num_points=getattr(data_cfg, "num_points", 100),
        drop_edge_samples=bool(getattr(data_cfg, "drop_edge_samples", True)),
        edge_drop_layers=getattr(data_cfg, "edge_drop_layers", None),
        return_coords=True,
        pre_normalize=getattr(data_cfg, "pre_normalize", True),
        normalize=getattr(data_cfg, "normalize", True),
        sampling_method=getattr(data_cfg, "sampling_method", "drop_farthest"),
        auto_cutoff_config=auto_cutoff_cfg,
    )

    full_dataset: PointCloudDataset | None = None
    if use_full_dataset and prefer_existing_full_dataset:
        full_dataset = _try_reuse_full_pointcloud_dataset(dm)
        if full_dataset is not None and not bool(getattr(full_dataset, "return_coords", False)):
            full_dataset = None

    if full_dataset is None:
        data_sources = getattr(data_cfg, "data_sources", None)
        data_files = getattr(data_cfg, "data_files", None)
        if data_sources:
            if isinstance(data_sources, (DictConfig, ListConfig)):
                data_sources = OmegaConf.to_container(data_sources, resolve=True)
            if not isinstance(data_sources, list) or not data_sources:
                raise ValueError(
                    "data_sources must be a non-empty list when provided in cfg.data."
                )
            full_dataset = PointCloudDataset(
                data_sources=data_sources,
                **dataset_kwargs,
            )
        elif data_files:
            file_list = data_files
            if isinstance(file_list, ListConfig):
                file_list = list(file_list)
            if isinstance(file_list, str):
                file_list = [file_list]
            if not file_list:
                raise ValueError("data_files is empty in cfg.data.")
            data_path = getattr(data_cfg, "data_path", None)
            if not data_path:
                raise ValueError(
                    "data_path is required when using data_files in build_static_coords_dataloader."
                )
            full_dataset = PointCloudDataset(
                root=data_path,
                data_files=file_list,
                **dataset_kwargs,
            )
        else:
            raise ValueError(
                "No static-data inputs configured. Provide either cfg.data.data_sources "
                "or cfg.data.data_files + cfg.data.data_path."
            )

    dataset = full_dataset
    if not use_full_dataset:
        target_dataset = dm.train_dataset if use_train_data else dm.test_dataset
        _, indices = _unwrap_subset_indices(target_dataset)
        if indices is not None:
            dataset = torch.utils.data.Subset(full_dataset, indices)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.batch_size) if batch_size is None else int(batch_size),
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def _torch_dtype_to_numpy_dtype(dtype: torch.dtype) -> np.dtype:
    dtype_map = {
        torch.float64: np.dtype(np.float64),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float16),
        torch.bfloat16: np.dtype(np.float32),
        torch.int64: np.dtype(np.int64),
        torch.int32: np.dtype(np.int32),
        torch.int16: np.dtype(np.int16),
        torch.int8: np.dtype(np.int8),
        torch.uint8: np.dtype(np.uint8),
        torch.bool: np.dtype(np.bool_),
    }
    if dtype not in dtype_map:
        raise TypeError(f"Unsupported tensor dtype for inference cache storage: {dtype!r}.")
    return dtype_map[dtype]


def _tensor_slice_to_numpy_for_storage(tensor: torch.Tensor, take: int) -> np.ndarray:
    # The returned NumPy view is consumed immediately by cache writers.  This
    # must be a blocking CPU copy; otherwise a preallocated NumPy cache can read
    # from the tensor before an async CUDA transfer has completed.
    tensor_cpu = tensor.detach()[:take].to("cpu", non_blocking=False).contiguous()
    if tensor_cpu.dtype == torch.bfloat16:
        tensor_cpu = tensor_cpu.to(torch.float32)
    return tensor_cpu.numpy()


def _expected_inference_sample_count(
    *,
    dataloader: torch.utils.data.DataLoader,
    max_batches: int | None,
    max_samples: int | None,
) -> int | None:
    dataset = getattr(dataloader, "dataset", None)
    dataset_len = None if dataset is None else int(len(dataset))
    if max_batches is None:
        if dataset_len is None:
            return None
        return dataset_len if max_samples is None else min(dataset_len, int(max_samples))
    batch_size = getattr(dataloader, "batch_size", None)
    if batch_size is None:
        return None if max_samples is None else int(max_samples)
    batch_cap = int(max_batches) * int(batch_size)
    if dataset_len is not None:
        batch_cap = min(batch_cap, dataset_len)
    return batch_cap if max_samples is None else min(batch_cap, int(max_samples))


def gather_inference_batches(
    model: AnalyzableModel,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int | None = 4,
    max_samples_total: int | None = None,
    collect_coords: bool = False,
    seed_base: int | None = 123,
    progress_every_batches: int = 25,
    verbose: bool = False,
    temporal_sequence_mode: str = "static_anchor",
    temporal_static_frame_index: int | None = 0,
) -> Dict[str, np.ndarray]:
    """Collect inputs and latents from batches."""
    inv_latents, eq_latents, phases, coords_list, instance_ids = [], [], [], [], []
    anchor_frame_indices = []
    collected = 0
    max_samples = None if max_samples_total is None else max(1, int(max_samples_total))
    every = max(1, int(progress_every_batches))
    expected_total_samples = _expected_inference_sample_count(
        dataloader=dataloader,
        max_batches=max_batches,
        max_samples=max_samples,
    )
    use_preallocated_arrays = expected_total_samples is not None
    preallocated: dict[str, np.ndarray] | None = None

    def _copy_tensor_to_numpy(
        *,
        key: str,
        tensor: torch.Tensor,
        take: int,
    ) -> None:
        if preallocated is None:
            raise RuntimeError(
                "Internal error: preallocated inference arrays are not initialized."
            )
        arr = preallocated[key]
        end = collected + int(take)
        if end > int(arr.shape[0]):
            raise RuntimeError(
                "Preallocated inference cache is too small. "
                f"key={key}, requested_end={end}, capacity={int(arr.shape[0])}, "
                f"expected_total_samples={expected_total_samples}."
            )
        arr[collected:end] = _tensor_slice_to_numpy_for_storage(tensor, take)

    rng_state = None if seed_base is None else _seed_inference_rng_once(seed_base, device)
    try:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                pc, phase, coords, instance_id = _extract_pc_phase_coords(batch)
                anchor_frame_index = _extract_optional_flat_tensor(batch, "anchor_frame_index")
                pc = _prepare_pointcloud_batch_for_model(
                    pc,
                    temporal_sequence_mode=temporal_sequence_mode,
                    temporal_static_frame_index=temporal_static_frame_index,
                )
                coords = _prepare_coords_batch_for_model(
                    batch,
                    coords,
                    temporal_sequence_mode=temporal_sequence_mode,
                    temporal_static_frame_index=temporal_static_frame_index,
                )
                pc = pc.to(device, non_blocking=True)

                z_inv_contrastive, _, eq_z = model(pc)
                if z_inv_contrastive is None:
                    raise RuntimeError(
                        "Analysis inference requires the model forward pass to return an "
                        "invariant latent as its first output, but got None. "
                        f"batch_idx={batch_idx}, input_shape={tuple(pc.shape)}, "
                        f"model_type={type(model)!r}."
                    )

                batch_size = int(z_inv_contrastive.shape[0])
                take = batch_size
                if max_samples is not None:
                    remaining = max_samples - collected
                    if remaining <= 0:
                        break
                    take = min(take, remaining)

                if take <= 0:
                    break

                if use_preallocated_arrays and preallocated is None:
                    preallocated = {
                        "inv_latents": np.empty(
                            (
                                int(expected_total_samples),
                                *tuple(z_inv_contrastive.shape[1:]),
                            ),
                            dtype=_torch_dtype_to_numpy_dtype(z_inv_contrastive.dtype),
                        )
                    }
                    if eq_z is not None:
                        preallocated["eq_latents"] = np.empty(
                            (int(expected_total_samples), *tuple(eq_z.shape[1:])),
                            dtype=_torch_dtype_to_numpy_dtype(eq_z.dtype),
                        )

                if preallocated is None:
                    inv_latents.append(z_inv_contrastive.detach()[:take].cpu())
                else:
                    _copy_tensor_to_numpy(
                        key="inv_latents",
                        tensor=z_inv_contrastive,
                        take=take,
                    )
                if collect_coords:
                    if coords is None:
                        raise RuntimeError(
                            "gather_inference_batches(..., collect_coords=True) received a batch "
                            f"without coordinates at batch_idx={batch_idx}. "
                            f"Batch type: {type(batch)!r}."
                        )
                    coords_t = coords.detach().cpu()
                    if coords_t.ndim == 1:
                        coords_t = coords_t.unsqueeze(0)
                    if coords_t.ndim != 2 or coords_t.shape[1] != 3:
                        raise ValueError(
                            "Expected center coordinates with shape [batch, 3] when "
                            f"collect_coords=True, got shape={tuple(coords_t.shape)} "
                            f"at batch_idx={batch_idx}."
                        )
                    if preallocated is None:
                        coords_list.append(coords_t[:take])
                    else:
                        if "coords" not in preallocated:
                            preallocated["coords"] = np.empty(
                                (int(expected_total_samples), 3),
                                dtype=_torch_dtype_to_numpy_dtype(coords_t.dtype),
                            )
                        end = collected + int(take)
                        preallocated["coords"][collected:end] = (
                            _tensor_slice_to_numpy_for_storage(coords_t, take)
                        )
                if eq_z is not None:
                    if preallocated is None:
                        eq_latents.append(eq_z.detach()[:take].cpu())
                    else:
                        if "eq_latents" not in preallocated:
                            preallocated["eq_latents"] = np.empty(
                                (int(expected_total_samples), *tuple(eq_z.shape[1:])),
                                dtype=_torch_dtype_to_numpy_dtype(eq_z.dtype),
                            )
                        _copy_tensor_to_numpy(
                            key="eq_latents",
                            tensor=eq_z,
                            take=take,
                        )
                elif preallocated is not None and "eq_latents" in preallocated:
                    raise RuntimeError(
                        "Model returned equivariant latents for an earlier inference batch "
                        f"but returned None at batch_idx={batch_idx}."
                    )
                if phase is not None:
                    phase_t = phase.detach().view(-1).cpu()
                    if preallocated is None:
                        phases.append(phase_t[:take])
                    else:
                        if "phases" not in preallocated:
                            preallocated["phases"] = np.empty(
                                (int(expected_total_samples),),
                                dtype=_torch_dtype_to_numpy_dtype(phase_t.dtype),
                            )
                        end = collected + int(take)
                        preallocated["phases"][collected:end] = (
                            _tensor_slice_to_numpy_for_storage(phase_t, take)
                        )
                elif preallocated is not None and "phases" in preallocated:
                    raise RuntimeError(
                        "Inference batches provided class_id for an earlier batch "
                        f"but omitted it at batch_idx={batch_idx}."
                    )
                if instance_id is not None:
                    instance_id_t = instance_id.detach().view(-1).cpu()
                    if preallocated is None:
                        instance_ids.append(instance_id_t[:take])
                    else:
                        if "instance_ids" not in preallocated:
                            preallocated["instance_ids"] = np.empty(
                                (int(expected_total_samples),),
                                dtype=_torch_dtype_to_numpy_dtype(instance_id_t.dtype),
                            )
                        end = collected + int(take)
                        preallocated["instance_ids"][collected:end] = (
                            _tensor_slice_to_numpy_for_storage(instance_id_t, take)
                        )
                elif preallocated is not None and "instance_ids" in preallocated:
                    raise RuntimeError(
                        "Inference batches provided instance_id for an earlier batch "
                        f"but omitted it at batch_idx={batch_idx}."
                    )
                if anchor_frame_index is not None:
                    if preallocated is None:
                        anchor_frame_indices.append(anchor_frame_index[:take])
                    else:
                        if "anchor_frame_indices" not in preallocated:
                            preallocated["anchor_frame_indices"] = np.empty(
                                (int(expected_total_samples),),
                                dtype=_torch_dtype_to_numpy_dtype(anchor_frame_index.dtype),
                            )
                        end = collected + int(take)
                        preallocated["anchor_frame_indices"][collected:end] = (
                            _tensor_slice_to_numpy_for_storage(anchor_frame_index, take)
                        )
                elif preallocated is not None and "anchor_frame_indices" in preallocated:
                    raise RuntimeError(
                        "Inference batches provided anchor_frame_index for an earlier batch "
                        f"but omitted it at batch_idx={batch_idx}."
                    )

                collected += int(take)
                if verbose and ((batch_idx + 1) % every == 0 or take != batch_size):
                    print(
                        f"[analysis][collect] batch={batch_idx + 1} "
                        f"samples={collected}"
                        + (f"/{max_samples}" if max_samples is not None else "")
                    )

                if max_samples is not None and collected >= max_samples:
                    if verbose:
                        print(f"[analysis][collect] reached sample cap: {collected}")
                    break
    finally:
        if rng_state is not None:
            _restore_inference_rng(rng_state)

    def _cat(tensors):
        return torch.cat(tensors, dim=0).numpy() if tensors else np.empty((0,))

    def _cat_coords(tensors):
        if not tensors:
            return np.empty((0, 3), dtype=np.float32)
        return torch.cat(tensors, dim=0).numpy()

    if preallocated is not None:
        return {
            "inv_latents": preallocated["inv_latents"][:collected],
            "eq_latents": preallocated.get(
                "eq_latents",
                np.empty((0,), dtype=np.float32),
            )[:collected if "eq_latents" in preallocated else 0],
            "phases": preallocated.get(
                "phases",
                np.empty((0,), dtype=np.int64),
            )[:collected if "phases" in preallocated else 0],
            "coords": preallocated.get(
                "coords",
                np.empty((0, 3), dtype=np.float32),
            )[:collected if "coords" in preallocated else 0],
            "instance_ids": preallocated.get(
                "instance_ids",
                np.empty((0,), dtype=np.int64),
            )[:collected if "instance_ids" in preallocated else 0],
            "anchor_frame_indices": preallocated.get(
                "anchor_frame_indices",
                np.empty((0,), dtype=np.int64),
            )[:collected if "anchor_frame_indices" in preallocated else 0],
        }

    return {
        "inv_latents": _cat(inv_latents),
        "eq_latents": _cat(eq_latents),
        "phases": _cat(phases),
        "coords": _cat_coords(coords_list),
        "instance_ids": _cat(instance_ids),
        "anchor_frame_indices": _cat(anchor_frame_indices),
    }


def _sample_indices(num_samples: int, max_samples: int | None) -> np.ndarray:
    if max_samples is None or num_samples <= max_samples:
        return np.arange(num_samples)
    rng = np.random.default_rng(0)
    return rng.choice(num_samples, size=max_samples, replace=False)


def _seed_inference_rng_once(
    seed: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.device | None, torch.Tensor | None]:
    """Seed RNG once for a whole inference pass, preserving caller RNG state."""
    cpu_state = torch.get_rng_state()
    cuda_device = _resolve_cuda_rng_device(device)
    cuda_state = (
        torch.cuda.get_rng_state(device=cuda_device)
        if cuda_device is not None
        else None
    )
    torch.random.default_generator.manual_seed(int(seed))
    if cuda_device is not None:
        with torch.cuda.device(cuda_device):
            torch.cuda.manual_seed(int(seed))
    return cpu_state, cuda_device, cuda_state


def _restore_inference_rng(
    rng_state: tuple[torch.Tensor, torch.device | None, torch.Tensor | None],
) -> None:
    cpu_state, cuda_device, cuda_state = rng_state
    torch.set_rng_state(cpu_state)
    if cuda_device is not None and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device=cuda_device)


def _resolve_cuda_rng_device(device: str | torch.device) -> torch.device | None:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Cannot seed CUDA inference RNG for device={device!r}: CUDA is not available."
        )
    device_index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
    return torch.device(f"cuda:{device_index}")


def _seeded_forward(model: AnalyzableModel, pc: torch.Tensor, seed: int):
    """Run forward pass with fixed random seed while preserving global RNG state."""
    cpu_state = torch.get_rng_state()
    cuda_device = (
        _resolve_cuda_rng_device(torch.device(f"cuda:{pc.get_device()}"))
        if pc.is_cuda
        else None
    )
    cuda_state = (
        torch.cuda.get_rng_state(device=cuda_device)
        if cuda_device is not None
        else None
    )
    torch.random.default_generator.manual_seed(int(seed))
    if pc.is_cuda:
        with torch.cuda.device(cuda_device):
            torch.cuda.manual_seed(int(seed))
    try:
        return model(pc)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device=cuda_device)


def evaluate_latent_equivariance(
    model: AnalyzableModel,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int = 2,
    temporal_sequence_mode: str = "static_anchor",
    temporal_static_frame_index: int | None = 0,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate equivariance in encoder outputs (seeded vs unseeded)."""
    eq_errors_seeded = []
    eq_errors_unseeded = []
    determinism_errors = []
    identity_errors = []

    model.eval()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            pc, _ = _extract_pc_and_phase(batch)
            pc = _prepare_pointcloud_batch_for_model(
                pc,
                temporal_sequence_mode=temporal_sequence_mode,
                temporal_static_frame_index=temporal_static_frame_index,
            )
            pc = pc.to(device, non_blocking=True)
            batch_size = pc.shape[0]

            rots = torch.stack(
                [
                    torch.tensor(random_rotation_matrix(), device=device, dtype=pc.dtype)
                    for _ in range(batch_size)
                ]
            )
            pc_rot = torch.einsum("bij,bnj->bni", rots, pc)
            identity = torch.eye(3, device=device, dtype=pc.dtype).unsqueeze(0).expand(
                batch_size, -1, -1
            )

            seed = 42 + batch_idx

            # Determinism and identity checks share the same seeded forwards.
            _, _, eq_z_seed_1 = _seeded_forward(model, pc, seed)
            _, _, eq_z_seed_2 = _seeded_forward(model, pc, seed)
            if eq_z_seed_1 is not None and eq_z_seed_2 is not None:
                det_diff = torch.linalg.norm(eq_z_seed_1 - eq_z_seed_2, dim=-1).mean(dim=1)
                determinism_errors.extend(det_diff.detach().cpu().numpy().tolist())
                expected_id = torch.einsum("bij,bcj->bci", identity, eq_z_seed_1)
                id_rel = torch.linalg.norm(eq_z_seed_2 - expected_id, dim=-1) / torch.linalg.norm(
                    expected_id, dim=-1
                ).clamp_min(1e-6)
                identity_errors.extend(id_rel.mean(dim=1).detach().cpu().numpy().tolist())

            # Random rotation WITH seed control
            _, _, eq_z_rot = _seeded_forward(model, pc_rot, seed)
            if eq_z_seed_1 is not None and eq_z_rot is not None:
                expected_eq = torch.einsum("bij,bcj->bci", rots, eq_z_seed_1)
                rel = torch.linalg.norm(eq_z_rot - expected_eq, dim=-1) / torch.linalg.norm(
                    expected_eq, dim=-1
                ).clamp_min(1e-6)
                eq_errors_seeded.extend(rel.mean(dim=1).detach().cpu().numpy().tolist())

            # Random rotation WITHOUT seed control
            _, _, eq_z_uns = model(pc)
            _, _, eq_z_rot_uns = model(pc_rot)
            if eq_z_uns is not None and eq_z_rot_uns is not None:
                expected_eq_uns = torch.einsum("bij,bcj->bci", rots, eq_z_uns)
                rel_uns = torch.linalg.norm(eq_z_rot_uns - expected_eq_uns, dim=-1) / torch.linalg.norm(
                    expected_eq_uns, dim=-1
                ).clamp_min(1e-6)
                eq_errors_unseeded.extend(rel_uns.mean(dim=1).detach().cpu().numpy().tolist())

    eq_seeded = np.asarray(eq_errors_seeded)
    eq_unseeded = np.asarray(eq_errors_unseeded)
    det_arr = np.asarray(determinism_errors)
    id_arr = np.asarray(identity_errors)

    print("\n" + "=" * 60)
    print("EQUIVARIANCE DIAGNOSTICS")
    print("=" * 60)
    print("1. DETERMINISM (same input, same seed):")
    print(f"   Mean abs diff: {det_arr.mean():.6e}" if det_arr.size else "   N/A")
    print("   Should be ~0 if model is deterministic with fixed seed")
    print()
    print("2. IDENTITY ROTATION (R=I, same seed):")
    print(f"   Mean rel error: {id_arr.mean():.6e}" if id_arr.size else "   N/A")
    print("   Should be ~0 (sanity check)")
    print()
    print("3. RANDOM ROTATION (WITH seed control):")
    print(f"   Latent rel error: {eq_seeded.mean():.4f}" if eq_seeded.size else "   N/A")
    print("   Tests true equivariance (FPS initialized same way)")
    print()
    print("4. RANDOM ROTATION (WITHOUT seed control):")
    print(f"   Latent rel error: {eq_unseeded.mean():.4f}" if eq_unseeded.size else "   N/A")
    print("   Includes non-determinism from FPS random init")
    print("=" * 60)

    nondet = float(eq_unseeded.mean() - eq_seeded.mean()) if eq_seeded.size and eq_unseeded.size else float("nan")
    if eq_seeded.size and eq_unseeded.size:
        print(f"\nNON-DETERMINISM CONTRIBUTION: {nondet:.4f}")

    metrics = {
        "eq_latent_rel_error_mean": float(eq_seeded.mean()) if eq_seeded.size else float("nan"),
        "eq_latent_rel_error_median": float(np.median(eq_seeded)) if eq_seeded.size else float("nan"),
        "eq_latent_rel_error_unseeded": float(eq_unseeded.mean()) if eq_unseeded.size else float("nan"),
        "eq_latent_rel_error_unseeded_median": float(np.median(eq_unseeded)) if eq_unseeded.size else float("nan"),
        "eq_latent_determinism_mean": float(det_arr.mean()) if det_arr.size else float("nan"),
        "eq_latent_identity_mean": float(id_arr.mean()) if id_arr.size else float("nan"),
        "eq_latent_nondeterminism_contribution": nondet,
        "num_samples": int(eq_seeded.size),
    }
    return metrics, eq_seeded


__all__ = [
    "AnalyzableModel",
    "_sample_indices",
    "_unwrap_dataset",
    "build_static_coords_dataloader",
    "cap_cluster_labels",
    "evaluate_latent_equivariance",
    "gather_inference_batches",
]
