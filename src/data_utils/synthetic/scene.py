"""Scene assembly, persistence, and dataset integration for synthetic point clouds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    DatasetConfig,
    EnvCenterSamplerSpec,
    GrainRadiusDistSpec,
    NoiseSpec,
    SplitSpec,
)
from .environments import (
    EnvironmentGT,
    lookup_ground_truth,
    render_environment,
    sample_environment_centers,
    _phase_id_map,
)
from .grains import (
    GrainRegionIndex,
    GrainSeed,
    assign_phase_to_grains,
    build_grain_regions,
    sample_grain_seeds_thomas,
    sample_grain_seeds_voronoi,
)
from .orientation import (
    OrientationField,
    build_orientation_field,
    matrix_to_quaternion,
    quaternion_to_matrix,
    random_quaternion,
)
from .phases import PhaseLibrary


@dataclass
class Scene:
    points: np.ndarray  # (N_env, M, 3)
    phase_labels: np.ndarray  # (N_env,)
    grain_labels: np.ndarray  # (N_env,)
    orientations: np.ndarray  # (N_env, 3, 3)
    quaternions: np.ndarray  # (N_env, 4)
    centers: np.ndarray  # (N_env, 3)
    meta: Dict[str, np.ndarray]
    grain_to_phase: Dict[int, str]
    phase_ids: Dict[str, int]
    seeds: List[GrainSeed]
    config: DatasetConfig
    region_index: GrainRegionIndex | None = None
    orientation_field: OrientationField | None = None

    def subset(self, indices: np.ndarray | Sequence[int]) -> "Scene":
        idx = np.asarray(indices, dtype=int)
        return Scene(
            points=self.points[idx],
            phase_labels=self.phase_labels[idx],
            grain_labels=self.grain_labels[idx],
            orientations=self.orientations[idx],
            quaternions=self.quaternions[idx],
            centers=self.centers[idx],
            meta={k: v[idx] for k, v in self.meta.items()},
            grain_to_phase=self.grain_to_phase,
            phase_ids=self.phase_ids,
            seeds=self.seeds,
            config=self.config,
            region_index=self.region_index,
            orientation_field=self.orientation_field,
        )

    @property
    def num_environments(self) -> int:
        return int(self.points.shape[0])

    @property
    def points_per_environment(self) -> int:
        return int(self.points.shape[1])


def _sample_seeds(cfg: DatasetConfig, rng: np.random.Generator) -> List[GrainSeed]:
    if cfg.grain_model == "thomas":
        return sample_grain_seeds_thomas(cfg, rng)
    if cfg.grain_model == "voronoi":
        return sample_grain_seeds_voronoi(cfg, rng)
    raise ValueError(f"Unsupported grain model {cfg.grain_model}")


def generate_scene(
    cfg: DatasetConfig,
    N_env: int,
    *,
    rng: np.random.Generator | None = None,
    phase_library: PhaseLibrary | None = None,
) -> Scene:
    cfg.validate()
    rng = rng or np.random.default_rng(cfg.seed)
    phase_library = phase_library or PhaseLibrary()

    seeds = _sample_seeds(cfg, rng)
    region_index = build_grain_regions(seeds, cfg)
    phase_map = assign_phase_to_grains(seeds, cfg, rng)
    orientation_field = build_orientation_field(seeds, cfg, rng)

    centers = sample_environment_centers(cfg, region_index, rng=rng, N_env=N_env)
    phase_ids = _phase_id_map(cfg)

    points_list: List[np.ndarray] = []
    phase_labels: List[int] = []
    grain_labels: List[int] = []
    orientation_mats: List[np.ndarray] = []
    quaternions: List[np.ndarray] = []
    boundary: List[float] = []
    anisotropic: List[np.ndarray] = []
    missing_rates: List[float] = []
    outlier_rates: List[float] = []

    for center in centers:
        gt: EnvironmentGT = lookup_ground_truth(
            center,
            region_index,
            phase_map,
            orientation_field,
            phase_ids,
        )
        points, meta = render_environment(
            center,
            gt,
            phase_library,
            cfg,
            rng=rng,
        )
        points_list.append(points)
        phase_labels.append(gt.phase_id)
        grain_labels.append(gt.grain_id)
        orientation_mats.append(gt.orientation)
        quaternions.append(meta["orientation_quaternion"])
        boundary.append(meta["boundary_distance"])
        missing_rates.append(meta["missing_rate"])
        outlier_rates.append(meta["outlier_rate"])
        anisotropic_scale = meta["anisotropic_scale"]
        if anisotropic_scale is None:
            anisotropic.append(np.array([np.nan, np.nan]))
        else:
            anisotropic.append(np.array(anisotropic_scale))

    points_arr = np.stack(points_list).astype(np.float32)
    centers_arr = np.asarray(centers, dtype=np.float32)
    phase_arr = np.asarray(phase_labels, dtype=np.int64)
    grain_arr = np.asarray(grain_labels, dtype=np.int64)
    orientation_arr = np.stack(orientation_mats).astype(np.float32)
    quat_arr = np.stack(quaternions).astype(np.float32)
    meta = {
        "boundary_distance": np.asarray(boundary, dtype=np.float32),
        "missing_rate": np.asarray(missing_rates, dtype=np.float32),
        "outlier_rate": np.asarray(outlier_rates, dtype=np.float32),
        "anisotropic_scale": np.asarray(anisotropic, dtype=np.float32),
    }
    return Scene(
        points=points_arr,
        phase_labels=phase_arr,
        grain_labels=grain_arr,
        orientations=orientation_arr,
        quaternions=quat_arr,
        centers=centers_arr,
        meta=meta,
        grain_to_phase=phase_map,
        phase_ids=phase_ids,
        seeds=seeds,
        config=cfg,
        region_index=region_index,
        orientation_field=orientation_field,
    )


def _split_grain_ids(cfg: DatasetConfig, grain_ids: np.ndarray, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    ratios = cfg.splits.ratios
    order = list(ratios.keys())
    perm = rng.permutation(grain_ids)
    splits: Dict[str, np.ndarray] = {}
    start = 0
    total = perm.shape[0]
    for name in order:
        count = int(round(ratios[name] * total))
        if name == order[-1]:
            end = total
        else:
            end = min(total, start + count)
        splits[name] = np.sort(perm[start:end])
        start = end
    return splits


def make_splits(
    scene: Scene,
    cfg: DatasetConfig,
    *,
    rng: np.random.Generator | None = None,
) -> Dict[str, Scene]:
    rng = rng or np.random.default_rng(cfg.seed + 1)
    grain_ids = np.unique(scene.grain_labels)
    grain_splits = _split_grain_ids(cfg, grain_ids, rng)
    split_scenes: Dict[str, Scene] = {}
    for name, grain_subset in grain_splits.items():
        mask = np.isin(scene.grain_labels, grain_subset)
        indices = np.nonzero(mask)[0]
        split_scene = scene.subset(indices)
        if name == "test" and cfg.splits.holdout_unseen_rotations:
            _inject_unseen_rotations(split_scene, rng)
        split_scenes[name] = split_scene
    return split_scenes


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    q = random_quaternion(rng)
    return quaternion_to_matrix(q)


def _inject_unseen_rotations(scene: Scene, rng: np.random.Generator) -> None:
    extra_rot = np.stack([_random_rotation_matrix(rng) for _ in range(scene.num_environments)])
    scene.orientations = np.einsum("nij,njk->nik", extra_rot, scene.orientations)
    scene.quaternions = np.stack([
        matrix_to_quaternion(scene.orientations[i]) for i in range(scene.num_environments)
    ]).astype(np.float32)
    centered = scene.points - scene.centers[:, None, :]
    rotated = np.einsum("nij,nkj->nki", extra_rot, centered)
    scene.points = rotated + scene.centers[:, None, :]


def export_scene(
    scene: Scene,
    path: str | Path,
    *,
    fmt: Literal["npz", "h5"] = "npz",
    compress: bool = True,
) -> None:
    if fmt != "npz":
        raise NotImplementedError("Only npz export is currently supported")
    path = Path(path)
    save_fn = np.savez_compressed if compress else np.savez

    seed_centers = np.stack([seed.center for seed in scene.seeds]) if scene.seeds else np.zeros((0, 3))
    seed_radii = np.array([seed.radius for seed in scene.seeds], dtype=np.float32)
    parent_ids = np.array([seed.parent_id if seed.parent_id is not None else -1 for seed in scene.seeds], dtype=int)
    grain_ids = np.array([seed.id for seed in scene.seeds], dtype=int)
    phase_names = np.array([scene.grain_to_phase[g] for g in grain_ids], dtype=object)

    cfg_dict = asdict(scene.config)

    save_fn(
        path,
        points=scene.points,
        phase_labels=scene.phase_labels,
        grain_labels=scene.grain_labels,
        orientations=scene.orientations,
        quaternions=scene.quaternions,
        centers=scene.centers,
        meta_boundary=scene.meta["boundary_distance"],
        meta_missing=scene.meta["missing_rate"],
        meta_outlier=scene.meta["outlier_rate"],
        meta_anisotropic=scene.meta["anisotropic_scale"],
        seed_centers=seed_centers,
        seed_radii=seed_radii,
        seed_parents=parent_ids,
        grain_ids=grain_ids,
        grain_phases=phase_names,
        phase_ids=json.dumps(scene.phase_ids),
        config=json.dumps(cfg_dict),
    )


def _load_config(config_json: str) -> DatasetConfig:
    cfg_dict = json.loads(config_json)
    grain_radius = cfg_dict.get("grain_radius_dist")
    if grain_radius is not None:
        grain_radius = GrainRadiusDistSpec(**grain_radius)
    noise = NoiseSpec(**cfg_dict["noise"])
    env_sampler = EnvCenterSamplerSpec(**cfg_dict["env_center_sampler"])
    splits = SplitSpec(**cfg_dict["splits"])
    cfg_dict["noise"] = noise
    cfg_dict["env_center_sampler"] = env_sampler
    cfg_dict["splits"] = splits
    cfg_dict["grain_radius_dist"] = grain_radius
    return DatasetConfig(**cfg_dict)


def load_scene(path: str | Path) -> Scene:
    data = np.load(path, allow_pickle=True)
    cfg = _load_config(data["config"].item())
    points = data["points"]
    phase_labels = data["phase_labels"]
    grain_labels = data["grain_labels"]
    orientations = data["orientations"]
    quaternions = data["quaternions"]
    centers = data["centers"]
    meta = {
        "boundary_distance": data["meta_boundary"],
        "missing_rate": data["meta_missing"],
        "outlier_rate": data["meta_outlier"],
        "anisotropic_scale": data["meta_anisotropic"],
    }
    seed_centers = data["seed_centers"]
    seed_radii = data["seed_radii"]
    seed_parents = data["seed_parents"].astype(int)
    grain_ids = data["grain_ids"].astype(int)
    grain_phases = data["grain_phases"]
    seeds = [
        GrainSeed(id=int(gid), center=seed_centers[i], radius=float(seed_radii[i]), parent_id=None if seed_parents[i] < 0 else int(seed_parents[i]))
        for i, gid in enumerate(grain_ids)
    ]
    grain_to_phase = {int(gid): str(grain_phases[i]) for i, gid in enumerate(grain_ids)}
    phase_ids = json.loads(data["phase_ids"].item())
    region_index = build_grain_regions(seeds, cfg)
    scene = Scene(
        points=points,
        phase_labels=phase_labels,
        grain_labels=grain_labels,
        orientations=orientations,
        quaternions=quaternions,
        centers=centers,
        meta=meta,
        grain_to_phase=grain_to_phase,
        phase_ids=phase_ids,
        seeds=seeds,
        config=cfg,
        region_index=region_index,
    )
    return scene


class SyntheticPointCloudDataset(Dataset):
    """Torch dataset view over a synthetic scene."""

    def __init__(
        self,
        scene: Scene,
        *,
        return_orientation: bool = True,
        return_meta: bool = False,
        device: torch.device | None = None,
    ) -> None:
        self.scene = scene
        self.return_orientation = return_orientation
        self.return_meta = return_meta
        self.device = device
        # Expose environment centers so downstream loaders (e.g. neighbor pairs)
        # can build spatial graphs without iterating the dataset item-by-item.
        self.coords = np.asarray(scene.centers, dtype=np.float32)
        self.phase_name_map = {int(v): str(k) for k, v in scene.phase_ids.items()} if scene.phase_ids else {}

    def __len__(self) -> int:
        return self.scene.num_environments

    def __getitem__(self, idx: int):
        center = torch.as_tensor(self.scene.centers[idx], dtype=torch.float32, device=self.device)
        pts = torch.as_tensor(self.scene.points[idx], dtype=torch.float32, device=self.device) - center
        phase = torch.as_tensor(self.scene.phase_labels[idx], dtype=torch.long, device=self.device)
        grain = torch.as_tensor(self.scene.grain_labels[idx], dtype=torch.long, device=self.device)
        batch = [pts, phase, grain]
        if self.return_orientation:
            orient = torch.as_tensor(self.scene.orientations[idx], dtype=torch.float32, device=self.device)
            quat = torch.as_tensor(self.scene.quaternions[idx], dtype=torch.float32, device=self.device)
            batch.extend([orient, quat])
        if self.return_meta:
            meta = {k: torch.as_tensor(v[idx], dtype=torch.float32, device=self.device) for k, v in self.scene.meta.items()}
            meta["center"] = center
            batch.append(meta)
        return tuple(batch)
