"""Orientation sampling utilities for synthetic polycrystal scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math

import numpy as np

from .config import DatasetConfig
from .grains import GrainSeed


Quaternion = np.ndarray  # shape (4,) in (x, y, z, w) order
RotationMatrix = np.ndarray  # shape (3, 3)


def random_quaternion(rng: np.random.Generator) -> Quaternion:
    u1, u2, u3 = rng.random(3)
    q = np.array([
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ])
    return q.astype(float)


def normalize_quaternion(q: Quaternion) -> Quaternion:
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero quaternion cannot be normalized.")
    return q / norm


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quaternion_to_matrix(q: Quaternion) -> RotationMatrix:
    q = normalize_quaternion(q)
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def matrix_to_quaternion(R: RotationMatrix) -> Quaternion:
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")
    trace = R.trace()
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w])
    return normalize_quaternion(quat)


def axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> Quaternion:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or abs(angle) < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / norm
    half = angle / 2.0
    sin_half = math.sin(half)
    return np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, math.cos(half)])


def so3_geodesic_angle(R1: RotationMatrix, R2: RotationMatrix, *, degrees: bool = False) -> float:
    delta = R1 @ R2.T
    trace = np.clip((np.trace(delta) - 1) / 2, -1.0, 1.0)
    angle = math.acos(trace)
    return math.degrees(angle) if degrees else angle


def apply_rotation(points: np.ndarray, R: RotationMatrix) -> np.ndarray:
    return points @ R.T


def sample_base_orientation(grain_id: int, rng: np.random.Generator) -> Quaternion:
    return normalize_quaternion(random_quaternion(rng))


@dataclass
class OrientationField:
    base_quaternions: Dict[int, Quaternion]
    linear_maps: Dict[int, np.ndarray]
    centers: Dict[int, np.ndarray]
    radii: Dict[int, float]
    kappa: float

    def R(self, grain_id: int, x: np.ndarray) -> RotationMatrix:
        q0 = self.base_quaternions[grain_id]
        delta = x - self.centers[grain_id]
        linear_map = self.linear_maps[grain_id]
        omega = linear_map @ delta
        angle = float(np.linalg.norm(omega))
        if angle > self.kappa:
            omega = omega * (self.kappa / (angle + 1e-9))
            angle = self.kappa
        q_delta = axis_angle_to_quaternion(omega, angle)
        q = quaternion_multiply(q_delta, q0)
        return quaternion_to_matrix(q)

    def base_matrix(self, grain_id: int) -> RotationMatrix:
        return quaternion_to_matrix(self.base_quaternions[grain_id])


def build_orientation_field(
    seeds: Iterable[GrainSeed],
    cfg: DatasetConfig,
    rng: np.random.Generator,
) -> OrientationField:
    base_quaternions: Dict[int, Quaternion] = {}
    linear_maps: Dict[int, np.ndarray] = {}
    centers: Dict[int, np.ndarray] = {}
    radii: Dict[int, float] = {}
    target_kappa = cfg.orientation_intra_grain_kappa
    for seed in seeds:
        base_quaternions[seed.id] = sample_base_orientation(seed.id, rng)
        # draw smooth variation matrix scaled to respect kappa near grain boundary
        scale = target_kappa / max(seed.radius, 1e-6)
        linear_maps[seed.id] = rng.normal(scale=scale / 2, size=(3, 3))
        centers[seed.id] = seed.center
        radii[seed.id] = seed.radius
    return OrientationField(
        base_quaternions=base_quaternions,
        linear_maps=linear_maps,
        centers=centers,
        radii=radii,
        kappa=target_kappa,
    )
