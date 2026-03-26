from __future__ import annotations

import math

import numpy as np


def random_rotation_matrices(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    rotation = np.zeros((n, 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    return random_rotation_matrices(rng, 1)[0]


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError(f"Axis-angle rotation requires a non-zero axis, got axis={axis}.")
    axis = axis / norm
    x, y, z = axis
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    one_minus = 1.0 - cosine
    return np.array(
        [
            [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
            [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
            [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
        ],
        dtype=np.float32,
    )


def axis_angle_to_quaternion(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError(f"Quaternion axis-angle conversion requires a non-zero axis, got axis={axis}.")
    axis = axis / norm
    half_angle = 0.5 * angle_rad
    sine = math.sin(half_angle)
    quat = np.array(
        [math.cos(half_angle), axis[0] * sine, axis[1] * sine, axis[2] * sine],
        dtype=np.float32,
    )
    return quat / np.linalg.norm(quat)


def quaternion_multiply(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    quat = np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=np.float32,
    )
    return quat / np.linalg.norm(quat)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (m21 - m12) / scale,
                (m02 - m20) / scale,
                (m10 - m01) / scale,
            ],
            dtype=np.float32,
        )
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quat = np.array(
            [
                (m21 - m12) / scale,
                0.25 * scale,
                (m01 + m10) / scale,
                (m02 + m20) / scale,
            ],
            dtype=np.float32,
        )
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quat = np.array(
            [
                (m02 - m20) / scale,
                (m01 + m10) / scale,
                0.25 * scale,
                (m12 + m21) / scale,
            ],
            dtype=np.float32,
        )
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quat = np.array(
            [
                (m10 - m01) / scale,
                (m02 + m20) / scale,
                (m12 + m21) / scale,
                0.25 * scale,
            ],
            dtype=np.float32,
        )
    return quat / np.linalg.norm(quat)


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float32,
    )


def quaternion_to_rotation_matrix_batch(quaternions: np.ndarray) -> np.ndarray:
    """Convert (N, 4) quaternions to (N, 3, 3) rotation matrices in one vectorized pass."""
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    n = quaternions.shape[0]
    rotation = np.empty((n, 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0_norm = q0 / np.linalg.norm(q0)
    q1_norm = q1 / np.linalg.norm(q1)
    dot = float(np.clip(np.dot(q0_norm, q1_norm), -1.0, 1.0))

    if dot < 0.0:
        q1_norm = -q1_norm
        dot = -dot

    if dot > 0.9995:
        blended = q0_norm + alpha * (q1_norm - q0_norm)
        return blended / np.linalg.norm(blended)

    theta_0 = math.acos(dot)
    sine_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sine = math.sin(theta)
    s0 = math.cos(theta) - dot * sine / sine_0
    s1 = sine / sine_0
    return s0 * q0_norm + s1 * q1_norm


def quaternion_slerp_batch(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized slerp for (N, 4) quaternion arrays with a scalar alpha."""
    norms0 = np.linalg.norm(q0, axis=1, keepdims=True)
    norms1 = np.linalg.norm(q1, axis=1, keepdims=True)
    norms0 = np.where(norms0 > 1e-12, norms0, 1.0)
    norms1 = np.where(norms1 > 1e-12, norms1, 1.0)
    q0n = q0 / norms0
    q1n = q1 / norms1

    dot = np.sum(q0n * q1n, axis=1)
    neg_mask = dot < 0.0
    q1n[neg_mask] *= -1.0
    dot[neg_mask] *= -1.0
    dot = np.clip(dot, -1.0, 1.0)

    # Linear interpolation path for nearly-parallel quaternions
    linear_mask = dot > 0.9995
    # Slerp path for all others
    slerp_mask = ~linear_mask

    result = np.empty_like(q0n)

    # Linear path
    if np.any(linear_mask):
        blended = q0n[linear_mask] + alpha * (q1n[linear_mask] - q0n[linear_mask])
        result[linear_mask] = blended / np.linalg.norm(blended, axis=1, keepdims=True)

    # Slerp path
    if np.any(slerp_mask):
        d = dot[slerp_mask]
        theta_0 = np.arccos(d)
        sine_0 = np.sin(theta_0)
        theta = theta_0 * alpha
        sine = np.sin(theta)
        s0 = np.cos(theta) - d * sine / sine_0
        s1 = sine / sine_0
        result[slerp_mask] = s0[:, None] * q0n[slerp_mask] + s1[:, None] * q1n[slerp_mask]

    return result


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise RuntimeError("Failed to sample a non-zero random unit vector.")
    return (vector / norm).astype(np.float32)
