"""Right-handed, Hamilton wxyz quaternion helpers; rotations map body to world."""

from __future__ import annotations

import numpy as np


def normalize(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise FloatingPointError("Invalid quaternion")
    return q / norm


def multiply(q, r):
    w, x, y, z = q
    a, b, c, d = r
    return np.array(
        [
            w * a - x * b - y * c - z * d,
            w * b + x * a + y * d - z * c,
            w * c - x * d + y * a + z * b,
            w * d + x * c - y * b + z * a,
        ]
    )


def rotation(q):
    w, x, y, z = normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def from_euler(roll, pitch, yaw):
    return normalize(
        multiply(
            multiply(
                [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0],
            ),
            [np.cos(roll / 2), np.sin(roll / 2), 0, 0],
        )
    )


def slerp(q0, q1, alpha):
    q0, q1 = normalize(q0), normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    dot = np.clip(dot, 0, 1)
    alpha = np.clip(alpha, 0, 1)
    if dot > 0.9995:
        return normalize((1 - alpha) * q0 + alpha * q1)
    theta = np.arccos(dot)
    return normalize(
        (np.sin((1 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)
    )
