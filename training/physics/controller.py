"""Collective/body-rate action mapping, proportional rate loop, and reference controller."""

import numpy as np

from training.physics.quaternion import rotation


def action_command(action, vehicle):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (4,) or not np.all(np.isfinite(action)):
        raise ValueError("Action must contain four finite numbers")
    u = np.clip(action, -1, 1)
    return (
        u,
        (u[0] + 1) * vehicle["action"]["max_collective_n"] / 2,
        (u[1:] * vehicle["action"]["max_body_rates_radps"]),
    )


def motor_command(state, collective, rates, physics):
    omega = state[10:13]
    torque = np.array(physics.config["rate_gains"]) * (rates - omega)
    torque += np.cross(omega, physics.inertia * omega)
    tx, ty, tz = torque
    b, k = 4 * physics.b, 4 * physics.kappa
    delta = np.array(
        [
            tx / b - ty / b + tz / k,
            tx / b + ty / b - tz / k,
            -tx / b + ty / b + tz / k,
            -tx / b - ty / b - tz / k,
        ]
    )
    center = collective / 4
    scale = 1.0
    for d in delta:
        if d > 0:
            scale = min(scale, (physics.fmax - center) / d)
        elif d < 0:
            scale = min(scale, center / -d)
    scale = float(np.clip(scale, 0, 1))
    return np.clip(center + scale * delta, 0, physics.fmax), scale


def reference_action(state, target, yaw, vehicle):
    desired_velocity = 1.5 * (target - state[:3])
    desired_velocity *= min(1, 3 / max(np.linalg.norm(desired_velocity), 1e-12))
    acceleration = 2 * (desired_velocity - state[7:10])
    acceleration *= min(1, 4 / max(np.linalg.norm(acceleration), 1e-12))
    force = vehicle["mass_kg"] * (acceleration + [0, 0, vehicle["gravity_mps2"]])
    z = force / np.linalg.norm(force)
    y = np.cross(z, [np.cos(yaw), np.sin(yaw), 0])
    y /= np.linalg.norm(y)
    desired = np.column_stack([np.cross(y, z), y, z])
    current = rotation(state[3:7])
    error = 0.5 * (desired.T @ current - current.T @ desired)
    rates = -4 * np.array([error[2, 1], error[0, 2], error[1, 0]])
    collective = np.clip(np.dot(force, current[:, 2]), 0, 16)
    return np.clip([collective / 8 - 1, *(rates / [6, 6, 3])], -1, 1)
