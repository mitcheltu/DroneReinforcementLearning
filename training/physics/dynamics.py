"""17-state quadrotor, held motor commands and coupled fixed-step RK4."""

from __future__ import annotations

import numpy as np

from training.physics.quaternion import multiply, normalize, rotation


class Quadrotor:
    def __init__(self, config):
        self.config = config
        self.mass = config["mass_kg"]
        self.gravity = config["gravity_mps2"]
        self.inertia = np.array(config["inertia_kgm2"])
        self.b = config["arm_length_m"] / np.sqrt(2)
        self.kappa = config["reaction_torque_per_thrust_m"]
        self.fmax = config["rotor_max_thrust_n"]
        self.dt = 1 / config["physics_hz"]

    def torque(self, f):
        f0, f1, f2, f3 = f
        return np.array(
            [
                self.b * (f0 + f1 - f2 - f3),
                self.b * (-f0 + f1 + f2 - f3),
                self.kappa * (f0 - f1 + f2 - f3),
            ]
        )

    def derivative(self, state, motor_command):
        q, v, omega, motors = state[3:7], state[7:10], state[10:13], state[13:17]
        result = np.empty(17)
        result[:3] = v
        result[3:7] = 0.5 * multiply(q, [0, *omega])
        result[7:10] = rotation(q)[:, 2] * np.sum(motors) / self.mass
        result[7:10] -= self.config["linear_drag_kgps"] * v / self.mass
        result[9] -= self.gravity
        result[10:13] = (self.torque(motors) - np.cross(omega, self.inertia * omega)) / self.inertia
        result[13:17] = (motor_command - motors) / self.config["motor_time_constant_s"]
        return result

    def step(self, state, motor_command, dt=None):
        h = self.dt if dt is None else dt
        k1 = self.derivative(state, motor_command)
        k2 = self.derivative(state + h * k1 / 2, motor_command)
        k3 = self.derivative(state + h * k2 / 2, motor_command)
        k4 = self.derivative(state + h * k3, motor_command)
        result = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Nonfinite RK4 state")
        result[3:7] = normalize(result[3:7])
        tolerance = self.config["motor_roundoff_tolerance_n"]
        if np.any(result[13:] < -tolerance) or np.any(result[13:] > self.fmax + tolerance):
            raise FloatingPointError("Motor state exceeded physical bounds")
        result[13:] = np.clip(result[13:], 0, self.fmax)
        return result


def state_array(state):
    return np.array(
        [
            *state["position_m"],
            *state["quaternion_wxyz"],
            *state["velocity_world_mps"],
            *state["omega_body_radps"],
            *state["motor_thrust_n"],
        ],
        dtype=np.float64,
    )


def state_json(state):
    return {
        "position_m": state[:3].tolist(),
        "quaternion_wxyz": state[3:7].tolist(),
        "velocity_world_mps": state[7:10].tolist(),
        "omega_body_radps": state[10:13].tolist(),
        "motor_thrust_n": state[13:].tolist(),
    }
