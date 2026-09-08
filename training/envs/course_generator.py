"""Seeded upright-gate rejection generator. Gates stay fixed for an episode."""

import uuid

import numpy as np

from training.envs.course_validation import course_errors, enclosing_box, wrap
from training.physics.dynamics import state_json
from training.physics.quaternion import from_euler


def initial_state(reference, yaw, stage, rng, vehicle):
    c, s = np.cos(yaw), np.sin(yaw)
    basis = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    state = np.zeros(17)
    state[:3] = reference + basis @ rng.uniform(
        -stage["position_perturbation_m"], stage["position_perturbation_m"], 3
    )
    state[3:7] = from_euler(
        *rng.uniform(-stage["tilt_perturbation_rad"], stage["tilt_perturbation_rad"], 2),
        yaw + rng.uniform(-stage["yaw_perturbation_rad"], stage["yaw_perturbation_rad"]),
    )
    state[7:10] = basis @ rng.uniform(
        -stage["velocity_perturbation_mps"], stage["velocity_perturbation_mps"], 3
    )
    state[10:13] = rng.uniform(
        -stage["rate_perturbation_radps"], stage["rate_perturbation_radps"], 3
    )
    state[13:] = vehicle["mass_kg"] * vehicle["gravity_mps2"] / 4
    return state_json(state)


def generate_course(seed, reset_seed, stage, bundle):
    vehicle, rules = bundle["vehicle"].data, bundle["course-rules"].data
    rng, resets = np.random.default_rng(seed), np.random.default_rng(reset_seed)
    namespace = uuid.UUID(rules["generator"]["uuid_namespace"])
    name = f"course-generator-v1:{stage['index']}:{seed}:{reset_seed}"
    course = {
        "schema_version": 1,
        "course_id": str(uuid.uuid5(namespace, name)),
        "name": f"Stage {stage['index']} · {seed}",
        "mode": "curriculum",
        "gates": [],
        "rules_sha256": bundle["course-rules"].sha256,
        "generator": {
            "version": "course-generator-v1",
            "seed": str(seed),
            "reset_seed": str(reset_seed),
            "stage": stage["index"],
            "attempt_count": 1,
        },
    }
    if stage["gate_count"] == 0:
        reference = np.array([*rng.uniform(-10, 10, 2), rng.uniform(4, 6)])
        yaw = rng.uniform(-np.pi, np.pi)
        course["start_reference_m"] = reference.tolist()
        course["initial_state"] = initial_state(reference, yaw, stage, resets, vehicle)
        return course
    for proposal in range(1, rules["generator"]["max_proposals"] + 1):
        points = [np.array([0.0, 0.0, rng.uniform(4, 6)])]
        heading, headings, gates = 0.0, [], []
        for j in range(stage["gate_count"]):
            length = rng.uniform(6, 8)
            if j:
                heading += rng.uniform(-stage["max_turn_rad"], stage["max_turn_rad"])
            dz = stage["max_height_change_m"] if j else min(0.25, stage["max_height_change_m"])
            points.append(
                points[-1]
                + [length * np.cos(heading), length * np.sin(heading), rng.uniform(-dz, dz)]
            )
            headings.append(heading)
            gates.append(
                {
                    "id": str(uuid.uuid5(namespace, f"{name}:gate:{j}")),
                    "label": j + 1,
                    "center_m": points[-1].tolist(),
                    "yaw_rad": heading
                    + rng.uniform(-stage["yaw_jitter_rad"], stage["yaw_jitter_rad"]),
                    **rules["gate"],
                }
            )
        points = np.array(points)
        if max(abs(np.array(headings))) > rules["max_heading_drift_rad"] or np.any(
            (points[1:, 2] < 3) | (points[1:, 2] > 10)
        ):
            continue
        yaw = rng.uniform(-np.pi, np.pi)
        matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        points[:, :2] = points[:, :2] @ matrix.T
        for j, gate in enumerate(gates):
            gate["center_m"] = points[j + 1].tolist()
            gate["yaw_rad"] = float(wrap(gate["yaw_rad"] + yaw))
        bounds = [enclosing_box(g) for g in gates]
        lower = np.min([b[0][:2] for b in bounds] + [points[0, :2] - 0.3], axis=0)
        upper = np.max([b[1][:2] for b in bounds] + [points[0, :2] + 0.3], axis=0)
        if np.any(upper - lower > 119):
            continue
        # Equivalent to recenter then sample the entire feasible translation interval.
        translation = rng.uniform(-59.5 - lower, 59.5 - upper)
        points[:, :2] += translation
        for j, gate in enumerate(gates):
            gate["center_m"] = points[j + 1].tolist()
        course["gates"] = gates
        course["start_reference_m"] = points[0].tolist()
        course["generator"]["attempt_count"] = proposal
        for _ in range(rules["generator"]["max_reset_proposals"]):
            course["initial_state"] = initial_state(points[0], yaw, stage, resets, vehicle)
            errors = course_errors(course, vehicle, rules, stage)
            if not errors:
                return course
            if errors != ["invalid_start"]:
                break
    raise RuntimeError(f"course_generation_exhausted seed={seed} stage={stage['index']}")
