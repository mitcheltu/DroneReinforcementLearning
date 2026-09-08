"""Geometric course checks shared by generation and fixed-case evaluation."""

import numpy as np

from training.physics.collision import CollisionWorld, gate_basis
from training.physics.dynamics import state_array


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def enclosing_box(gate):
    half = np.array(
        [
            gate["frame_depth_m"] / 2,
            gate["width_m"] / 2 + gate["frame_bar_m"],
            gate["height_m"] / 2 + gate["frame_bar_m"],
        ]
    )
    extent = np.abs(gate_basis(gate)) @ half
    center = np.array(gate["center_m"])
    return center - extent, center + extent


def course_errors(course, vehicle, rules, stage=None):
    errors = []
    gates = course["gates"]
    if not gates:
        return errors
    boxes = [enclosing_box(g) for g in gates]
    for minimum, maximum in boxes:
        if np.any(minimum < np.array(rules["workspace_min_m"]) + 0.5 - 1e-9) or np.any(
            maximum > np.array(rules["workspace_max_m"]) - 0.5 + 1e-9
        ):
            errors.append("frame_outside_workspace")
    for i, (amin, amax) in enumerate(boxes):
        for bmin, bmax in boxes[i + 1 :]:
            if np.all(amin - 0.3 <= bmax + 0.3) and np.all(bmin - 0.3 <= amax + 0.3):
                errors.append("frame_overlap")
    points = np.array([course["start_reference_m"], *[g["center_m"] for g in gates]])
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta[:, :2], axis=1)
    if np.any(lengths < 6 - 1e-9) or np.any(lengths > 8 + 1e-9):
        errors.append("spacing")
    heights = points[1:, 2]
    if np.any(heights < 3) or np.any(heights > 10):
        errors.append("gate_height")
    height_limit = stage["max_height_change_m"] if stage else rules["max_height_change_m"]
    if abs(delta[0, 2]) > min(height_limit, 0.25) + 1e-9 or np.any(
        abs(delta[1:, 2]) > height_limit + 1e-9
    ):
        errors.append("altitude_change")
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    turn_limit = stage["max_turn_rad"] if stage else rules["max_turn_rad"]
    if np.any(abs(wrap(np.diff(headings))) > turn_limit + 1e-9):
        errors.append("turn_angle")
    if np.any(abs(wrap(headings - headings[0])) > rules["max_heading_drift_rad"] + 1e-9):
        errors.append("heading_drift")
    yaws = np.array([g["yaw_rad"] for g in gates])
    jitter = stage["yaw_jitter_rad"] if stage else rules["max_approach_rad"]
    if np.any(abs(wrap(yaws - headings)) > jitter + 1e-9):
        errors.append("approach_angle")
    if np.any(abs(wrap(yaws[:-1] - headings[1:])) > rules["max_outgoing_rad"] + 1e-9):
        errors.append("outgoing_angle")
    for i in range(len(delta)):
        for j in range(i + 2, len(delta)):
            matrix = np.column_stack([delta[i, :2], -delta[j, :2]])
            if abs(np.linalg.det(matrix)) < 1e-10:
                # Collinear overlapping XY intervals with overlapping expanded heights.
                offset = points[j, :2] - points[i, :2]
                if abs(delta[i, 0] * offset[1] - delta[i, 1] * offset[0]) < 1e-9:
                    axis = int(np.argmax(abs(delta[i, :2])))
                    a = sorted(points[[i, i + 1], axis])
                    b = sorted(points[[j, j + 1], axis])
                    az, bz = sorted(points[[i, i + 1], 2]), sorted(points[[j, j + 1], 2])
                    if (
                        max(a[0], b[0]) <= min(a[1], b[1])
                        and max(az[0], bz[0]) - min(az[1], bz[1]) <= 3
                    ):
                        errors.append("route_intersection")
                continue
            t, u = np.linalg.solve(matrix, points[j, :2] - points[i, :2])
            if 0 <= t <= 1 and 0 <= u <= 1:
                if abs(points[i, 2] + t * delta[i, 2] - points[j, 2] - u * delta[j, 2]) < 3:
                    errors.append("route_intersection")
    start = state_array(course["initial_state"])[:3]
    if CollisionWorld(gates, vehicle, rules).first_collision(start, start):
        errors.append("invalid_start")
    return sorted(set(errors))
