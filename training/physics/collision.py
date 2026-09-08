"""Conservative sphere-expanded gate boxes and segment events."""

from __future__ import annotations

import numpy as np


def gate_basis(gate):
    yaw = gate["yaw_rad"]
    return np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])


def frame_boxes(gate):
    w, h, b, d = (
        gate["width_m"] / 2,
        gate["height_m"] / 2,
        gate["frame_bar_m"],
        gate["frame_depth_m"] / 2,
    )
    return [
        ("left", [-d, -w - b, -h - b], [d, -w, h + b]),
        ("right", [-d, w, -h - b], [d, w + b, h + b]),
        ("bottom", [-d, -w, -h - b], [d, w, -h]),
        ("top", [-d, -w, h], [d, w, h + b]),
    ]


def segment_box(start, end, minimum, maximum):
    enter, leave, normal = 0.0, 1.0, np.zeros(3)
    delta = end - start
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            if start[axis] < minimum[axis] - 1e-9 or start[axis] > maximum[axis] + 1e-9:
                return None
            continue
        a, b = (
            (minimum[axis] - start[axis]) / delta[axis],
            (maximum[axis] - start[axis]) / delta[axis],
        )
        lo, hi = min(a, b), max(a, b)
        if lo > enter:
            enter = lo
            normal = np.zeros(3)
            normal[axis] = -np.sign(delta[axis])
        leave = min(leave, hi)
        if enter > leave + 1e-9:
            return None
    if enter > 1 or leave < 0:
        return None
    return max(0.0, enter), normal


class CollisionWorld:
    def __init__(self, gates, vehicle, rules):
        self.radius = vehicle["collision_radius_m"]
        self.minimum = np.array(rules["workspace_min_m"]) + self.radius
        self.maximum = np.array(rules["workspace_max_m"]) - self.radius
        self.gates = [(g, np.array(g["center_m"]), gate_basis(g), frame_boxes(g)) for g in gates]

    def first_collision(self, start, end):
        hits = []
        for axis in range(3):
            for sign, boundary in [(-1, self.minimum[axis]), (1, self.maximum[axis])]:
                if sign * (start[axis] - boundary) >= 0:
                    alpha = 0.0
                elif sign * (end[axis] - boundary) >= 0:
                    alpha = (boundary - start[axis]) / (end[axis] - start[axis])
                else:
                    continue
                reason = "ground_collision" if axis == 2 and sign == -1 else "workspace_exit"
                normal = np.zeros(3)
                normal[axis] = -sign
                hits.append((alpha, reason, f"workspace:{axis}:{sign}", normal))
        # Broad phase: a gate more than its bounding radius from the segment cannot collide.
        midpoint, half_length = (start + end) / 2, np.linalg.norm(end - start) / 2
        for gate, center, basis, boxes in self.gates:
            if np.linalg.norm(center - midpoint) > half_length + 2.5:
                continue
            a, b = (start - center) @ basis, (end - center) @ basis
            for name, minimum, maximum in boxes:
                hit = segment_box(
                    a, b, np.array(minimum) - self.radius, np.array(maximum) + self.radius
                )
                if hit is not None:
                    hits.append((hit[0], "frame_collision", f"{gate['id']}:{name}", basis @ hit[1]))
        return min(hits, key=lambda h: h[0]) if hits else None


def gate_crossing(start, end, gate, margin):
    basis = gate_basis(gate)
    a = (start - np.array(gate["center_m"])) @ basis
    b = (end - np.array(gate["center_m"])) @ basis
    if a[0] < 0 <= b[0]:
        alpha = -a[0] / (b[0] - a[0])
        point = a + alpha * (b - a)
        passed = (
            abs(point[1]) <= gate["width_m"] / 2 - margin
            and abs(point[2]) <= gate["height_m"] / 2 - margin
        )
        return alpha, "gate_pass" if passed else "gate_miss_forward"
    if a[0] > 0 >= b[0]:
        return -a[0] / (b[0] - a[0]), "gate_cross_backward"
    return None
