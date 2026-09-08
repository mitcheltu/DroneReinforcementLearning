import numpy as np

from training.physics.collision import gate_basis
from training.physics.quaternion import rotation


def observation(state, course, target, previous_action, elapsed, timeout, bundle):
    vehicle, rules = bundle["vehicle"].data, bundle["course-rules"].data
    transform = rotation(state[3:7]).T
    raw = np.zeros(40)
    raw[:3], raw[3:6], raw[6:9] = transform @ state[7:10], state[10:13], transform @ [0, 0, -1]
    gates = course["gates"]
    if not gates:
        raw[9:12] = transform @ (np.array(course["start_reference_m"]) - state[:3])
        # Heading does not affect hover reward; use initial heading's projected body nose.
        normal = rotation(course["initial_state"]["quaternion_wxyz"])[:, 0].copy()
        normal[2] = 0
        normal /= np.linalg.norm(normal)
        raw[12:15], raw[30] = transform @ normal, 1
    for offset in range(2):
        if target + offset < len(gates):
            gate = gates[target + offset]
            start = 9 + 6 * offset
            raw[start : start + 3] = transform @ (np.array(gate["center_m"]) - state[:3])
            raw[start + 3 : start + 6] = transform @ gate_basis(gate)[:, 0]
            raw[30 + offset] = 1
    radius = vehicle["collision_radius_m"]
    raw[21], raw[22:26], raw[26:30] = state[2] - radius, previous_action, state[13:]
    for axis in range(3):
        raw[32 + 2 * axis] = state[axis] - radius - rules["workspace_min_m"][axis]
        raw[33 + 2 * axis] = rules["workspace_max_m"][axis] - state[axis] - radius
    raw[38], raw[39] = max(0, len(gates) - target), max(0, 1 - elapsed / timeout)
    if not np.all(np.isfinite(raw)):
        raise FloatingPointError("Nonfinite observation")
    fields = bundle["observation"].data["fields"]
    scaled = raw / [f["scale"] for f in fields]
    lower, upper = [f["minimum"] for f in fields], [f["maximum"] for f in fields]
    clipped = (scaled < lower) | (scaled > upper)
    return np.clip(scaled, lower, upper).astype(np.float32), clipped
