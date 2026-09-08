"""Experimental broad-layout curriculum; original training contracts stay unchanged."""
import copy
import json
import uuid
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from training.envs.drone import DroneEnv
from training.physics.controller import reference_action
from training.physics.quaternion import from_euler, rotation

# Metres and degrees. A global rigid rotation is also randomized per course.
PROFILES = [
    dict(name="gentle", count=1, spacing=(6, 10), turn=0, heading=15),
    dict(name="wide_spacing", count=1, spacing=(12, 35), turn=0, heading=20),
    dict(name="turns_60", count=3, spacing=(7, 12), turn=60, heading=15),
    dict(name="sharp_120", count=3, spacing=(9, 16), turn=120, heading=25),
    dict(name="reversals", count=3, spacing=(10, 22), turn=180, heading=20),
    dict(name="rotated_gates", count=1, spacing=(8, 25), turn=0, heading=180),
    dict(name="mixed", count=3, spacing=(10, 30), turn=180, heading=180),
    dict(name="ten_gate_mixed", count=10, spacing=(8, 25), turn=180, heading=180),
]


def wrap(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def course_for(env, level, seed):
    rng = np.random.default_rng(seed)
    profile = PROFILES[level]
    # Use the original generator only for the contract and hover-state template.
    env.reset(seed=seed)
    course = copy.deepcopy(env.course)
    course.update(course_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"agile-v1:{level}:{seed}")),
                  name=f"Agile {profile['name']} seed {seed}", mode="experimental", generator=None)
    for _ in range(1000):
        points = [np.array([0., 0., rng.uniform(5, 9)])]
        heading = rng.uniform(-np.pi, np.pi)
        initial_yaw = heading + rng.uniform(-np.pi, np.pi)
        gates = []
        for index in range(profile["count"]):
            turn = rng.uniform(-profile["turn"], profile["turn"])
            if level == 4 and index:
                turn = rng.choice([-1, 1]) * rng.uniform(150, 180)
            if index:
                heading += np.deg2rad(turn)
            length = rng.uniform(*profile["spacing"])
            point = points[-1] + [length*np.cos(heading), length*np.sin(heading), rng.uniform(-1, 1)]
            if max(abs(point[:2])) > 45 or not 3 < point[2] < 14:
                break
            if min(np.linalg.norm(point-p) for p in points) < 5:
                break
            points.append(point)
            gates.append(dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{course['course_id']}:{index}")),
                              label=index+1, center_m=point.tolist(),
                              yaw_rad=wrap(heading+np.deg2rad(rng.uniform(-profile["heading"], profile["heading"]))),
                              **env.rules["gate"]))
        if len(gates) == profile["count"]:
            break
    else:
        raise RuntimeError("Course proposal budget exhausted")
    course["gates"] = gates
    course["start_reference_m"] = points[0].tolist()
    state = course["initial_state"]
    state.update(position_m=points[0].tolist(), quaternion_wxyz=from_euler(0, 0, initial_yaw).tolist(),
                 velocity_world_mps=[0., 0., 0.], omega_body_radps=[0., 0., 0.])
    return course


def reset_course(env, level, seed):
    course = course_for(env, level, seed)
    obs, info = env.reset(seed=seed, options={"course": course})
    # Explicit extended time budget: wide courses are not original 12/20/45-second tasks.
    env.timeout = {1: 60., 3: 180., 10: 600.}[len(course["gates"])]
    return env._observation(), info


def navigation_target(env):
    if not env.course["gates"]:
        return np.array(env.course["start_reference_m"]), 0.
    gate = env.course["gates"][env.target]
    center = np.array(gate["center_m"])
    n = np.array([np.cos(gate["yaw_rad"]), np.sin(gate["yaw_rad"]), 0.])
    side = np.array([-n[1], n[0], 0.])
    displacement = env.state[:3]-center
    x = np.dot(displacement, n)
    lateral = displacement - x*n
    transverse = np.linalg.norm(lateral)
    forward_speed = np.dot(env.state[7:10], n)
    # Stage on the entrance side before crossing. Exit-side arrivals first route
    # around the frame. This expert supplies labels only, never inference actions.
    if transverse < .65 and (x < -1.2 or forward_speed > .5):
        target = center + 3*n
    elif x > -1 and abs(np.dot(displacement, side)) < 3:
        sign = 1 if np.dot(displacement, side) >= 0 else -1
        target = center + max(x, 1.5)*n + 3.5*sign*side
    else:
        target = center - 4*n
    return target, gate["yaw_rad"]


def expert(env):
    target, yaw = navigation_target(env)
    if not env.course["gates"]:
        nose = rotation(env.course["initial_state"]["quaternion_wxyz"])[:, 0]
        yaw = np.arctan2(nose[1], nose[0])
    nose = rotation(env.state[3:7])[:, 0]
    current_yaw = np.arctan2(nose[1], nose[0])
    # Avoid the reference controller's zero yaw torque at exactly 180 degrees.
    yaw = current_yaw + np.clip(wrap(yaw-current_yaw), -np.pi/3, np.pi/3)
    return reference_action(env.state, target, yaw, env.vehicle).astype(np.float32)


class AgileFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=25)

    def forward(self, obs):
        # Retain gate displacement and heading separately, plus next gate and
        # previous action. Unlike the old extractor, no fixed beyond-gate target.
        return torch.cat((obs[:, :3]*3, obs[:, 3:9], obs[:, 9:12]*4,
                          obs[:, 12:15], obs[:, 15:18]*4, obs[:, 18:21],
                          obs[:, 22:26]), dim=1)


if __name__ == "__main__":
    env = DroneEnv(evaluation=True, record=False)
    rows = []
    for level, profile in enumerate(PROFILES):
        for case in range(2):
            obs, _ = reset_course(env, level, 110000000+level*10000+case)
            while True:
                obs, _, done, _, info = env.step(expert(env))
                if done:
                    break
            row = dict(level=level, name=profile["name"], case=case, **info)
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path("runs/agile-expert-diagnostic.json").write_text(json.dumps(rows, indent=2))
