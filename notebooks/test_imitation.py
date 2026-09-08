"""Geometry and observation-contract checks for the replacement policy."""

import numpy as np
import pytest
import torch

from notebooks.train_imitation import FlightFeatures, teacher
from training.envs.drone import DroneEnv
from training.physics.quaternion import from_euler, multiply, rotation


@pytest.mark.parametrize("stage", [0, 1])
def test_control_features_and_teacher_are_invariant_to_course_translation_and_heading(stage):
    env = DroneEnv(frontier=stage, evaluation=True)
    obs, _ = env.reset(seed=791)
    features = FlightFeatures(env.observation_space)
    before = features(torch.tensor(obs[None])).detach().numpy()
    action = teacher(env)
    angle = 1.2
    quaternion = from_euler(0, 0, angle)
    transform = rotation(quaternion)
    offset = np.array([2.0, -3.0, 0.0])
    env.state[:3] = transform @ env.state[:3] + offset
    env.state[3:7] = multiply(quaternion, env.state[3:7])
    env.state[7:10] = transform @ env.state[7:10]
    initial = env.course["initial_state"]
    initial["quaternion_wxyz"] = multiply(quaternion, initial["quaternion_wxyz"]).tolist()
    env.course["start_reference_m"] = (
        transform @ np.array(env.course["start_reference_m"]) + offset
    ).tolist()
    for gate in env.course["gates"]:
        gate["center_m"] = (transform @ np.array(gate["center_m"]) + offset).tolist()
        gate["yaw_rad"] += angle
    after = features(torch.tensor(env._observation()[None])).detach().numpy()
    np.testing.assert_allclose(before, after, atol=1e-6)
    np.testing.assert_allclose(action, teacher(env), atol=1e-6)
    env.close()


def test_gate_lookahead_only_applies_when_remaining_gates_are_present():
    env = DroneEnv(evaluation=True)
    features = FlightFeatures(env.observation_space)
    obs = torch.zeros(2, 40)
    obs[:, 12] = 1
    obs[1, 38] = 0.1
    result = features(obs)
    assert result.shape == (2, 12)
    assert result[0, 6] == 0
    assert result[1, 6].item() == pytest.approx(0.2)
    env.close()
