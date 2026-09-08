"""Coverage and control contract tests for the experimental curriculum."""
import numpy as np
import pytest
import torch

from notebooks.agile_curriculum import PROFILES, AgileFeatures, course_for, expert, reset_course, wrap
from training.envs.drone import DroneEnv
from training.physics.quaternion import from_euler


@pytest.mark.parametrize("level", range(8))
def test_seeded_layouts_are_reproducible_and_within_workspace(level):
    env = DroneEnv(evaluation=True)
    first = course_for(env, level, 99173)
    assert first == course_for(env, level, 99173)
    assert len(first["gates"]) == PROFILES[level]["count"]
    points = [np.array(first["start_reference_m"])]
    for i, gate in enumerate(first["gates"]):
        point = np.array(gate["center_m"])
        assert gate["label"] == i+1
        assert np.max(np.abs(point[:2])) <= 45
        assert 3 < point[2] < 14
        length = np.linalg.norm((point-points[-1])[:2])
        assert PROFILES[level]["spacing"][0]-1e-8 <= length <= PROFILES[level]["spacing"][1]+1e-8
        assert min(np.linalg.norm(point-p) for p in points) >= 5
        points.append(point)
    env.close()


def test_reversal_and_heading_coverage_are_not_original_gentle_generator():
    env = DroneEnv(evaluation=True)
    course = course_for(env, 4, 99173)
    points = np.array([course["start_reference_m"]]+[g["center_m"] for g in course["gates"]])
    headings = np.arctan2(np.diff(points, axis=0)[:, 1], np.diff(points, axis=0)[:, 0])
    assert np.all(np.abs(wrap(np.diff(headings))) >= np.deg2rad(150)-1e-8)
    offsets = []
    for seed in range(30):
        course = course_for(env, 5, seed)
        vector = np.array(course["gates"][0]["center_m"])-course["start_reference_m"]
        offsets.append(abs(wrap(course["gates"][0]["yaw_rad"]-np.arctan2(vector[1], vector[0]))))
    assert max(offsets) > np.deg2rad(150)
    env.close()


def test_opposite_heading_has_nonzero_turn_action_and_features_are_finite():
    env = DroneEnv(evaluation=True)
    reset_course(env, 0, 123)
    yaw = env.course["gates"][0]["yaw_rad"]
    env.state[3:7] = from_euler(0, 0, yaw+np.pi)
    action = expert(env)
    assert abs(action[3]) > .1
    values = AgileFeatures(env.observation_space)(torch.tensor(env._observation()[None]))
    assert values.shape == (1, 25)
    assert torch.isfinite(values).all()
    assert env.timeout == 60
    env.close()
