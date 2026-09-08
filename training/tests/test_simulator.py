import numpy as np
from stable_baselines3.common.env_checker import check_env

from training.envs.course_generator import generate_course
from training.envs.course_validation import course_errors
from training.envs.drone import DroneEnv
from training.envs.settings import settings
from training.physics.collision import gate_crossing
from training.physics.controller import motor_command, reference_action
from training.physics.dynamics import Quadrotor


def test_gym_contract():
    check_env(DroneEnv())


def test_hover_equilibrium_and_mixer():
    vehicle = settings()["vehicle"].data
    physics = Quadrotor(vehicle)
    state = np.zeros(17)
    state[2], state[3], state[13:] = 5, 1, physics.mass * physics.gravity / 4
    command, scale = motor_command(state, physics.mass * physics.gravity, np.zeros(3), physics)
    assert scale == 1
    np.testing.assert_allclose(physics.step(state, command), state, atol=1e-12)
    command, _ = motor_command(state, 8, np.array([0.1, 0, 0]), physics)
    assert physics.torque(command)[0] > 0
    np.testing.assert_allclose(physics.torque(command)[1:], 0, atol=1e-12)


def test_courses_all_stages():
    bundle = settings()
    for stage in bundle["training"].data["curriculum"]["stages"]:
        for seed in range(30):
            a = generate_course(seed, seed + 100, stage, bundle)
            assert a == generate_course(seed, seed + 100, stage, bundle)
            assert len(a["gates"]) == stage["gate_count"]
            assert not course_errors(a, bundle["vehicle"].data, bundle["course-rules"].data, stage)
            assert [g["label"] for g in a["gates"]] == list(range(1, stage["gate_count"] + 1))


def test_crossing_direction_and_margin():
    gate = {"center_m": [0, 0, 5], "yaw_rad": 0, "width_m": 2.5, "height_m": 2.5}
    assert gate_crossing(np.array([-1.0, 0, 5]), np.array([1.0, 0, 5]), gate, 0.35) == (
        0.5,
        "gate_pass",
    )
    assert (
        gate_crossing(np.array([1.0, 0, 5]), np.array([-1.0, 0, 5]), gate, 0.35)[1]
        == "gate_cross_backward"
    )
    assert (
        gate_crossing(np.array([-1.0, 1, 5]), np.array([1.0, 1, 5]), gate, 0.35)[1]
        == "gate_miss_forward"
    )


def test_reference_hover():
    for seed in range(20):
        env = DroneEnv(evaluation=True)
        env.reset(seed=seed)
        for _ in range(301):
            action = reference_action(
                env.state, np.array(env.course["start_reference_m"]), 0, env.vehicle
            )
            _, reward, done, truncated, info = env.step(action)
            assert np.isfinite(reward)
            if done:
                assert info["outcome"] == "success"
                assert not truncated
                break
        else:
            raise AssertionError("Hover did not terminate")


def test_random_transitions_finite():
    env = DroneEnv(frontier=6, record=True)
    env.reset(seed=7)
    rng = np.random.default_rng(8)
    for _ in range(2000):
        obs, reward, done, truncated, _ = env.step(rng.uniform(-1, 1, 4))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward) and not truncated
        assert abs(np.linalg.norm(env.state[3:7]) - 1) < 1e-12
        if done:
            assert len(env.completed["states"]) == len(env.completed["commands"]) + 1
            env.reset()
