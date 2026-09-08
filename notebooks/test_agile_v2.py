import numpy as np
import torch

from notebooks.agile_v2 import DroneEnv, AgileFeatures, expert, reset_course
from training.envs.drone import DroneEnv as OriginalEnv
from training.physics.quaternion import rotation


def test_previous_gate_observation_is_explicit_and_original_fields_unchanged():
    env=DroneEnv(evaluation=True)
    reset_course(env,4,120400002)
    assert env._observation().shape == (46,)
    assert np.all(env._observation()[40:] == 0)
    env.target=1
    actual=env._observation()
    np.testing.assert_array_equal(actual[:40],OriginalEnv._observation(env))
    gate=env.course["gates"][0]
    r=rotation(env.state[3:7]).T
    expected=np.r_[r@(np.array(gate["center_m"])-env.state[:3])/40,
                   r@[np.cos(gate["yaw_rad"]),np.sin(gate["yaw_rad"]),0]]
    np.testing.assert_allclose(actual[40:],np.clip(expected,-1,1),atol=1e-7)
    assert AgileFeatures(env.observation_space)(torch.tensor(actual[None])).shape==(1,31)


def test_clearance_demonstration_finishes_a_seeded_reversal():
    env=DroneEnv(evaluation=True)
    reset_course(env,4,120400002)
    while True:
        action=expert(env)
        assert np.isfinite(action).all()
        _,_,done,_,info=env.step(action)
        if done: break
    assert info["outcome"]=="success"
    assert info["gates_passed"]==3
    assert env.elapsed<60
