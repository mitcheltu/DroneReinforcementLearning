import numpy as np
import torch

from notebooks.agile_obstacles import DroneEnv, AgileFeatures, Roadmap, expert, reset_course
from training.physics.quaternion import rotation


def test_obstacle_map_contains_every_non_target_gate():
    env = DroneEnv(evaluation=True)
    reset_course(env, 7, 230080000)
    env.target = 3
    observation = env._observation()
    slots = observation[46:].reshape(9, 7)
    assert observation.shape == (109,)
    np.testing.assert_array_equal(slots[:, 6], 1)
    reconstructed = slots[:, :3]*200 @ rotation(env.state[3:7]).T + env.state[:3]
    expected = [np.asarray(gate['center_m']) for i, gate in enumerate(env.course['gates']) if i != env.target]
    expected.sort(key=lambda center: np.linalg.norm(center-env.state[:3]))
    np.testing.assert_allclose(reconstructed, expected, atol=1e-5)
    feature = AgileFeatures(env.observation_space)(torch.tensor(observation[None]))
    assert feature.shape == (1, 138)
    assert torch.isfinite(feature).all()


def test_single_gate_has_no_other_obstacles():
    env = DroneEnv(evaluation=True)
    reset_course(env, 0, 230010000)
    np.testing.assert_array_equal(env._observation()[46:], 0)


def test_roadmap_blocks_bars_but_preserves_gate_opening():
    env = DroneEnv(evaluation=True)
    reset_course(env, 0, 230010000)
    graph = Roadmap(env)
    gate = env.course['gates'][0]
    from training.physics.collision import gate_basis
    center, basis = np.asarray(gate['center_m']), gate_basis(gate)
    assert graph.clear(center-basis[:, 0]*3, center+basis[:, 0]*3)[0]
    offset = basis[:, 1]*1.3
    assert not graph.clear(center-basis[:, 0]*3+offset, center+basis[:, 0]*3+offset)[0]
    start, goal = center-basis[:, 0]*3+offset, center+basis[:, 0]*3+offset
    waypoint = graph.waypoint(start, goal)
    assert graph.clear(start, waypoint)[0]
    assert not np.allclose(waypoint, start)


def test_known_reversal_obstacle_failure_completes():
    env = DroneEnv(evaluation=True)
    reset_course(env, 5, 120500010)
    while True:
        _, _, done, _, info = env.step(expert(env))
        if done:
            break
    assert info['outcome'] == 'success'
    assert info['gates_passed'] == 3


def test_obstacle_features_export_to_onnx(tmp_path):
    import onnxruntime as ort
    env = DroneEnv(evaluation=True)
    reset_course(env, 7, 230080000)
    observations = [env._observation()]
    env.target = 1
    observations.append(env._observation())
    values = np.asarray(observations)
    features = AgileFeatures(env.observation_space).eval()
    path = tmp_path/'obstacles.onnx'
    torch.onnx.export(features, torch.tensor(values), str(path), opset_version=17,
                      input_names=['observation'], output_names=['features'], dynamo=False,
                      dynamic_axes={'observation': {0: 'batch'}, 'features': {0: 'batch'}})
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    for batch in (values, values[:1]):
        actual = session.run(None, {'observation': batch})[0]
        expected = features(torch.tensor(batch)).detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
