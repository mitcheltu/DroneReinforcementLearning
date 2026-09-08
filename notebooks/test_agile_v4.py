import numpy as np
import torch

from notebooks.agile_v4 import AgileFeatures, DroneEnv, expert, reset_course


def test_geometric_features_are_invariant_to_body_coordinates():
    env = DroneEnv(evaluation=True)
    reset_course(env, 5, 170050001)
    env.target = 1
    observation = torch.tensor(env._observation()[None])
    features = AgileFeatures(env.observation_space)
    original = features(observation)
    # Rotate every body vector together; gate-relative scalar geometry must agree.
    angle = .71
    transform = torch.tensor([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=torch.float32)
    rotated = observation.clone()
    for start in (0, 3, 6, 9, 12, 15, 18, 40, 43):
        rotated[:, start:start+3] = observation[:, start:start+3] @ transform.T
    actual = features(rotated)
    assert actual.shape == (1, 39)
    torch.testing.assert_close(actual[:, 31:], original[:, 31:], atol=1e-6, rtol=1e-5)


def test_expert_has_no_hidden_history():
    env = DroneEnv(evaluation=True)
    reset_course(env, 4, 170040000)
    state = env.state.copy()
    action = expert(env)
    for _ in range(10):
        env.step(expert(env))
    env.state[:] = state
    np.testing.assert_array_equal(expert(env), action)


def test_braking_teacher_completes_independently_rotated_gate():
    env = DroneEnv(evaluation=True)
    reset_course(env, 4, 170040000)
    while True:
        action = expert(env)
        assert np.isfinite(action).all()
        _, _, done, _, info = env.step(action)
        if done:
            break
    assert info['outcome'] == 'success'
    assert info['gates_passed'] == 1


def test_geometric_features_export_to_onnx(tmp_path):
    import onnx
    import onnxruntime as ort

    env = DroneEnv(evaluation=True)
    reset_course(env, 5, 170050001)
    observations = [env._observation()]
    env.target = 1
    observations.append(env._observation())
    values = np.asarray(observations, dtype=np.float32)
    features = AgileFeatures(env.observation_space).eval()
    path = tmp_path / 'features.onnx'
    torch.onnx.export(features, torch.tensor(values), str(path), opset_version=17,
                      input_names=['observation'], output_names=['features'],
                      dynamic_axes={'observation': {0: 'batch'}, 'features': {0: 'batch'}},
                      dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    for batch in (values, values[:1]):
        actual = session.run(None, {'observation': batch})[0]
        expected = features(torch.tensor(batch)).detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
