"""Export an experimental deterministic actor and verify ONNX inference parity."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import PPO

from training.envs.drone import DroneEnv


class Actor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        features = self.policy.extract_features(observation)
        latent = self.policy.mlp_extractor.forward_actor(features)
        return torch.clamp(self.policy.action_net(latent), -1, 1)


def export(checkpoint):
    checkpoint = Path(checkpoint)
    torch.set_num_threads(1)
    model = PPO.load(checkpoint / "model.zip", device="cpu")
    actor = Actor(model.policy).eval()
    path = checkpoint / "actor.onnx"
    torch.onnx.export(
        actor,
        torch.zeros(1, 40),
        str(path),
        opset_version=17,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))
    env = DroneEnv(frontier=6)
    obs, _ = env.reset(seed=4567)
    samples = []
    rng = np.random.default_rng(4567)
    for _ in range(512):
        samples.append(obs)
        obs, _, done, _, _ = env.step(rng.uniform(-1, 1, 4))
        if done:
            obs, _ = env.reset()
    values = np.stack(samples)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"observation": values})[0]
    with torch.no_grad():
        expected = actor(torch.from_numpy(values)).numpy()
    error = float(np.max(np.abs(actual - expected)))
    if error > 1e-5:
        path.unlink()
        raise AssertionError(f"ONNX parity failed: {error}")
    (checkpoint / "export-report.json").write_text(
        json.dumps(
            {
                "status": "experimental",
                "input": "float32[batch,40]",
                "output": "clipped deterministic float32[batch,4]",
                "max_absolute_error": error,
                "cases": len(values),
                "release_qualified": False,
            },
            indent=2,
        )
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    print(export(parser.parse_args().checkpoint))
