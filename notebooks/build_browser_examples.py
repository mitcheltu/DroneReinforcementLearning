"""Prepare browser examples without modifying training code or a running campaign."""
import shutil
from pathlib import Path

import numpy as np

from training.envs.drone import DroneEnv
from training.learning.train import save_trace
from training.physics.collision import gate_basis
from training.physics.controller import reference_action


def build():
    root = Path(__file__).resolve().parents[1]
    output = root / "web" / "public" / "examples"
    output.mkdir(parents=True, exist_ok=True)
    failure = root / "runs/notebook-validation/training-traces/0-0-hover_departure-0.npz"
    if not failure.exists():
        raise FileNotFoundError("The original actual-training example is unavailable")
    shutil.copy2(failure, output / "failure.npz")
    env = DroneEnv(frontier=3, evaluation=True, record=True)
    env.reset(seed=0)
    while True:
        gate = env.course["gates"][env.target]
        target = np.array(gate["center_m"]) + 2*gate_basis(gate)[:, 0]
        action = reference_action(env.state, target, gate["yaw_rad"], env.vehicle)
        _, _, done, _, info = env.step(action)
        if done:
            assert info["is_success"]
            save_trace(output / "reference.npz", env.completed)
            print(info)
            break


if __name__ == "__main__":
    build()
