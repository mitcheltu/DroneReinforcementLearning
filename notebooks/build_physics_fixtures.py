"""Generate numerical browser fixtures from the unchanged Python training implementation."""
import json
from pathlib import Path

import numpy as np

from training.envs.settings import settings
from training.physics.controller import action_command, motor_command
from training.physics.dynamics import Quadrotor


def build():
    config = settings()["vehicle"]
    physics = Quadrotor(config.data)
    rng = np.random.default_rng(615)
    scenarios = []
    for kind in ("hover", "yaw_step", "mixed_saturation"):
        state = np.zeros(17)
        state[2], state[3] = 5, 1
        state[13:] = config.data["mass_kg"]*config.data["gravity_mps2"]/4
        initial = state.tolist()
        transitions = []
        action = np.zeros(4)
        for tick in range(240):
            if tick % 2 == 0:
                action = rng.uniform(-1.5,1.5,4) if kind == "mixed_saturation" else np.array([
                    config.data["mass_kg"]*config.data["gravity_mps2"]/8-1,0,0,
                    .4 if kind == "yaw_step" and tick >= 60 else 0])
            _, collective, rates = action_command(action, config.data)
            motors, scale = motor_command(state, collective, rates, physics)
            state = physics.step(state, motors)
            transitions.append({"action":action.tolist(),"motors":motors.tolist(),
                                "scale":scale,"state":state.tolist()})
        scenarios.append({"name":kind,"initial":initial,"transitions":transitions})
    path = Path(__file__).resolve().parents[1]/"web/tests/fixtures/physics-parity.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"vehicle_sha256":config.sha256,"scenarios":scenarios}),encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
