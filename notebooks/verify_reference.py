"""Independent controller baseline for ordered straight courses; no learned policy involved."""

import numpy as np

from training.envs.drone import DroneEnv
from training.physics.collision import gate_basis
from training.physics.controller import reference_action


def verify():
    for stage in (1, 3):
        outcomes = []
        for seed in range(20):
            env = DroneEnv(frontier=stage, evaluation=True)
            env.reset(seed=seed)
            while True:
                gate = env.course["gates"][env.target]
                target = np.array(gate["center_m"]) + 2 * gate_basis(gate)[:, 0]
                action = reference_action(env.state, target, gate["yaw_rad"], env.vehicle)
                _, _, done, _, info = env.step(action)
                if done:
                    outcomes.append(info["outcome"])
                    break
        successes = outcomes.count("success")
        print(f"Stage {stage}: {successes}/20 successes; outcomes={outcomes}", flush=True)
        if successes < 19:
            raise AssertionError("Straight-course reference baseline failed")


if __name__ == "__main__":
    verify()
