"""Prepare model, course definitions, and Python flight parity fixtures for the browser."""

import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort

from training.envs.drone import DroneEnv


def build():
    root = Path(__file__).resolve().parents[1]
    public = root / "web/public"
    (public / "models").mkdir(exist_ok=True)
    (public / "onnx").mkdir(exist_ok=True)
    actor = root / "runs/imitation-repair/iteration-0/actor.onnx"
    shutil.copyfile(actor, public / "models/actor.onnx")
    for suffix in ("mjs", "wasm"):
        name = f"ort-wasm-simd-threaded.{suffix}"
        shutil.copyfile(
            root / "web/node_modules/onnxruntime-web/dist" / name, public / "onnx" / name
        )
    courses = []
    for stage in range(7):
        for case in range(16):
            env = DroneEnv(frontier=stage, evaluation=True)
            env.reset(seed=90000000 + stage * 100000 + case)
            courses.append({"stage": stage, "variant": case + 1, "course": env.course})
            env.close()
    (public / "courses.json").write_text(json.dumps(courses), encoding="utf-8")
    session = ort.InferenceSession(str(actor), providers=["CPUExecutionProvider"])
    fixtures = []
    for stage, failure in [(0, False), (1, False), (6, False), (1, True)]:
        env = DroneEnv(frontier=stage, evaluation=True)
        obs, _ = env.reset(options={"course": courses[stage * 16]["course"]})
        rows = []
        while True:
            action = (
                np.array([-1, 0, 0, 0], dtype=np.float32)
                if failure
                else session.run(None, {"observation": obs[None]})[0][0]
            )
            before = obs.tolist()
            obs, reward, done, _, info = env.step(action)
            rows.append(
                {
                    "observation": before,
                    "action": action.tolist(),
                    "state": env.state.tolist(),
                    "reward": reward,
                    "time": env.elapsed,
                    "target": env.target,
                }
            )
            if done:
                break
        fixtures.append(
            {
                "stage": stage,
                "failure": failure,
                "course": env.course,
                "steps": rows,
                "outcome": info["outcome"],
                "reward": env.total_reward,
            }
        )
        print(f"Fixture stage {stage}, failure={failure}: {info['outcome']}", flush=True)
        env.close()
    (root / "web/tests/fixtures/flight-parity.json").write_text(
        json.dumps(fixtures), encoding="utf-8"
    )


if __name__ == "__main__":
    build()
