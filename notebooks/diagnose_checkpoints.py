"""Read-only policy comparison on fixed hover/gate cases; writes separate diagnostics."""

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from training.envs.drone import DroneEnv
from training.envs.settings import ROOT
from training.learning.train import fingerprint, save_trace


def diagnose(output, cases=64):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    current = fingerprint()
    checkpoints = [
        ROOT / "training_checkpoints" / run / "checkpoints" / step
        for run, step in (
            ("aerorl-run-003", "step-000003000320"),
            ("aerorl-run-004", "step-000003151872"),
        )
    ]
    for checkpoint in checkpoints:
        state = json.loads((checkpoint / "campaign.json").read_text())
        if state["fingerprint"] != current:
            raise ValueError("Use the matching source for the original checkpoint comparison")
    with zipfile.ZipFile(output / "baseline-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for directory, pattern in [("training", "*.py"), ("shared", "*.json")]:
            for path in sorted((ROOT / directory).rglob(pattern)):
                archive.write(path, path.relative_to(ROOT).as_posix())
    torch.set_num_threads(1)
    report = {"source_fingerprint": current, "cases_per_stage": cases, "models": []}
    for checkpoint in checkpoints:
        model = PPO.load(checkpoint / "model.zip", device="cpu")
        entry = {
            "checkpoint": str(checkpoint),
            "steps": model.num_timesteps,
            "action_std": model.policy.log_std.detach().exp().cpu().tolist(),
            "stages": [],
        }
        for stage in (0, 1):
            env = DroneEnv(frontier=stage, evaluation=True, record=True)
            rows, recorded = [], Counter()
            for case in range(cases):
                obs, _ = env.reset(seed=10001 + stage * 100000 + case)
                actions, min_gate_distance = [], float("inf")
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    actions.append(action.tolist())
                    obs, _, done, _, info = env.step(action)
                    if env.course["gates"]:
                        min_gate_distance = min(
                            min_gate_distance,
                            float(
                                np.linalg.norm(env.state[:3] - env.course["gates"][0]["center_m"])
                            ),
                        )
                    if done:
                        trajectory = np.array(env.completed["states"])
                        row = {
                            "case": case,
                            "outcome": info["outcome"],
                            **info["episode"],
                            "initial_position": env.course["initial_state"]["position_m"],
                            "final_position": env.state[:3].tolist(),
                            "min_altitude": float(trajectory[:, 3].min()),
                            "max_altitude": float(trajectory[:, 3].max()),
                            "mean_action": np.mean(actions, axis=0).tolist(),
                            "min_gate_distance": min_gate_distance if stage else None,
                        }
                        rows.append(row)
                        outcome = info["outcome"]
                        if recorded[outcome] < 2:
                            save_trace(
                                output
                                / f"{model.num_timesteps}-stage-{stage}-{outcome}-{case}.npz",
                                env.completed,
                            )
                            recorded[outcome] += 1
                        break
                if (case + 1) % 8 == 0:
                    print(
                        f"{model.num_timesteps} stage {stage}: {case + 1}/{cases} "
                        f"{dict(Counter(r['outcome'] for r in rows))}",
                        flush=True,
                    )
            entry["stages"].append(
                {
                    "stage": stage,
                    "outcomes": dict(Counter(r["outcome"] for r in rows)),
                    "mean_duration": float(np.mean([r["t"] for r in rows])),
                    "cases": rows,
                }
            )
            env.close()
        report["models"].append(entry)
        (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", output / "comparison.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/recovery-diagnostics")
    parser.add_argument("--cases", type=int, default=64)
    args = parser.parse_args()
    diagnose(args.output, args.cases)
