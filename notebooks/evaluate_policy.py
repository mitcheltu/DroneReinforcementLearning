"""Run a saved deterministic neural policy locally and retain successes and failures."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from stable_baselines3 import PPO

from training.envs.drone import DroneEnv
from training.learning.train import fingerprint, save_trace


def run(checkpoint, output, stages=(0, 1), cases=64, seed=80000000):
    if cases < 1 or seed < 0 or not stages or any(stage not in range(7) for stage in stages):
        raise ValueError("Require positive cases, nonnegative seed, and stages in 0..6")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = PPO.load(Path(checkpoint) / "model.zip", device="cpu")
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "fingerprint": fingerprint(),
        "deterministic": True,
        "seed": seed,
        "stages": [],
    }
    for stage in stages:
        env = DroneEnv(frontier=stage, evaluation=True, record=True)
        rows, recorded = [], Counter()
        for case in range(cases):
            obs, _ = env.reset(seed=seed + stage * 100000 + case)
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(action)
                if done:
                    rows.append(
                        {
                            "case": case,
                            "reset_seed": seed + stage * 100000 + case,
                            "outcome": info["outcome"],
                            "gates_passed": info["gates_passed"],
                            **info["episode"],
                        }
                    )
                    if recorded[info["outcome"]] < 2:
                        save_trace(
                            output / f"stage-{stage}-{info['outcome']}-{case}.npz", env.completed
                        )
                        recorded[info["outcome"]] += 1
                    break
            if (case + 1) % 8 == 0:
                print(
                    f"Stage {stage}, {case + 1}/{cases}: "
                    f"{dict(Counter(r['outcome'] for r in rows))}",
                    flush=True,
                )
        env.close()
        report["stages"].append(
            {
                "stage": stage,
                "cases": cases,
                "successes": sum(r["outcome"] == "success" for r in rows),
                "outcomes": dict(Counter(r["outcome"] for r in rows)),
                "results": rows,
            }
        )
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stages", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=80000000)
    run(**vars(parser.parse_args()))
