"""Build a compact montage of every available saved trace, with honest provenance."""
import json
from collections import Counter
from pathlib import Path

import numpy as np


def compact(states):
    indices = np.unique(np.r_[np.arange(0, len(states), max(1, int(np.ceil(len(states)/600)))), len(states)-1])
    return np.asarray(states)[indices, :8].tolist()


def build():
    root = Path(__file__).resolve().parents[1]
    traces, logs = [], []
    for run in sorted((root / "training_checkpoints").glob("aerorl-run-*")):
        counts = Counter()
        for line in (run / "episodes.jsonl").read_text().splitlines():
            counts[json.loads(line)["stage"]] += 1
        logs.append({"run": run.name, "episodes_by_stage": dict(counts)})
        for path in sorted((run / "training-traces").glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata_json"]))
                traces.append({"id": f"{run.name}/{path.name}", "source": "ppo", "run": run.name,
                               "stage": meta["metrics"]["stage"], "outcome": meta["metrics"]["outcome"],
                               "duration": float(data["states"][-1, 0]), "course": meta["course"],
                               "samples": compact(data["states"]), "provenance": "recorded training trace"})
    paths = list((root / "runs/imitation-repair/iteration-0/validation-traces").glob("*.npz"))
    paths += list((root / "runs/imitation-broad").glob("*.npz"))
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            traces.append({"id": str(path.relative_to(root / "runs")), "source": "validation",
                           "run": "Repaired neural policy", "stage": meta["metrics"]["stage"],
                           "outcome": meta["metrics"]["outcome"], "duration": float(data["states"][-1, 0]),
                           "course": meta["course"], "samples": compact(data["states"]),
                           "provenance": "recorded validation trace, not training"})
    # Reproduce the 32 deterministic reference flights used for imitation labels.
    # These are explicitly identified as reconstructed, not original recordings.
    from notebooks.train_imitation import teacher
    from training.envs.drone import DroneEnv
    from training.learning.train import fingerprint
    report = json.loads((root / "runs/imitation-repair/report.json").read_text(encoding="utf-8-sig"))
    if report["fingerprint"] != fingerprint():
        raise ValueError("Reconstruction requires the original simulator configuration")
    for stage in (0, 1):
        for case in range(16):
            env = DroneEnv(frontier=stage, evaluation=True, record=True)
            env.reset(seed=20000000+stage*10000+case)
            while True:
                _, _, done, _, info = env.step(teacher(env))
                if done:
                    break
            traces.append({"id": f"demonstration-{stage}-{case}", "source": "demonstrations",
                           "run": "Imitation reference demonstrations", "stage": stage,
                           "outcome": info["outcome"], "duration": env.elapsed, "course": env.course,
                           "samples": compact(env.completed["states"]),
                           "provenance": "reconstructed from original demonstration seeds and controller"})
            env.close()
    output = root / "web/public/montage.json"
    output.write_text(json.dumps({"version": 1, "traces": traces, "episode_logs": logs,
                                 "note": "All retained traces are included. PPO recordings are a subset; unrecorded flights cannot be recovered. Geometry is downsampled to at most 601 poses per attempt."}), encoding="utf-8")
    print(json.dumps({"traces_by_source": dict(Counter(t["source"] for t in traces)), "episode_logs": logs,
                      "bytes": output.stat().st_size}, indent=2))


if __name__ == "__main__":
    build()
