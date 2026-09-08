"""Run the original PPO trainer while recording every completed training episode."""
import argparse
import hashlib
import json
from pathlib import Path

from training.learning import train


class RecordingBudgetReached(RuntimeError):
    pass


def run(output, recording_bytes=2_000_000_000, **kwargs):
    if recording_bytes < 1:
        raise ValueError("recording_bytes must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    directory = output / "all-training-traces"
    directory.mkdir(exist_ok=True)
    used = sum(p.stat().st_size for p in directory.glob("*.npz"))
    if used >= recording_bytes:
        raise ValueError("Recording budget already exhausted; use a new output directory")
    base = train.Metrics

    class AllMetrics(base):
        def _on_step(self):
            nonlocal used
            completed = [(i, self.training_env.envs[i].completed) for i, info in
                         enumerate(self.locals["infos"]) if "episode" in info]
            result = super()._on_step()  # Appends the final raw action before clearing completed.
            for index, trace in completed:
                path = directory / f"step-{self.num_timesteps:012d}-env-{index}.npz"
                if path.exists():
                    raise RuntimeError(f"Refusing to overwrite {path}")
                train.save_trace(path, trace)
                used += path.stat().st_size
            if used >= recording_bytes:
                raise RecordingBudgetReached("Recording budget reached after saving completed episodes")
            return result

    (output / "recording-policy.json").write_text(json.dumps({
        "mode": "every completed training episode", "directory": "all-training-traces",
        "budget_bytes": recording_bytes,
        "on_budget": "stop training and checkpoint; final vector step may exceed quota",
        "unfinished_episodes": "not recorded on interruption",
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    train.Metrics = AllMetrics
    try:
        return train.run(output=output, **kwargs)
    except RecordingBudgetReached as error:
        print(f"{error}. Trainer saved its checkpoint. No more transitions will be collected.", flush=True)
        return Path(json.loads((output / "latest.json").read_text())["checkpoint"])
    finally:
        train.Metrics = base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--budget", type=int, default=10000000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recording-bytes", type=int, default=2_000_000_000)
    run(**vars(parser.parse_args()))
