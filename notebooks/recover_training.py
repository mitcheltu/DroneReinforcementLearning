"""Bounded, guarded PPO recovery for hover and the first gate (not a full campaign)."""

import argparse
import hashlib
import json
import random
import tempfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from training.envs.drone import DroneEnv
from training.learning.train import Metrics, evaluate, fingerprint


@contextmanager
def model_file(checkpoint):
    """Read ordinary or Kaggle-extracted SB3 archives without modifying input."""
    checkpoint = Path(checkpoint)
    for name in ("model.zip", "model"):
        path = checkpoint / name
        if path.is_file() and zipfile.is_zipfile(path):
            yield path
            return
    directory = checkpoint / "model"
    required = ("data", "policy.pth", "policy.optimizer.pth")
    if not all((directory / name).is_file() for name in required):
        raise ValueError(f"No complete SB3 model archive found in {checkpoint}")
    with tempfile.TemporaryDirectory(prefix="aerorl-model-") as temporary:
        path = Path(temporary) / "model.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for child in sorted(directory.rglob("*")):
                if child.is_file():
                    archive.write(child, child.relative_to(directory).as_posix())
        yield path


def version():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def assessment(model, output, cases):
    results = [evaluate(model, stage, cases=cases, output=output / "traces") for stage in (0, 1)]
    return {
        "steps": model.num_timesteps,
        "cases_per_stage": cases,
        "successes": [sum(r["outcome"] == "success" for r in rows) for rows in results],
        "mean_rewards": [float(np.mean([r["r"] for r in rows])) for rows in results],
        "outcomes": [dict(Counter(r["outcome"] for r in rows)) for rows in results],
        "results": results,
    }


def regressed(candidate, accepted):
    # Same fixed seeds: never accept loss of measured successes in either skill.
    return any(
        new < old for new, old in zip(candidate["successes"], accepted["successes"], strict=True)
    )


def save(model, env, output, state, report):
    target = output / "checkpoints" / f"step-{model.num_timesteps:012d}"
    temporary = target.with_suffix(".partial")
    temporary.mkdir(parents=True, exist_ok=False)
    model.save(temporary / "model.zip")
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "envs": [e.np_random.bit_generator.state for e in env.envs],
        },
        temporary / "rng.pt",
    )
    (temporary / "recovery.json").write_text(
        json.dumps(
            {
                **state,
                "steps": model.num_timesteps,
                "evaluation": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.rename(target)
    return target


def pointer(output, name, checkpoint):
    temporary = output / f"{name}.json.partial"
    temporary.write_text(json.dumps({"checkpoint": str(checkpoint.resolve())}), encoding="utf-8")
    temporary.replace(output / f"{name}.json")


def run(source, output, steps=32768, cases=64, seed=2027, resume=False):
    if steps <= 0 or steps % 4096 or cases <= 0 or seed < 0:
        raise ValueError("steps must be a positive multiple of 4096; cases > 0; seed >= 0")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; inputs are never overwritten")
    metadata = json.loads((source / ("recovery.json" if resume else "campaign.json")).read_text())
    if metadata["fingerprint"] != fingerprint():
        raise ValueError("Original simulator/source fingerprint does not match")
    if resume and any(
        metadata[key] != value
        for key, value in {
            "recovery_version": version(),
            "seed": seed,
            "cases": cases,
        }.items()
    ):
        raise ValueError("Recovery source, seed and evaluation case count must match on resume")
    torch.set_num_threads(1)
    # Every vector step contributes exactly one transition to each task. Evaluation=True
    # disables the old episode-based stage lottery; courses still randomize on reset.
    env = DummyVecEnv(
        [
            lambda stage=stage: DroneEnv(
                frontier=stage, seed=seed + stage * 1000000, evaluation=True, record=True
            )
            for stage in (0, 1)
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    with model_file(source) as path:
        if resume:
            model = PPO.load(path, env=env, device="cpu")
            rng = torch.load(source / "rng.pt", map_location="cpu", weights_only=False)
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            for environment, state in zip(env.envs, rng["envs"], strict=True):
                environment.reset()
                environment.np_random.bit_generator.state = state
        else:
            original = PPO.load(path, device="cpu")
            model = PPO(
                "MlpPolicy",
                env,
                seed=seed,
                device="cpu",
                verbose=0,
                learning_rate=3e-5,
                n_steps=2048,
                batch_size=512,
                n_epochs=3,
                gamma=original.gamma,
                gae_lambda=original.gae_lambda,
                clip_range=0.1,
                ent_coef=0,
                vf_coef=original.vf_coef,
                max_grad_norm=0.5,
                target_kl=0.01,
                policy_kwargs=original.policy_kwargs,
            )
            model.policy.load_state_dict(original.policy.state_dict())
            # Deliberate migration: keep actor, critic and learned std; reset Adam and
            # the recovery step counter. This is not an exact continuation of old PPO.
            del original
    model.tensorboard_log = str(output / "tensorboard")
    state = {
        "fingerprint": fingerprint(),
        "recovery_version": version(),
        "source_checkpoint": str(source),
        "original_steps": metadata["original_steps"] if resume else metadata["steps"],
        "seed": seed,
        "cases": cases,
        "n_envs": 2,
        "stage_transition_shares": [0.5, 0.5],
        "migration": "actor/critic/std transferred; fresh Adam on initial migration",
        "resume_semantics": "optimizer/RNG restored; active episodes restart",
        "scope": "stage 0 and 1 only; no automatic curriculum promotion",
    }
    callback = Metrics(output, {"frontier": 1, "frontier_transitions": 0})
    stop = model.num_timesteps + steps
    status = "budget_reached"
    try:
        accepted = assessment(model, output / "baseline", cases)
        latest = save(model, env, output, state, accepted)
        best = latest
        pointer(output, "latest", latest)
        pointer(output, "best", best)
        print(f"Baseline successes: {accepted['successes']} / {cases}", flush=True)
        while model.num_timesteps < stop:
            model.learn(
                min(16384, stop - model.num_timesteps),
                reset_num_timesteps=False,
                callback=callback,
                tb_log_name="recovery",
            )
            model.logger.dump(step=model.num_timesteps)
            candidate = assessment(model, output / f"eval-{model.num_timesteps}", cases)
            rejected = regressed(candidate, accepted)
            candidate["rejected"] = rejected
            latest = save(model, env, output, state, candidate)
            pointer(output, "latest", latest)
            with (output / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(candidate) + "\n")
            print(
                f"Recovery {model.num_timesteps}: {candidate['successes']} / {cases}; "
                f"rejected={rejected}",
                flush=True,
            )
            if rejected:
                status = "stopped_skill_regression"
                break
            # Success counts dominate; return breaks ties. Preserve a separate best.
            if (sum(candidate["successes"]), sum(candidate["mean_rewards"])) > (
                sum(accepted["successes"]),
                sum(accepted["mean_rewards"]),
            ):
                accepted, best = candidate, latest
                pointer(output, "best", best)
            if cases >= 64 and all(n / cases >= 61 / 64 for n in candidate["successes"]):
                status = "candidate_ready_for_independent_validation"
                break
    finally:
        env.close()
    summary = {
        "status": status,
        "best_checkpoint": str(best),
        "latest_checkpoint": str(latest),
        "best_successes": accepted["successes"],
        "cases_per_stage": cases,
        "recovery_steps": model.num_timesteps,
        "validated_recovery": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--resume", action="store_true")
    run(**vars(parser.parse_args()))
