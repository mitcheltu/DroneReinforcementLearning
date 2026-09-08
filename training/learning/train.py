"""CPU PPO notebook/CLI runner with atomic checkpoints and fixed promotion evaluations."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from training.envs.drone import DroneEnv
from training.envs.settings import ROOT, settings


def fingerprint():
    digest = hashlib.sha256()
    for directory, pattern in [(ROOT / "training", "*.py"), (ROOT / "shared", "*.json")]:
        for path in sorted(directory.rglob(pattern)):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class Metrics(BaseCallback):
    def __init__(self, output, campaign):
        super().__init__()
        self.output, self.campaign = output, campaign

    def _on_step(self):
        for index, info in enumerate(self.locals["infos"]):
            environment = self.training_env.envs[index]
            trace = environment.completed if "episode" in info else environment.trace
            trace.setdefault("raw_actions", []).append(self.locals["actions"][index].tolist())
            trace.setdefault("policy_update", []).append(self.model._n_updates)
            if info["stage"] == self.campaign["frontier"]:
                self.campaign["frontier_transitions"] += 1
            if "episode" in info:
                row = {
                    "transition": self.num_timesteps,
                    **info["episode"],
                    "stage": info["stage"],
                    "outcome": info["outcome"],
                    "gates_passed": info["gates_passed"],
                }
                with (self.output / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                bucket = f"{self.num_timesteps // 1000000}-{info['stage']}-{info['outcome']}"
                counts = self.campaign.setdefault("recorded_buckets", {})
                if counts.get(bucket, 0) < 2:
                    directory = self.output / "training-traces"
                    used = (
                        sum(p.stat().st_size for p in directory.glob("*.npz"))
                        if directory.exists()
                        else 0
                    )
                    if used < 1000000000:
                        save_trace(directory / f"{bucket}-{counts.get(bucket, 0)}.npz", trace)
                        counts[bucket] = counts.get(bucket, 0) + 1
                environment.completed = None
        return True


def evaluate(model, stage, cases=64, seed=10001, output=None):
    env = DroneEnv(frontier=stage, evaluation=True, record=output is not None)
    results = []
    for index in range(cases):
        obs, _ = env.reset(seed=seed + stage * 100000 + index)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            if done:
                results.append(
                    {"case": index, "stage": stage, "outcome": info["outcome"], **info["episode"]}
                )
                if output is not None and index < 4:
                    save_trace(output / f"stage-{stage}-case-{index}.npz", env.completed)
                break
    env.close()
    return results


def save_trace(path, trace):
    """Notebook diagnostic format, explicitly distinct from the browser replay contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.asarray(trace[key])
        for key in (
            "states",
            "commands",
            "observations",
            "actions",
            "rewards",
            "terminal_observation",
        )
    }
    for key in ("raw_actions", "policy_update"):
        if key in trace:
            arrays[key] = np.asarray(trace[key])
    arrays["metadata_json"] = np.array(
        json.dumps(
            {
                "format": "aerorl-notebook-trace-v1",
                **{key: trace[key] for key in ("course", "events", "metrics")},
            }
        )
    )
    np.savez_compressed(path, **arrays)


def checkpoint(model, env, campaign, output):
    name = f"step-{model.num_timesteps:012d}"
    target = output / "checkpoints" / name
    if target.exists():
        return target
    temporary = target.with_name(name + ".partial")
    temporary.mkdir(parents=True, exist_ok=True)
    model.save(temporary / "model.zip")
    # Trusted local checkpoint only: torch serialization contains RNG tuples.
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "envs": [e.np_random.bit_generator.state for e in env.envs],
        },
        temporary / "rng.pt",
    )
    state = {
        **campaign,
        "steps": model.num_timesteps,
        "fingerprint": fingerprint(),
        "resume_semantics": "optimizer and RNG restored; active episodes restart",
    }
    (temporary / "campaign.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    pointer = output / "latest.json.partial"
    pointer.write_text(json.dumps({"checkpoint": str(target.resolve())}), encoding="utf-8")
    os.replace(pointer, output / "latest.json")
    # Retain newest two within this output directory; never touch resume input.
    import shutil

    checkpoints = sorted((output / "checkpoints").glob("step-*"))
    complete = [
        p for p in checkpoints if (p / "campaign.json").exists() and not p.name.endswith(".partial")
    ]
    for old in complete[:-2]:
        shutil.rmtree(old)
    return target


def run(
    output,
    steps=1_000_000,
    budget=10_000_000,
    seed=101,
    n_envs=1,
    resume=None,
    smoke=False,
    wall_seconds=32400,
):
    if steps <= 0 or budget <= 0 or n_envs not in (1, 2, 4, 8) or seed < 0:
        raise ValueError("Require positive steps/budget, nonnegative seed, and n_envs in 1,2,4,8")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.json").exists() and resume is None:
        raise ValueError(
            "Output already has checkpoints; set resume or choose a new output directory"
        )
    torch.set_num_threads(1)
    config = settings()["training"].data
    campaign = {
        "frontier": 0,
        "frontier_transitions": 0,
        "streak": 0,
        "last_eval": 0,
        "budget": budget,
        "seed": seed,
        "n_envs": n_envs,
        "smoke": smoke,
    }
    if resume:
        resume = Path(resume)
        campaign = json.loads((resume / "campaign.json").read_text())
        if campaign["fingerprint"] != fingerprint():
            raise ValueError(
                "Checkpoint source/config fingerprint differs; use its original source bundle"
            )
        if any(
            campaign[k] != v
            for k, v in {"budget": budget, "n_envs": n_envs, "seed": seed, "smoke": smoke}.items()
        ):
            raise ValueError("Resume requires identical budget, n_envs, seed, and smoke settings")
    env = DummyVecEnv(
        [
            lambda rank=i: DroneEnv(campaign["frontier"], seed + rank * 1000000, record=True)
            for i in range(n_envs)
        ]
    )
    p = config["ppo"]
    if resume:
        model = PPO.load(resume / "model", env=env, device="cpu")
        model.tensorboard_log = str(output / "tensorboard")
        rng = torch.load(resume / "rng.pt", weights_only=False, map_location="cpu")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        # Initialize before replacing RNG so reset does not reapply the original seed.
        for e, state in zip(env.envs, rng["envs"], strict=True):
            e.reset()
            e.np_random.bit_generator.state = state
    else:
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            device="cpu",
            verbose=1,
            n_steps=64 if smoke else p["n_steps"],
            batch_size=64 if smoke else p["batch_size"],
            n_epochs=2 if smoke else p["n_epochs"],
            gamma=p["gamma"],
            gae_lambda=p["gae_lambda"],
            clip_range=p["clip_range"],
            ent_coef=p["ent_coef"],
            vf_coef=p["vf_coef"],
            max_grad_norm=p["max_grad_norm"],
            target_kl=p["target_kl"],
            policy_kwargs={
                "net_arch": {"pi": p["actor_layers"], "vf": p["critic_layers"]},
                "activation_fn": torch.nn.Tanh,
                "ortho_init": True,
                "log_std_init": -1,
            },
            tensorboard_log=str(output / "tensorboard"),
        )
    started = time.monotonic()
    stop = min(model.num_timesteps + steps, budget)
    last_save = model.num_timesteps
    callback = Metrics(output, campaign)
    try:
        while model.num_timesteps < stop and time.monotonic() - started < wall_seconds:
            rate = p["learning_rate"]["start"] + (
                p["learning_rate"]["end"] - p["learning_rate"]["start"]
            ) * min(1, model.num_timesteps / budget)
            model.lr_schedule = lambda _, value=rate: value
            model.learn(
                model.n_steps * n_envs,
                reset_num_timesteps=False,
                callback=callback,
                tb_log_name="ppo",
            )
            model.logger.dump(step=model.num_timesteps)
            interval = config["curriculum"]["promotion_interval_transitions"]
            if not smoke and model.num_timesteps - campaign["last_eval"] >= interval:
                results = evaluate(model, campaign["frontier"], output=output / "evaluation-traces")
                successes = sum(r["outcome"] == "success" for r in results)
                with (output / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"steps": model.num_timesteps, "results": results}) + "\n"
                    )
                campaign["last_eval"] = model.num_timesteps
                campaign["streak"] = campaign["streak"] + 1 if successes >= 61 else 0
                if (
                    campaign["streak"] >= 2
                    and campaign["frontier_transitions"] >= interval
                    and campaign["frontier"] < 6
                ):
                    campaign.update(
                        frontier=campaign["frontier"] + 1, frontier_transitions=0, streak=0
                    )
                    env.env_method("set_frontier", campaign["frontier"])
                print(
                    f"Evaluation: {successes}/64; next frontier {campaign['frontier']}", flush=True
                )
            if model.num_timesteps - last_save >= config["checkpoint"]["interval_transitions"]:
                checkpoint(model, env, campaign, output)
                last_save = model.num_timesteps
    except KeyboardInterrupt:
        print("Interrupted: saving current optimizer state; unfinished rollout is discarded.")
    finally:
        saved = checkpoint(model, env, campaign, output)
        env.close()
    print(f"Checkpoint: {saved}", flush=True)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--budget", type=int, default=10000000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
