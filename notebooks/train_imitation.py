"""Train a replacement neural flight policy from reference actions and learner rollouts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from training.envs.drone import DroneEnv
from training.learning.train import evaluate, fingerprint
from training.physics.collision import gate_basis
from training.physics.controller import reference_action
from training.physics.quaternion import from_euler, multiply, rotation


class FlightFeatures(BaseFeaturesExtractor):
    """Body-frame control features computed solely from the public 40-value observation."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=12)

    def forward(self, obs):
        # Aim two metres beyond the current gate to avoid stopping in its plane.
        racing = (obs[:, 38:39] > 0).to(obs.dtype)
        target = obs[:, 9:12] * 4 + 0.2 * racing * obs[:, 12:15]
        return torch.cat((obs[:, :3] * 3, obs[:, 6:9], target, obs[:, 12:15]), dim=1)


def teacher(env):
    if env.course["gates"]:
        gate = env.course["gates"][env.target]
        target = np.array(gate["center_m"]) + 2 * gate_basis(gate)[:, 0]
        yaw = gate["yaw_rad"]
    else:
        target = np.array(env.course["start_reference_m"])
        nose = rotation(env.course["initial_state"]["quaternion_wxyz"])[:, 0]
        yaw = np.arctan2(nose[1], nose[0])
    return reference_action(env.state, target, yaw, env.vehicle).astype(np.float32)


def collect(model, iteration, observations, labels, episodes=16):
    rng = np.random.default_rng(73000 + iteration)
    counts = Counter()
    for stage in (0, 1):
        env = DroneEnv(frontier=stage, evaluation=True)
        for episode in range(episodes):
            obs, _ = env.reset(seed=20000000 + iteration * 100000 + stage * 10000 + episode)
            step = 0
            while True:
                target_action = teacher(env)
                if step % 3 == 0:
                    observations.append(obs.copy())
                    labels.append(target_action)
                    # Label nearby physically consistent states as well as actual trajectory states.
                    original = env.state.copy()
                    for _ in range(2):
                        env.state = original.copy()
                        env.state[:3] += rng.normal(0, 0.25, 3)
                        env.state[7:10] += rng.normal(0, 0.35, 3)
                        env.state[3:7] = multiply(
                            original[3:7], from_euler(*rng.normal(0, 0.10, 3))
                        )
                        observations.append(env._observation())
                        labels.append(teacher(env))
                    env.state = original
                # Initial demonstrations, then DAgger: label states visited by the learner.
                action = (
                    target_action if iteration == 0 else model.predict(obs, deterministic=True)[0]
                )
                obs, _, done, _, info = env.step(action)
                step += 1
                if done:
                    counts[f"{stage}:{info['outcome']}"] += 1
                    break
        env.close()
        print(f"Collection iteration {iteration}, stage {stage}: {dict(counts)}", flush=True)
    return dict(counts)


def fit(model, observations, labels, epochs, seed):
    x = torch.tensor(np.asarray(observations), dtype=torch.float32)
    y = torch.tensor(np.asarray(labels), dtype=torch.float32)
    actor_parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    optimizer = torch.optim.Adam(actor_parameters, lr=0.001)
    generator = torch.Generator().manual_seed(seed)
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        total = 0.0
        for indices in torch.randperm(len(x), generator=generator).split(512):
            features = model.policy.extract_features(x[indices])
            prediction = model.policy.action_net(model.policy.mlp_extractor.forward_actor(features))
            loss = ((prediction - y[indices]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            optimizer.step()
            total += loss.item() * len(indices)
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: action MSE {total / len(x):.8f}", flush=True)
    model.policy.set_training_mode(False)
    return total / len(x)


def assess(model, cases, seed, output):
    report = []
    for stage in (0, 1):
        rows = evaluate(model, stage, cases=cases, seed=seed, output=output)
        report.append(
            {
                "stage": stage,
                "successes": sum(r["outcome"] == "success" for r in rows),
                "cases": cases,
                "outcomes": dict(Counter(r["outcome"] for r in rows)),
                "results": rows,
            }
        )
        print(
            f"Evaluation seed {seed}, stage {stage}: {report[-1]['successes']}/{cases}", flush=True
        )
    return report


def run(output, iterations=5, epochs=150, episodes=16):
    if min(iterations, epochs, episodes) < 1:
        raise ValueError("iterations, epochs and episodes must be positive")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    executed_source = Path(__file__).read_bytes()
    (output / "train_imitation.executed.py").write_bytes(executed_source)
    torch.set_num_threads(1)
    env = DroneEnv(evaluation=True)
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        seed=2029,
        n_steps=1024,
        batch_size=512,
        policy_kwargs={
            "features_extractor_class": FlightFeatures,
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": torch.nn.Tanh,
            "log_std_init": -3,
        },
    )
    observations, labels, history = [], [], []
    best_score, best_path = -1, None
    for iteration in range(iterations):
        collection = collect(model, iteration, observations, labels, episodes)
        mse = fit(model, observations, labels, epochs, seed=2029 + iteration)
        target = output / f"iteration-{iteration}"
        target.mkdir()
        model.save(target / "model.zip")
        report = assess(model, 16, 40000000, target / "selection-traces")
        history.append(
            {
                "iteration": iteration,
                "samples": len(labels),
                "mse": mse,
                "collection": collection,
                "selection": report,
            }
        )
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        score = min(r["successes"] for r in report)
        if score > best_score:
            best_score, best_path = score, target
        if score == 16:
            break
    model = PPO.load(best_path / "model.zip", device="cpu")
    # The validation seed namespace is never used to collect labels or select an iteration.
    validation = assess(model, 64, 60000000, best_path / "validation-traces")
    np.savez_compressed(
        output / "demonstrations.npz",
        observations=np.asarray(observations),
        actions=np.asarray(labels),
    )
    report = {
        "method": "behavior cloning with perturbed-state augmentation; replacement policy",
        "dagger_rounds": len(history) - 1,
        "configuration": {
            "iterations": iterations,
            "epochs": epochs,
            "episodes_per_stage": episodes,
            "seed": 2029,
            "batch_size": 512,
            "actor_learning_rate": 0.001,
        },
        "fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(executed_source).hexdigest(),
        "best_checkpoint": str(best_path),
        "samples": len(labels),
        "history": history,
        "validation": validation,
        "critic_trained": False,
        "ppo_updates": 0,
        "scope": "deterministic hover and stage-1 single gate in simulation",
        "passed": all(r["successes"] >= 61 for r in validation),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
    print(f"Final: {best_path}; passed={report['passed']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=16)
    run(**vars(parser.parse_args()))
