"""Gated imitation/DAgger curriculum followed by guarded PPO, with retained evidence."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from notebooks.agile_curriculum import AgileFeatures, PROFILES, expert, reset_course
from notebooks.train_imitation import fit
from training.envs.drone import DroneEnv
from training.learning.train import fingerprint, save_trace
from training.physics.quaternion import from_euler, multiply


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def record(env, output, phase, level, seed):
    trace = env.completed
    states = np.asarray(trace["states"])
    indices = np.unique(np.r_[np.arange(0, len(states), max(1, int(np.ceil(len(states)/600)))), len(states)-1])
    row = dict(source=phase, level=level, seed=int(seed), stage=env.stage,
               course=env.course, outcome=trace["metrics"]["outcome"],
               duration=env.elapsed, reward=env.total_reward,
               samples=states[indices, :8].tolist())
    with (output / "trajectories.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row)+"\n")


def reset(env, level, seed):
    if level < 0:
        env.frontier = 0
        return env.reset(seed=seed)[0]
    obs, _ = reset_course(env, level, seed)
    env.stage = {1: 1, 3: 4, 10: 6}[len(env.course["gates"])]
    return obs


def assess(model, levels, cases, base, output, phase):
    env = DroneEnv(evaluation=True, record=True)
    rows = []
    for level in levels:
        results = []
        for case in range(cases):
            seed = base + (level+1)*10000 + case
            obs = reset(env, level, seed)
            while True:
                obs, _, done, _, info = env.step(model.predict(obs, deterministic=True)[0])
                if done:
                    break
            record(env, output, phase, level, seed)
            results.append(dict(seed=seed, outcome=info["outcome"], gates_passed=info["gates_passed"], **info["episode"]))
            if case == 0:
                save_trace(output / phase / f"level-{level}.npz", env.completed)
        row = dict(level=level, name="hover" if level < 0 else PROFILES[level]["name"],
                   successes=sum(r["outcome"] == "success" for r in results), cases=cases,
                   outcomes=dict(Counter(r["outcome"] for r in results)), results=results)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != "results"}), flush=True)
    env.close()
    return rows


def collect(model, level, round_index, episodes, output, observations, labels):
    env = DroneEnv(evaluation=True, record=True)
    rng = np.random.default_rng(123000000+level*1000+round_index)
    outcomes = Counter()
    for case in range(episodes):
        seed = 120000000 + level*100000 + round_index*1000 + case
        obs = reset(env, level, seed)
        step = 0
        while True:
            action = expert(env)
            if step % 3 == 0:
                observations.append(obs.copy()); labels.append(action.copy())
                original = env.state.copy()
                env.state[:3] += rng.normal(0, .15, 3)
                env.state[7:10] += rng.normal(0, .2, 3)
                env.state[3:7] = multiply(original[3:7], from_euler(*rng.normal(0, .08, 3)))
                observations.append(env._observation()); labels.append(expert(env))
                env.state = original
            # DAgger visits learner states. Expert rollouts remain in every round.
            if round_index and case % 2:
                action = model.predict(obs, deterministic=True)[0]
            obs, _, done, _, info = env.step(action)
            step += 1
            if done:
                outcomes[info["outcome"]] += 1
                record(env, output, "demonstration" if not round_index or case%2==0 else "dagger", level, seed)
                break
        print(f"Collect level {level} round {round_index} episode {case+1}/{episodes}: {info['outcome']}", flush=True)
    env.close()
    return dict(outcomes)


class CurriculumEnv(DroneEnv):
    def __init__(self, level, output):
        super().__init__(evaluation=True, record=True)
        self.level, self.output = level, output
        self.rng = np.random.default_rng(140000000+level)

    def reset(self, *, seed=None, options=None):
        # The helper internally calls base reset; do not recurse through this override.
        chosen = int(self.rng.choice([-1]+list(range(self.level+1)))) if self.rng.random()<.4 else self.level
        chosen_seed = int(self.rng.integers(140000000, 150000000))
        self.chosen, self.chosen_seed = chosen, chosen_seed
        helper = DroneEnv(evaluation=True)
        obs = reset(helper, chosen, chosen_seed)
        _, info = super().reset(seed=chosen_seed, options={"course": helper.course})
        self.stage, self.timeout = helper.stage, helper.timeout
        helper.close()
        return self._observation(), info

    def step(self, action):
        result = super().step(action)
        if result[2]:
            record(self, self.output, "ppo", self.chosen, self.chosen_seed)
        return result


def warm_critic(model, vec, steps=2048):
    obs = vec.reset(); xs, rewards, dones = [], [], []
    for _ in range(steps):
        xs.append(obs[0].copy())
        obs, reward, done, _ = vec.step(model.predict(obs, deterministic=True)[0])
        rewards.append(float(reward[0])); dones.append(bool(done[0]))
    returns = []; value = 0.
    for reward, done in zip(reversed(rewards), reversed(dones)):
        value = reward + .995*value*(not done); returns.append(value)
    x = torch.tensor(np.asarray(xs), dtype=torch.float32)
    y = torch.tensor(list(reversed(returns)), dtype=torch.float32)
    params = list(model.policy.mlp_extractor.value_net.parameters())+list(model.policy.value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-4)
    for _ in range(30):
        for indices in torch.randperm(len(x)).split(256):
            prediction = model.policy.predict_values(x[indices]).flatten()
            loss = ((prediction-y[indices])**2).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.); optimizer.step()


def run(output, episodes=12, epochs=100, rounds=3, ppo_steps=8192, max_level=7, final_cases=16):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(2031); np.random.seed(2031)
    for name in ("train_agile.py", "agile_curriculum.py"):
        (output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    dump(output/"configuration.json", dict(profiles=PROFILES, episodes=episodes, epochs=epochs,
         rounds=rounds, ppo_steps=ppo_steps, max_level=max_level, final_cases=final_cases,
         original_fingerprint=fingerprint(), promotion="all selection cases in current and sampled retained levels",
         observation="unchanged public 40 float32", inference="neural actor only; no expert/planner calls",
         timeout_s={1:60,3:180,10:600}, seed=2031))
    model = PPO("MlpPolicy", DroneEnv(), device="cpu", seed=2031, n_steps=512, batch_size=256,
                n_epochs=3, learning_rate=2e-5, gamma=.995, gae_lambda=.95, clip_range=.05,
                target_kl=.005, ent_coef=0., max_grad_norm=.5,
                policy_kwargs=dict(features_extractor_class=AgileFeatures, net_arch=dict(pi=[256,256],vf=[256,256]),
                                   activation_fn=torch.nn.Tanh, log_std_init=-4))
    with np.load("runs/imitation-repair/demonstrations.npz") as data:
        observations, labels = list(data["observations"]), list(data["actions"])
    history = []; mastered = -1; total_ppo = 0
    try:
        for level in range(max_level+1):
            retained = sorted(set([-1, 0, max(0,level-1), level]))
            accepted = False
            for round_index in range(rounds):
                collection = collect(model, level, round_index, episodes, output, observations, labels)
                mse = fit(model, observations, labels, epochs, 2031+level*10+round_index)
                candidate = output / f"level-{level}-round-{round_index}"
                candidate.mkdir(); model.save(candidate/"model.zip")
                scores = assess(model, retained, 4, 130000000, output, f"selection-{level}-{round_index}")
                history.append(dict(level=level, round=round_index, collection=collection, samples=len(labels), mse=mse, selection=scores))
                dump(output/"history.json", history)
                accepted = all(r["successes"] == r["cases"] for r in scores)
                if accepted:
                    break
            if not accepted:
                print(f"Promotion blocked at level {level}; preserving evidence and stopping escalation", flush=True)
                break
            # Warm only the critic, then perform genuine PPO updates. A candidate
            # that loses deterministic selection successes is rolled back.
            vec = DummyVecEnv([lambda: CurriculumEnv(level, output)])
            model.set_env(vec)
            warm_critic(model, vec)
            protected = copy.deepcopy(model.policy.state_dict())
            model.learn(ppo_steps, reset_num_timesteps=False)
            total_ppo += ppo_steps
            model.save(candidate/"ppo-candidate.zip")
            ppo_scores = assess(model, retained, 4, 130000000, output, f"ppo-selection-{level}")
            ppo_accepted = all(r["successes"] == r["cases"] for r in ppo_scores)
            if not ppo_accepted:
                model.policy.load_state_dict(protected)
                model.policy.optimizer.state.clear()
            history[-1].update(ppo_selection=ppo_scores, ppo_accepted=ppo_accepted)
            mastered = level
            model.save(output/"accepted-model.zip")
            dump(output/"history.json", history)
            vec.close()
        model.save(output/"last-model.zip")
        # Select only the last curriculum-promoted policy, never a failed advanced fit.
        selected = output/"accepted-model.zip"
        if selected.exists():
            model = PPO.load(selected, device="cpu")
        validation = assess(model, [-1]+list(range(max_level+1)), final_cases, 160000000, output, "final-validation")
        original = PPO.load("runs/imitation-repair/iteration-0/model.zip", device="cpu")
        baseline = assess(original, [-1]+list(range(max_level+1)), final_cases, 160000000, output, "original-baseline")
        dump(output/"report.json", dict(mastered_level=mastered, total_ppo_transitions=total_ppo,
             selected_model=str(selected if selected.exists() else output/"last-model.zip"),
             validation=validation, baseline=baseline, history=history,
             release_ready=mastered==max_level and all(r["successes"]/r["cases"]>=.95 for r in validation)))
    finally:
        model.save(output/"interruption-model.zip")
        np.savez_compressed(output/"labels.npz", observations=np.asarray(observations), actions=np.asarray(labels))
        dump(output/"history.json", history)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--episodes",type=int,default=12)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--ppo-steps",type=int,default=8192)
    parser.add_argument("--max-level",type=int,default=7)
    parser.add_argument("--final-cases",type=int,default=16)
    run(**vars(parser.parse_args()))
