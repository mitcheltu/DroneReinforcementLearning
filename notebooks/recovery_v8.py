"""Run 008: lossless workspace geometry and deliberately recoverable resets."""
import copy
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from notebooks.maneuver import DroneEnv as Parent
from notebooks.agile_v4 import AgileFeatures as LegacyFeatures
from notebooks.broad_maneuver import reset_course, PROFILES
from training.physics.quaternion import from_euler, rotation

CONTRACT = 'recovery-workspace-v8'
WIDTH = 74
GAMMA = .995
RECOVERIES = ['lateral_drift', 'overshoot', 'wrong_height', 'wrong_heading', 'combined']


class RecoveryFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=113)
        self.legacy = LegacyFeatures(gym.spaces.Box(-1, 1, (46,), np.float32))

    def forward(self, obs):
        # Exact transfer is possible by zeroing newly added first-layer columns.
        # Every observation field, including clearance and boundary distances,
        # remains available to the actor and critic after transfer.
        return torch.cat((self.legacy(obs[:, :46]), obs), dim=1)


class RecoveryEnv(Parent):
    def __init__(self, seed=801, worker=0, output=None, training_mix=False, **kwargs):
        super().__init__(evaluation=True, **kwargs)
        self.observation_space = gym.spaces.Box(-1, 1, (WIDTH,), np.float32)
        self.rng = np.random.default_rng(seed + worker*100000)
        self.output = Path(output) if output else None
        self.worker = worker
        self.training_mix = training_mix
        self.difficulty = 0
        self.episodes = 0
        self._constructing = False

    def _observation(self):
        old = super()._observation()
        transform = rotation(self.state[3:7]).T
        geometry = []
        gates = self.course['gates']
        for slot in (self.target, self.target+1, self.target-1):
            if 0 <= slot < len(gates):
                delta = transform @ (np.array(gates[slot]['center_m'])-self.state[:3])
                distance = np.linalg.norm(delta)
                geometry.extend([*(delta/200), *(delta/max(distance, 1e-9)), distance/200])
            else:
                geometry.extend([0.]*7)
        # Max workspace diagonal <171 m, so /200 preserves all valid distances.
        clearance = (self.state[2]-self.vehicle['collision_radius_m'])/20
        margins = []
        for axis in range(3):
            scale = 20 if axis == 2 else 120
            margins.extend([(self.state[axis]-.3-self.rules['workspace_min_m'][axis])/scale,
                            (self.rules['workspace_max_m'][axis]-self.state[axis]-.3)/scale])
        return np.r_[old, np.clip([*geometry, clearance, *margins], -1, 1)].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if self._constructing or (options and 'course' in options):
            return super().reset(seed=seed, options=options)
        options = options or {}
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        episode_seed = int(self.rng.integers(400000000, 500000000)) if seed is None else seed
        if 'level' in options:
            level = int(options['level'])
            recovery = options.get('recovery')
        else:
            # 40% retention, 20% normal broad routes, 40% recovery starts.
            draw = self.rng.random()
            level = int(self.rng.choice([-1, *range(13)] if draw < .4 else list(range(13, 21))))
            recovery = str(self.rng.choice(RECOVERIES)) if draw >= .6 else None
        self.level, self.recovery = level, recovery
        self.record = not self.training_mix or self.episodes % 32 == 0
        self.episodes += 1
        self._constructing = True
        try:
            if level < 0:
                self.frontier = 0
                super().reset(seed=episode_seed)
            else:
                reset_course(self, level, episode_seed)
            course = copy.deepcopy(self.course)
            target = 0
            if recovery and course['gates']:
                rng = self.rng
                target = int(rng.integers(len(course['gates'])))
                gate = course['gates'][target]
                center = np.array(gate['center_m'])
                yaw = gate['yaw_rad']
                normal = np.array([np.cos(yaw), np.sin(yaw), 0.])
                side = np.array([-normal[1], normal[0], 0.])
                strength = (.35, .65, 1.)[min(2, int(options.get('difficulty', self.difficulty)))]
                distance = rng.uniform(8, 45)*strength
                position = center-distance*normal
                velocity = np.zeros(3)
                if recovery in ('lateral_drift', 'combined'):
                    position += side*rng.uniform(-20, 20)*strength
                    velocity += side*rng.choice([-1, 1])*rng.uniform(1, 4)*strength
                if recovery in ('overshoot', 'combined'):
                    position = center+rng.uniform(2, 12)*strength*normal+(position-(center-distance*normal))
                    velocity += normal*rng.uniform(1, 4)*strength
                if recovery in ('wrong_height', 'combined'):
                    position[2] += rng.choice([-1, 1])*rng.uniform(4, 10)*strength
                    velocity[2] = rng.uniform(-2, 2)*strength
                if recovery in ('wrong_heading', 'combined'):
                    yaw += rng.choice([-1, 1])*rng.uniform(np.pi/2, np.pi)
                position = np.clip(position, [-52, -52, 2.5], [52, 52, 17])
                # Avoid downward launches that are physically unrecoverable.
                if position[2] < 4: velocity[2] = max(0., velocity[2])
                if position[2] > 15: velocity[2] = min(0., velocity[2])
                course['initial_state'].update(position_m=position.tolist(),
                    quaternion_wxyz=from_euler(*rng.uniform(-.2, .2, 2)*strength, yaw).tolist(),
                    velocity_world_mps=velocity.tolist(), omega_body_radps=rng.uniform(-.3, .3, 3).tolist())
            super().reset(seed=episode_seed, options={'course':course})
            self.target = self.start_target = target
            self.stage = 0 if level < 0 else {1:1, 3:4, 10:6}[len(course['gates'])]
            self.timeout = {0:5., 1:60., 3:180., 10:600.}[len(course['gates'])]
            self.episode_seed = episode_seed
            self.reacquired_s = None
            self.start_height = self.state[2]
            self.min_height = self.state[2]
            self.trace.update(observation_contract=CONTRACT, recovery=recovery, start_target=target)
            return self._observation(), {}
        finally:
            self._constructing = False

    def potential(self):
        if self.target >= len(self.course['gates']): return 0.
        gate = self.course['gates'][self.target]
        delta = self.state[:3]-np.array(gate['center_m'])
        x = delta @ np.array([np.cos(gate['yaw_rad']), np.sin(gate['yaw_rad']), 0.])
        # Penalize being on the exit side of an unpassed gate. Returning to its
        # entrance can now earn progress even while distance to centre grows.
        return -float(np.tanh((np.linalg.norm(delta)+2*max(0., x))/20))

    def step(self, action):
        phi = self.potential()
        previous = self.previous.copy()
        old_target = self.target
        old_elapsed = self.elapsed
        obs, old_reward, done, truncated, info = super().step(action)
        self.min_height = min(self.min_height, self.state[2])
        if self.target > self.start_target and self.reacquired_s is None:
            self.reacquired_s = self.elapsed
        reward = old_reward
        if self.course['gates']:
            clipped = np.clip(action, -1, 1)
            reward = (-.02*np.sum((clipped-previous)**2)-.2*(self.elapsed-old_elapsed)
                      +10*(self.target-old_target)+10*((0 if done else GAMMA*self.potential())-phi))
            if done: reward += 50 if info['outcome'] == 'success' else -50
            self.total_reward += float(reward-old_reward)
            if self.record: self.trace['rewards'][-1] = float(reward)
        if done:
            info['episode']['r'] = self.total_reward
            info.update(level=self.level, recovery=self.recovery, start_target=self.start_target,
                reacquired_s=self.reacquired_s, altitude_loss_m=float(max(0, self.start_height-self.min_height)),
                remaining_passed=self.target-self.start_target, seed=self.episode_seed)
            if self.record:
                self.completed['metrics'] = info.copy()
            if self.output:
                self.output.mkdir(parents=True, exist_ok=True)
                with (self.output/f'episodes-worker-{self.worker}.jsonl').open('a') as handle:
                    handle.write(json.dumps(info)+'\n')
                if self.record:
                    from training.learning.train import save_trace
                    save_trace(self.output/f'traces-worker-{self.worker}'/f'{self.episodes:07d}.npz', self.completed)
        return obs, float(reward), done, truncated, info
