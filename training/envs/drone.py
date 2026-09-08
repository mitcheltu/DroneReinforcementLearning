"""Headless Gymnasium environment. A reset samples a course; gates never move mid-flight."""

import copy

import gymnasium as gym
import numpy as np

from training.envs.course_generator import generate_course
from training.envs.observations import observation
from training.envs.settings import settings
from training.physics.collision import CollisionWorld, gate_crossing
from training.physics.controller import action_command, motor_command
from training.physics.dynamics import Quadrotor, state_array
from training.physics.quaternion import rotation, slerp


class DroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frontier=0, seed=101, evaluation=False, record=False):
        self.bundle = settings()
        self.vehicle = self.bundle["vehicle"].data
        self.rules = self.bundle["course-rules"].data
        self.training = self.bundle["training"].data
        self.physics = Quadrotor(self.vehicle)
        self.frontier = frontier
        self.evaluation = evaluation
        self.record = record
        self.initial_seed = seed
        self.action_space = gym.spaces.Box(-1, 1, (4,), np.float32)
        fields = self.bundle["observation"].data["fields"]
        self.observation_space = gym.spaces.Box(
            np.array([f["minimum"] for f in fields], dtype=np.float32),
            np.array([f["maximum"] for f in fields], dtype=np.float32),
        )
        self.completed = None

    def set_frontier(self, frontier):
        self.frontier = int(frontier)

    def reset(self, *, seed=None, options=None):
        if not hasattr(self, "state") and seed is None:
            seed = self.initial_seed
        super().reset(seed=seed)
        self.stage = self.frontier
        if not self.evaluation and self.frontier and self.np_random.random() >= 0.8:
            self.stage = int(self.np_random.integers(self.frontier))
        seeds = self.np_random.integers(0, 2**53, size=2)
        self.course = (
            copy.deepcopy(options["course"])
            if options and "course" in options
            else generate_course(
                int(seeds[0]),
                int(seeds[1]),
                self.training["curriculum"]["stages"][self.stage],
                self.bundle,
            )
        )
        self.state = state_array(self.course["initial_state"])
        self.world = CollisionWorld(self.course["gates"], self.vehicle, self.rules)
        self.timeout = {0: 5.0, 1: 12.0, 3: 20.0, 10: 45.0}[len(self.course["gates"])]
        self.elapsed, self.target, self.total_reward, self.steps = 0.0, 0, 0.0, 0
        self.tick = 0
        self.previous = np.zeros(4)
        self.done = False
        self.trace = {
            "course": self.course,
            "states": [[0.0, *self.state]],
            "commands": [],
            "observations": [],
            "actions": [],
            "rewards": [],
            "events": [],
        }
        return self._observation(), {}

    def _observation(self):
        return observation(
            self.state,
            self.course,
            self.target,
            self.previous,
            self.elapsed,
            self.timeout,
            self.bundle,
        )[0]

    def step(self, action):
        if self.done:
            raise RuntimeError("reset() is required after termination")
        obs = self._observation()
        u, collective, rates = action_command(action, self.vehicle)
        reward = -0.02 * float(np.sum((u - self.previous) ** 2))
        reason = None
        gates = self.course["gates"]
        start_time = self.elapsed
        for _ in range(2):
            before = self.state.copy()
            command, scale = motor_command(before, collective, rates, self.physics)
            h = self.physics.dt
            after = self.physics.step(before, command, h)
            hit = self.world.first_collision(before[:3], after[:3])
            crossing = (
                gate_crossing(before[:3], after[:3], gates[self.target], 0.35)
                if self.target < len(gates)
                else None
            )
            fraction = hit[0] if hit else 1.0
            reason = hit[1] if hit else None
            if crossing and crossing[1] != "gate_pass" and crossing[0] < fraction:
                self.trace["events"].append(
                    {
                        "time": self.elapsed + h * crossing[0],
                        "type": crossing[1],
                        "label": self.target + 1,
                    }
                )
            if (
                crossing
                and crossing[1] == "gate_pass"
                and (hit is None or crossing[0] < hit[0] - 1e-9)
            ):
                point = before[:3] + crossing[0] * (after[:3] - before[:3])
                center = np.array(gates[self.target]["center_m"])
                reward += (
                    0.5 * (np.linalg.norm(before[:3] - center) - np.linalg.norm(point - center))
                    + 10
                )
                self.target += 1
                self.trace["events"].append(
                    {
                        "time": self.elapsed + h * crossing[0],
                        "type": "gate_pass",
                        "label": self.target,
                    }
                )
                if self.target == len(gates):
                    fraction, reason = crossing[0], "success"
                    reward += 50
                else:
                    center = np.array(gates[self.target]["center_m"])
                    endpoint = before[:3] + fraction * (after[:3] - before[:3])
                    reward += 0.5 * (
                        np.linalg.norm(point - center) - np.linalg.norm(endpoint - center)
                    )
            elif gates:
                center = np.array(gates[self.target]["center_m"])
                endpoint = before[:3] + fraction * (after[:3] - before[:3])
                reward += 0.5 * (
                    np.linalg.norm(before[:3] - center) - np.linalg.norm(endpoint - center)
                )
            self.state = before + fraction * (after - before)
            self.state[3:7] = slerp(before[3:7], after[3:7], fraction)
            duration = h * fraction
            if self.record:
                self.trace["commands"].append([self.elapsed, *command, scale, duration])
            self.elapsed += duration
            self.tick += 1
            if fraction == 1:
                self.elapsed = self.tick * self.physics.dt
            if reason is None and (
                np.linalg.norm(self.state[7:10]) > self.vehicle["state_limit_speed_mps"]
                or np.linalg.norm(self.state[10:13]) > self.vehicle["state_limit_rate_radps"]
            ):
                reason = "state_limit"
            if not gates:
                pos_error = np.linalg.norm(self.state[:3] - self.course["start_reference_m"])
                tilt = rotation(self.state[3:7])[2, 2]
                reward += duration * (
                    1
                    - 0.5 * pos_error**2
                    - 0.1 * np.sum(self.state[7:10] ** 2)
                    - 0.1 * (1 - tilt)
                    - 0.02 * np.sum(self.state[10:13] ** 2)
                )
                if reason is None and (pos_error > 1 or tilt < np.cos(np.pi / 6)):
                    reason = "hover_departure"
            else:
                reward -= 0.2 * duration
            if self.record:
                self.trace["states"].append([self.elapsed, *self.state])
            if reason is None and self.tick >= round(self.timeout / self.physics.dt):
                reason = "success" if not gates else "timeout"
                if not gates:
                    reward += 5
            if reason is not None:
                break
        if reason is not None and reason != "success":
            reward -= 50 if gates else 5
        self.done = reason is not None
        self.previous = u.copy()
        self.steps += 1
        self.total_reward += float(reward)
        terminal_obs = self._observation()
        if self.record:
            self.trace["observations"].append(obs.tolist())
            self.trace["actions"].append([start_time, self.elapsed, *u, collective, *rates])
            self.trace["rewards"].append(float(reward))
        info = {"stage": self.stage}
        if self.done:
            self.trace["events"].append({"time": self.elapsed, "type": reason})
            info["episode"] = {"r": self.total_reward, "l": self.steps, "t": self.elapsed}
            info.update(outcome=reason, gates_passed=self.target, is_success=reason == "success")
            if self.record:
                self.trace["terminal_observation"] = terminal_obs.tolist()
                self.trace["metrics"] = info.copy()
                self.completed = self.trace
        return terminal_obs, float(reward), self.done, False, info
