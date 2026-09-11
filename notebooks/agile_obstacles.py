"""Full gate-map observations and stateless collision-aware training demonstrations.

The roadmap supplies training labels only. Neural inference consumes 109 values
and does not invoke the roadmap. Gates retain their fixed standard dimensions.
"""
import heapq
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from notebooks.agile_v4 import AgileFeatures as PreviousFeatures, PROFILES, LEVEL_MAP, reset_course as filtered_reset
from notebooks.agile_v2 import DroneEnv as PreviousEnv
from notebooks.agile_curriculum import wrap
from training.physics.collision import frame_boxes, gate_basis
from training.physics.controller import reference_action
from training.physics.quaternion import rotation


def reset_course(env, level, seed):
    if level < 6:
        return filtered_reset(env, level, seed)
    # Mixed lessons deliberately retain other frames on direct course legs.
    from notebooks.agile_curriculum import course_for
    course = course_for(env, LEVEL_MAP[level], seed)
    obs, info = env.reset(seed=seed, options={'course': course})
    env.timeout = {3: 180., 10: 600.}[len(course['gates'])]
    return obs, info


class DroneEnv(PreviousEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(-1, 1, (109,), np.float32)

    def _observation(self):
        original = super()._observation()
        obstacles = np.zeros((9, 7), np.float32)
        transform = rotation(self.state[3:7]).T
        gates = [(i, gate) for i, gate in enumerate(self.course['gates']) if i != self.target]
        gates.sort(key=lambda item: (np.linalg.norm(np.asarray(item[1]['center_m'])-self.state[:3]), item[0]))
        for slot, (_, gate) in enumerate(gates[:9]):
            obstacles[slot, :3] = transform @ (np.asarray(gate['center_m'])-self.state[:3]) / 200.
            obstacles[slot, 3:6] = transform @ gate_basis(gate)[:, 0]
            obstacles[slot, 6] = 1
        return np.concatenate((original, np.clip(obstacles.flatten(), -1, 1))).astype(np.float32)


class AgileFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=138)
        self.previous = PreviousFeatures(observation_space)

    def forward(self, obs):
        base = self.previous(obs[:, :46])
        obstacle = obs[:, 46:].reshape(-1, 9, 7)
        displacement, normal, mask = obstacle[:, :, :3]*20, obstacle[:, :, 3:6], obstacle[:, :, 6:7]
        up = -obs[:, None, 6:9].expand(-1, 9, -1)
        side = torch.linalg.cross(up, normal, dim=2)
        velocity = obs[:, None, :3]*3
        geometry = torch.cat((displacement, normal, mask,
                              (-displacement*normal).sum(2, keepdim=True),
                              (-displacement*side).sum(2, keepdim=True),
                              (-displacement*up).sum(2, keepdim=True),
                              (velocity*normal).sum(2, keepdim=True)), dim=2)*mask
        return torch.cat((base, geometry.flatten(1)), dim=1)


class Roadmap:
    """Static graph cached by geometry, with fresh position-dependent routing.

    No visited-waypoint phase is retained. Inflated frame bars, including the
    active frame, are obstacles; apertures remain traversable in either direction.
    """
    def __init__(self, env):
        centers, bases, lower, upper, nodes = [], [], [], [], []
        for gate in env.course['gates']:
            center, basis = np.asarray(gate['center_m']), gate_basis(gate)
            for _, minimum, maximum in frame_boxes(gate):
                centers.append(center); bases.append(basis)
                lower.append(np.asarray(minimum)-.8)
                upper.append(np.asarray(maximum)+.8)
            for sign in (-1, 1):
                nodes.append(center+basis @ [3*sign, 0, 0])
                for lateral in (-1, 1):
                    for vertical in (-1, 1):
                        nodes.append(center+basis @ [1.5*sign, 2.7*lateral, 2.7*vertical])
        self.centers, self.bases = np.asarray(centers), np.asarray(bases)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.minimum = np.asarray(env.rules['workspace_min_m'])+.85
        self.maximum = np.asarray(env.rules['workspace_max_m'])-.85
        self.nodes = np.asarray([node for node in nodes if np.all(node>self.minimum) and np.all(node<self.maximum)])
        size = len(self.nodes)
        starts = np.repeat(self.nodes, size, axis=0)
        ends = np.tile(self.nodes, (size, 1))
        clear = self.clear(starts, ends).reshape(size, size)
        self.edges = np.linalg.norm(self.nodes[:, None]-self.nodes[None, :], axis=2)
        self.edges[~clear] = np.inf
        self.cache = {}

    def clear(self, starts, ends, margin=.8):
        starts, ends = np.atleast_2d(starts), np.atleast_2d(ends)
        a = np.einsum('nbi,bij->nbj', starts[:, None]-self.centers, self.bases)
        b = np.einsum('nbi,bij->nbj', ends[:, None]-self.centers, self.bases)
        delta = b-a
        parallel = np.abs(delta)<1e-10
        divisor = np.where(parallel, 1., delta)
        lower, upper = self.lower+(.8-margin), self.upper-(.8-margin)
        first, second = (lower-a)/divisor, (upper-a)/divisor
        lo, hi = np.minimum(first, second), np.maximum(first, second)
        outside = parallel & ((a<lower) | (a>upper))
        lo = np.where(parallel, -np.inf, lo)
        hi = np.where(parallel, np.inf, hi)
        hit = (np.maximum(0, lo.max(2)) <= np.minimum(1, hi.min(2))) & ~outside.any(2)
        bounds = (starts>self.minimum).all(1) & (starts<self.maximum).all(1) & (ends>self.minimum).all(1) & (ends<self.maximum).all(1)
        return ~hit.any(1) & bounds

    def waypoint(self, position, goal):
        if self.clear(position, goal)[0]:
            return goal
        key = tuple(goal)
        if key not in self.cache:
            visible = self.clear(self.nodes, np.broadcast_to(goal, self.nodes.shape))
            costs = np.where(visible, np.linalg.norm(self.nodes-goal, axis=1), np.inf)
            queue = [(float(cost), i) for i, cost in enumerate(costs) if np.isfinite(cost)]
            heapq.heapify(queue)
            while queue:
                cost, node = heapq.heappop(queue)
                if cost != costs[node]:
                    continue
                for other in np.flatnonzero(np.isfinite(self.edges[node])):
                    candidate = cost+self.edges[node, other]
                    if candidate < costs[other]-1e-10:
                        costs[other] = candidate
                        heapq.heappush(queue, (float(candidate), int(other)))
            self.cache[key] = costs
        visible = self.clear(np.broadcast_to(position, self.nodes.shape), self.nodes)
        costs = np.where(visible, self.cache[key]+np.linalg.norm(self.nodes-position, axis=1), np.inf)
        if not np.isfinite(costs).any():
            # Escape the extra planning margin slowly while preserving a margin
            # greater than the physical 0.35 m radius. Holding here can deadlock.
            visible = self.clear(np.broadcast_to(position, self.nodes.shape), self.nodes, margin=.4)
            costs = np.where(visible, self.cache[key]+np.linalg.norm(self.nodes-position, axis=1), np.inf)
            if not np.isfinite(costs).any():
                return position.copy()
            direction = self.nodes[int(np.argmin(costs))]-position
            return position+direction*min(1., .5/max(np.linalg.norm(direction), 1e-9))
        return self.nodes[int(np.argmin(costs))]


def expert(env):
    if not env.course['gates']:
        from notebooks.agile_v4 import expert as hover_expert
        return hover_expert(env)
    gate = env.course['gates'][env.target]
    center, normal = np.asarray(gate['center_m']), gate_basis(gate)[:, 0]
    delta = env.state[:3]-center
    lateral = np.linalg.norm(delta-(delta@normal)*normal)
    lateral_speed = np.linalg.norm(env.state[7:10]-(env.state[7:10]@normal)*normal)
    crossing = delta@normal<0 and lateral<.4 and lateral_speed<.4
    goal = center+(3 if crossing else -3)*normal
    if getattr(env, '_roadmap_course', None) != env.course['course_id']:
        env._roadmap = Roadmap(env)
        env._roadmap_course = env.course['course_id']
    target = env._roadmap.waypoint(env.state[:3], goal)
    nose = rotation(env.state[3:7])[:, 0]
    yaw = np.arctan2(nose[1], nose[0])
    yaw += np.clip(wrap(gate['yaw_rad']-yaw), -np.pi/3, np.pi/3)
    return reference_action(env.state, target, yaw, env.vehicle).astype(np.float32)
