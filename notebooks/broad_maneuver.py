"""Seeded non-solid navigation stress curriculum within the simulator workspace."""
import copy
import uuid
import numpy as np
from notebooks.maneuver import DroneEnv, expert, PROFILES as OLD, reset_course as old_reset
from training.physics.quaternion import from_euler
from notebooks.agile_curriculum import wrap

PROFILES = OLD + [
    dict(name='lateral_small', count=3, span=(8, 16)),
    dict(name='lateral_large', count=3, span=(20, 35)),
    dict(name='lateral_extreme', count=3, span=(40, 70)),
    dict(name='backtracking', count=3, span=(15, 45)),
    dict(name='independent_headings', count=3, span=(10, 40)),
    dict(name='steep_zigzag', count=3, span=(12, 40)),
    dict(name='near_coincident', count=3, span=(0.5, 3)),
    dict(name='broad_ten', count=10, span=(8, 70)),
]
LESSONS = [13, 14, 16, 17, 18, 15, 19, 20]


def reset_course(env, level, seed):
    if level < len(OLD):
        return old_reset(env, level, seed)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    course = copy.deepcopy(env.course)
    profile = PROFILES[level]
    angle = rng.uniform(-np.pi, np.pi)
    transform = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Local sideways coordinate is relative to the drone's initial forward axis.
    sign = rng.choice([-1, 1])
    span = rng.uniform(*profile['span'])
    start = np.array([0., -sign*span/2, rng.uniform(4, 13)])
    if level in (17, 18, 19, 20):
        start[:2] = rng.uniform(-15, 15, 2)
    points = [start]
    for i in range(profile['count']):
        if level in (13, 14, 15):
            point = np.array([rng.uniform(-3, 3), sign*(-1)**i*span/2, start[2]+rng.uniform(-1, 1)])
        elif level == 16:
            point = np.array([(-1)**(i+1)*span/2, rng.uniform(-6, 6), start[2]+rng.uniform(-1, 1)])
        elif level == 19:
            direction = rng.uniform(-np.pi, np.pi)
            point = points[-1]+[span*np.cos(direction), span*np.sin(direction), rng.uniform(-.25, .25)]
        else:
            for _ in range(1000):
                point = np.r_[rng.uniform(-35, 35, 2), rng.uniform(2.5, 16)]
                distance = np.linalg.norm(point[:2]-points[-1][:2])
                if profile['span'][0] <= distance <= profile['span'][1]:
                    break
            else:
                raise RuntimeError('Broad course proposal budget exhausted')
            if level == 18:
                point[2] = rng.uniform(2.5, 4.5) if i % 2 == 0 else rng.uniform(12, 16)
        points.append(point)
    for point in points:
        point[:2] = transform @ point[:2]
    course.update(course_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f'broad-v1:{level}:{seed}')),
                  name=f'Broad {profile["name"]} seed {seed}', mode='experimental', generator=None)
    gates = []
    for i, point in enumerate(points[1:]):
        delta = point-points[i]
        heading = np.arctan2(delta[1], delta[0])
        # Begin with aligned gates, then decouple rotation from travel direction.
        yaw = heading+rng.uniform(-.25, .25) if level in (13, 14, 15, 16) else rng.uniform(-np.pi, np.pi)
        gates.append(dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f'{course["course_id"]}:{i}')),
                          label=i+1, center_m=point.tolist(), yaw_rad=float(wrap(yaw)), **env.rules['gate']))
    course['gates'] = gates
    course['start_reference_m'] = points[0].tolist()
    course['initial_state'].update(position_m=points[0].tolist(),
        quaternion_wxyz=from_euler(0, 0, angle).tolist(),
        velocity_world_mps=[0., 0., 0.], omega_body_radps=[0., 0., 0.])
    _, info = env.reset(seed=seed, options={'course':course})
    env.stage = 4 if profile['count'] == 3 else 6
    env.timeout = 180. if profile['count'] == 3 else 600.
    return env._observation(), info
