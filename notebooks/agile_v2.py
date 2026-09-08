"""Previous-gate-aware experimental observation and clearance demonstrations."""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from notebooks.agile_curriculum import PROFILES, course_for as original_course_for, navigation_target, wrap
from training.envs.drone import DroneEnv as OriginalEnv
from training.physics.quaternion import rotation
from training.physics.controller import reference_action


class DroneEnv(OriginalEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(-1, 1, (46,), np.float32)

    def _observation(self):
        obs = super()._observation()
        previous = np.zeros(6)
        if self.course["gates"] and self.target > 0:
            gate = self.course["gates"][self.target-1]
            transform = rotation(self.state[3:7]).T
            previous[:3] = transform @ (np.array(gate["center_m"])-self.state[:3])/40
            previous[3:] = transform @ [np.cos(gate["yaw_rad"]),np.sin(gate["yaw_rad"]),0]
        return np.concatenate((obs,np.clip(previous,-1,1))).astype(np.float32)


class AgileFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=31)

    def forward(self, obs):
        return torch.cat((obs[:, :3]*3, obs[:, 3:9], obs[:, 9:12]*4,
                          obs[:, 12:15], obs[:, 15:18]*4, obs[:, 18:21],
                          obs[:, 22:26], obs[:,40:43]*4, obs[:,43:46]), dim=1)


def reset_course(env, level, seed):
    # This curriculum isolates maneuvering from obstruction by non-target gates.
    # A later obstacle-avoidance curriculum must expose all nearby frames.
    for proposal in range(100):
        course = original_course_for(env, level, seed+proposal*1000000000000)
        points = [np.array(course["start_reference_m"])]+[np.array(g["center_m"]) for g in course["gates"]]
        clear = True
        for i, (a,b) in enumerate(zip(points, points[1:])):
            d = b-a
            for j, p in enumerate(points[1:]):
                if j in (i-1,i): continue
                fraction = np.clip(np.dot(p-a,d)/np.dot(d,d),0,1)
                if np.linalg.norm(p-(a+fraction*d)) < 3.5:
                    clear = False
        if clear: break
    else:
        raise RuntimeError("Unobstructed curriculum proposal budget exhausted")
    env.reset(seed=seed,options={"course":course})
    env.timeout={1:60.,3:180.,10:600.}[len(course["gates"])]
    return env._observation(), {}


def expert(env):
    gates = env.course["gates"]
    if not gates:
        target = np.array(env.course["start_reference_m"])
        nose = rotation(env.course["initial_state"]["quaternion_wxyz"])[:,0]
        yaw = np.arctan2(nose[1],nose[0])
    else:
        gate = gates[env.target]
        center = np.array(gate["center_m"])
        yaw = gate["yaw_rad"]
        normal = np.array([np.cos(yaw),np.sin(yaw),0.])
        side = np.array([-normal[1],normal[0],0.])
        key = (env.course["course_id"],env.target)
        if getattr(env,"_expert_key",None) != key or env.steps < getattr(env,"_expert_step",0):
            queue = []
            if env.target:
                prior = gates[env.target-1]
                p = np.array(prior["center_m"])
                n = np.array([np.cos(prior["yaw_rad"]),np.sin(prior["yaw_rad"]),0.])
                s = np.array([-n[1],n[0],0.])
                queue.append(p+4*n)
                if np.dot(center-p,n)<0:
                    sign = 1 if np.dot(center-p,s)>=0 else -1
                    queue.append(p+4*n+4*sign*s)
                    queue.append(p-4*n+4*sign*s)
            origin = queue[-1] if queue else env.state[:3]
            if np.dot(origin-center,normal)>-2:
                sign = 1 if np.dot(origin-center,side)>=0 else -1
                queue.append(center+4*normal+4*sign*side)
                queue.append(center-4*normal+4*sign*side)
            queue.extend([center-4*normal,center+4*normal])
            env._expert_queue = queue
            env._expert_key = key
            env._expert_step = -1
        if env.steps != env._expert_step:
            if len(env._expert_queue)>1 and np.linalg.norm(env.state[:3]-env._expert_queue[0])<.65 and np.linalg.norm(env.state[7:10])<1.2:
                env._expert_queue.pop(0)
            env._expert_step = env.steps
        target = env._expert_queue[0]
    nose = rotation(env.state[3:7])[:,0]
    current = np.arctan2(nose[1],nose[0])
    yaw = current+np.clip(wrap(yaw-current),-np.pi/3,np.pi/3)
    return reference_action(env.state,target,yaw,env.vehicle).astype(np.float32)

if __name__ == "__main__":
    import json
    env=DroneEnv(evaluation=True)
    for level in (4,5,6):
        for case in range(4):
            reset_course(env,level,120000000+level*100000+case)
            while True:
                _,_,done,_,info=env.step(expert(env))
                if done: break
            print(json.dumps(dict(level=level,case=case,**info)),flush=True)
