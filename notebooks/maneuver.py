"""Non-solid gate curriculum; original physics and checkpoints remain unchanged."""
import numpy as np
from notebooks.agile_v4 import AgileFeatures, PROFILES as OLD, reset_course as old_reset
from notebooks.agile_v2 import DroneEnv as PreviousEnv
from training.physics.collision import CollisionWorld, gate_basis
from training.physics.controller import reference_action
from training.physics.quaternion import rotation
from notebooks.agile_curriculum import wrap

PROFILES=OLD+[
    dict(name='climbs',count=3),dict(name='descents',count=3),
    dict(name='alternating_heights',count=3),dict(name='descending_turns',count=3),
    dict(name='maneuver_ten',count=10)]

class DroneEnv(PreviousEnv):
    def reset(self, *, seed=None, options=None):
        result=super().reset(seed=seed,options=options)
        self.world=CollisionWorld([],self.vehicle,self.rules)
        self.trace['collision_mode']='non-solid-gates'
        return result

    def step(self,action):
        result=super().step(action)
        if result[2]:
            result[4]['collision_mode']='non-solid-gates'
            if self.completed is not None:
                self.completed['metrics']['collision_mode']='non-solid-gates'
        return result

def reset_course(env,level,seed):
    if level<8:
        return old_reset(env,level,seed)
    rng=np.random.default_rng(seed)
    base={8:2,9:2,10:2,11:3,12:7}[level]
    for proposal in range(100):
        old_reset(env,base,seed+proposal*1000000)
        course=env.course
        start=float(rng.uniform(3,5) if level==8 else rng.uniform(10,13) if level in (9,11) else rng.uniform(5,10))
        course['start_reference_m'][2]=start
        course['initial_state']['position_m'][2]=start
        height=start
        for i,gate in enumerate(course['gates']):
            change=rng.uniform(1.5,2.5)
            if level in (9,11): change=-change
            elif level in (10,12): change*=(-1 if i%2==0 else 1)
            height=float(np.clip(height+change,2.5,14))
            gate['center_m'][2]=height
        # Reject overlapping or nearly coincident gates regardless of yaw.
        points=np.asarray([g['center_m'] for g in course['gates']])
        if all(np.linalg.norm(a-b)>=5 for i,a in enumerate(points) for b in points[i+1:]): break
    else: raise RuntimeError('Separated maneuver course proposal budget exhausted')
    course['name']=f'Maneuver {PROFILES[level]["name"]} seed {seed}'
    obs,info=env.reset(seed=seed,options={'course':course})
    env.timeout={3:180.,10:600.}[len(course['gates'])]
    return env._observation(),info

def expert(env):
    if not env.course['gates']:
        from notebooks.agile_v4 import expert as hover
        return hover(env)
    gate=env.course['gates'][env.target]
    center=np.asarray(gate['center_m']); normal=gate_basis(gate)[:,0]
    delta=env.state[:3]-center
    lateral=np.linalg.norm(delta-(delta@normal)*normal)
    speed=np.linalg.norm(env.state[7:10]-(env.state[7:10]@normal)*normal)
    # Descend/climb to the entrance first; crossing remains directional.
    target=center+(3 if delta@normal<0 and lateral<.5 and speed<.5 else -3)*normal
    nose=rotation(env.state[3:7])[:,0]; yaw=np.arctan2(nose[1],nose[0])
    yaw+=np.clip(wrap(gate['yaw_rad']-yaw),-np.pi/3,np.pi/3)
    return reference_action(env.state,target,yaw,env.vehicle).astype(np.float32)
