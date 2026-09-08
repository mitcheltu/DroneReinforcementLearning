"""Experimental next-frame avoidance; validate before using for training."""
import numpy as np
from notebooks.agile_v2 import DroneEnv, PROFILES as ORIGINAL_PROFILES, reset_course as original_reset
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from notebooks.agile_curriculum import wrap
from training.physics.quaternion import rotation
from training.physics.controller import reference_action

LEVEL_MAP=[0,1,2,3,5,4,6,7]
PROFILES=[ORIGINAL_PROFILES[i] for i in LEVEL_MAP]

def reset_course(env,level,seed):
    return original_reset(env,LEVEL_MAP[level],seed)

class AgileFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        super().__init__(observation_space,features_dim=39)

    def forward(self,obs):
        base=torch.cat((obs[:,:3]*3,obs[:,3:9],obs[:,9:12]*4,obs[:,12:15],
                        obs[:,15:18]*4,obs[:,18:21],obs[:,22:26],
                        obs[:,40:43]*4,obs[:,43:46]),dim=1)
        up=-obs[:,6:9]
        geometry=[]
        for displacement,normal in ((obs[:,9:12]*4,obs[:,12:15]),(obs[:,40:43]*4,obs[:,43:46])):
            side=torch.linalg.cross(up,normal,dim=1)
            geometry.extend([(-displacement*normal).sum(1,keepdim=True),
                             (-displacement*side).sum(1,keepdim=True),
                             (-displacement*up).sum(1,keepdim=True),
                             (obs[:,:3]*3*normal).sum(1,keepdim=True)])
        return torch.cat([base,*geometry],dim=1)


def expert(env):
    gates=env.course["gates"]
    if not gates:
        target=np.array(env.course["start_reference_m"])
        nose=rotation(env.course["initial_state"]["quaternion_wxyz"])[:,0]
        yaw=np.arctan2(nose[1],nose[0])
    else:
        gate=gates[env.target]; center=np.array(gate["center_m"]); yaw=gate["yaw_rad"]
        n=np.array([np.cos(yaw),np.sin(yaw),0.]); side=np.array([-n[1],n[0],0.])
        delta=env.state[:3]-center; x=np.dot(delta,n); y=np.dot(delta,side)
        lateral=np.linalg.norm(delta-x*n); sign=1 if y>=0 else -1
        # Crossing the current gate forward advances target immediately. Thus an
        # exit-side current-gate position needs repositioning, never an exit goal.
        lateral_speed=np.linalg.norm(env.state[7:10]-np.dot(env.state[7:10],n)*n)
        if x<0 and lateral<.65 and lateral_speed<.5:
            target=center+3*n
        elif x < -3:
            target=center-4*n
        elif abs(y)<3:
            target=center+max(2,x)*n+4*sign*side
        else:
            target=center-4*n+4*sign*side
        if env.target:
            gate=gates[env.target-1]; p=np.array(gate["center_m"])
            pn=np.array([np.cos(gate["yaw_rad"]),np.sin(gate["yaw_rad"]),0.])
            ps=np.array([-pn[1],pn[0],0.]); d=env.state[:3]-p
            px,py=np.dot(d,pn),np.dot(d,ps)
            tx=np.dot(target-p,pn)
            fraction=px/max(px-tx,1e-9)
            crossing_y=py+(np.dot(target-p,ps)-py)*fraction
            crossing_z=d[2]+(target[2]-env.state[2])*fraction
            if tx<0 and px>=0 and abs(crossing_y)<2.5 and abs(crossing_z)<2.5:
                sign=1 if np.dot(target-p,ps)>=0 else -1
                if px<1.5 and abs(py)<1:
                    target=p+3.5*pn
                else:
                    target=p+max(3,px)*pn+4*sign*ps
        # The next gate is observable too. A reversal detour around the previous
        # frame can hit it before it becomes the target. Climb above its frame
        # when the desired segment crosses it; this decision is memoryless.
        if env.target+1 < len(gates):
            obstacle=gates[env.target+1]
            p=np.asarray(obstacle['center_m'])
            pn=np.array([np.cos(obstacle['yaw_rad']),np.sin(obstacle['yaw_rad']),0.])
            ps=np.array([-pn[1],pn[0],0.])
            d=env.state[:3]-p
            px=float(d@pn); tx=float((target-p)@pn)
            fraction=px/(px-tx) if abs(px-tx)>1e-9 else -1.
            crossing=env.state[:3]+fraction*(target-env.state[:3])-p
            if 0<=fraction<=1 and abs(crossing@ps)<2.5 and abs(crossing[2])<2.5:
                # Hold the approach side while gaining vertical clearance.
                sign=1 if px>=0 else -1
                target=p+sign*max(3.,abs(px))*pn
                target[2]=p[2]+3.5
    nose=rotation(env.state[3:7])[:,0]; current=np.arctan2(nose[1],nose[0])
    yaw=current+np.clip(wrap(yaw-current),-np.pi/3,np.pi/3)
    return reference_action(env.state,target,yaw,env.vehicle).astype(np.float32)


if __name__=="__main__":
    import json
    from pathlib import Path
    rows=[]
    env=DroneEnv(evaluation=True)
    for level in (3,4,5,6,7):
        for case in range(4):
            reset_course(env,level,170000000+level*10000+case)
            while True:
                _,_,done,_,info=env.step(expert(env))
                if done: break
            rows.append(dict(level=level,name=PROFILES[level]["name"],case=case,**info))
            Path("runs/agile-v5-expert.json").write_text(json.dumps(rows,indent=2))
            print(json.dumps(rows[-1]),flush=True)
