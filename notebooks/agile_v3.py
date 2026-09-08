"""Observable, memoryless maneuver demonstrations for the 46-value policy."""
import numpy as np
from notebooks.agile_v2 import DroneEnv, AgileFeatures, PROFILES as ORIGINAL_PROFILES, reset_course as original_reset
from notebooks.agile_curriculum import wrap
from training.physics.quaternion import rotation
from training.physics.controller import reference_action

LEVEL_MAP=[0,1,2,3,5,4,6,7]
PROFILES=[ORIGINAL_PROFILES[i] for i in LEVEL_MAP]

def reset_course(env,level,seed):
    return original_reset(env,LEVEL_MAP[level],seed)


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
        if x<0 and lateral<.7:
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
            if np.dot(target-p,pn)<0 and px>=0:
                sign=1 if np.dot(target-p,ps)>=0 else -1
                if px<1.5 and abs(py)<1:
                    target=p+3.5*pn
                elif abs(py)<3:
                    target=p+max(3,px)*pn+4*sign*ps
                else:
                    target=p-4*pn+4*sign*ps
    nose=rotation(env.state[3:7])[:,0]; current=np.arctan2(nose[1],nose[0])
    yaw=current+np.clip(wrap(yaw-current),-np.pi/3,np.pi/3)
    return reference_action(env.state,target,yaw,env.vehicle).astype(np.float32)


if __name__=="__main__":
    import json
    env=DroneEnv(evaluation=True)
    for level in (4,5,6,7):
        for case in range(4):
            reset_course(env,level,170000000+level*10000+case)
            while True:
                _,_,done,_,info=env.step(expert(env))
                if done: break
            print(json.dumps(dict(level=level,case=case,**info)),flush=True)
