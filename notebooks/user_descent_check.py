"""Reconstruct the visible editor values (4-decimal precision) and compare controllers."""
import copy
import json
import uuid
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from notebooks.agile_obstacles import DroneEnv
from notebooks.maneuver import expert
from training.physics.collision import CollisionWorld
from training.learning.train import save_trace

EDITED=[
    [-11.4099,-39.4677,5.8314,51.8361],[-11.6341,-29.4427,11.3224,60.0235],
    [-7.7337,-36.8696,4.5517,20.7879],[4.0266,-26.0969,7.1158,56.5801],
    [6.7918,-19.549,6.623,71.4338],[10.3372,-12.7701,7.2309,62.6424],
    [14.5144,-6.2995,7.2737,66.0041],[17.1644,.538,7.4587,76.8432],
    [20.0021,7.0151,7.2004,59.1084],[19.4174,13.3863,7.0735,103.1219]]

def run():
    root=Path('runs/user-descent-regression');root.mkdir(exist_ok=True)
    presets=json.loads(Path('web/public/courses.json').read_text())
    course=copy.deepcopy(next(row['course'] for row in presets if row['stage']==6 and row['variant']==1))
    course.update(course_id=str(uuid.uuid5(uuid.NAMESPACE_URL,'user-descent-ui-2026-09-08')),
                  name='User descent layout reconstructed from editor',mode='experimental',generator=None)
    for gate,values in zip(course['gates'],EDITED):
        gate['center_m']=[original if round(original,4)==displayed else displayed for original,displayed in zip(gate['center_m'],values[:3])]
        if round(float(np.degrees(gate['yaw_rad'])),4)!=values[3]: gate['yaw_rad']=float(np.radians(values[3]))
    (root/'course.json').write_text(json.dumps(course,indent=2))
    model=PPO.load('runs/agile-curriculum-005/level-5-round-3/model.zip',device='cpu')
    rows=[]
    for name,solid,teacher in [('run005-solid',True,False),('run005-non-solid',False,False),('teacher-non-solid',False,True)]:
        env=DroneEnv(evaluation=True,record=True)
        env.reset(seed=280900000,options={'course':course});env.timeout=600.
        if not solid: env.world=CollisionWorld([],env.vehicle,env.rules)
        while True:
            action=expert(env) if teacher else model.predict(env._observation(),deterministic=True)[0]
            _,_,done,_,info=env.step(action)
            if done: break
        info['collision_mode']='solid' if solid else 'non-solid-gates'
        env.completed['metrics'].update(info)
        save_trace(root/f'{name}.npz',env.completed)
        row=dict(name=name,outcome=info['outcome'],gates_passed=info['gates_passed'],duration=env.elapsed)
        rows.append(row);print(json.dumps(row),flush=True)
        (root/'report.json').write_text(json.dumps(dict(provenance='Reconstructed from visible editor values; up to 0.00005 m/degree rounding.',results=rows),indent=2))
        env.close()

if __name__=='__main__': run()
