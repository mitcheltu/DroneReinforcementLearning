"""Real Python observations for browser contract parity."""
import json
from pathlib import Path
import numpy as np
from notebooks.agile_obstacles import DroneEnv, reset_course

rows=[]
env=DroneEnv(evaluation=True)
for level in (0,5,7):
    reset_course(env,level,250000000+level)
    for target in (0,min(2,len(env.course['gates'])-1)):
        env.target=target
        env.previous=np.array([.2,.3,-.4,.5])
        env.elapsed=2.5
        rows.append(dict(course=env.course,state=env.state.tolist(),target=target,
                         previous=env.previous.tolist(),elapsed=env.elapsed,
                         observation=env._observation().tolist()))
Path('web/tests/fixtures/obstacle-observation.json').write_text(json.dumps(rows))
