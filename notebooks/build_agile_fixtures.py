"""Python fixtures for explicit browser previous-gate observations."""
import json
from pathlib import Path
import numpy as np
from notebooks.agile_v3 import DroneEnv, reset_course

rows=[]
env=DroneEnv(evaluation=True)
for level in (0,2,5,7):
    reset_course(env,level,195000000+level)
    for target in (0, min(1,len(env.course["gates"])-1)):
        env.target=target
        env.previous=np.array([.2,.3,-.4,.5])
        env.elapsed=1.25
        rows.append(dict(course=env.course,state=env.state.tolist(),target=target,
                         previous=env.previous.tolist(),elapsed=env.elapsed,timeout=env.timeout,
                         observation=env._observation().tolist()))
Path("web/tests/fixtures/agile-observation.json").write_text(json.dumps(rows))
print(f"Wrote {len(rows)} observation fixtures")
reset_course(env,0,195000009)
np.savez_compressed("web/tests/fixtures/agile-long.npz",
    states=np.array([[0.,*env.state],[60.,*env.state]]),
    metadata_json=np.array(json.dumps(dict(format="aerorl-notebook-trace-v1",
        course=env.course,metrics=dict(stage=1,outcome="timeout",episode=dict(r=0.)),
        events=[dict(time=60.,type="timeout")],provenance="synthetic parser test only"))))
