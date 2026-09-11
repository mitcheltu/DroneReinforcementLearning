"""Generate actual Python observations for the browser's versioned contract test."""
import json
from pathlib import Path
from notebooks.recovery_v8 import RecoveryEnv,RECOVERIES


def main():
    rows=[];env=RecoveryEnv()
    for i,level in enumerate([-1,0,13,15,18,20]):
        env.reset(seed=703000000+i,options=dict(level=level,recovery=None if level<0 else RECOVERIES[i%5],difficulty=2))
        rows.append(dict(course=env.course,state=env.state.tolist(),target=env.target,
            previous=env.previous.tolist(),elapsed=env.elapsed,observation=env._observation().tolist()))
    env.reset(seed=703000020,options=dict(level=20))
    env.state[:3]=[-54,-53,2];env.course['gates'][0]['center_m']=[54,52,17]
    rows.append(dict(course=env.course,state=env.state.tolist(),target=0,previous=env.previous.tolist(),elapsed=0,observation=env._observation().tolist()))
    destination=Path(__file__).resolve().parents[1]/'web/tests/fixtures/recovery-observation.json'
    destination.write_text(json.dumps(rows))


if __name__=='__main__':main()
