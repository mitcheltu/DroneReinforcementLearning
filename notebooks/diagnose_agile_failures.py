"""Replay failed v4 expert demonstrations and retain terminal obstacle evidence."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np

def run(directory, version='v4', limit=7):
    module = importlib.import_module(f'notebooks.agile_{version}')
    DroneEnv, expert = module.DroneEnv, module.expert
    directory = Path(directory)
    rows = []
    # Snapshot the complete lines; the training process can continue appending.
    lines = (directory / 'trajectories.jsonl').read_text().splitlines()
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record['source'] != 'demonstration' or record['outcome'] == 'success':
            continue
        env = DroneEnv(evaluation=True)
        env.reset(seed=record['seed'], options={'course': record['course']})
        env.timeout = min(300., {1: 60., 3: 180., 10: 600.}[len(env.course['gates'])])
        while True:
            _, _, done, _, info = env.step(expert(env))
            if done:
                break
        nearest = []
        for gate in env.course['gates']:
            delta = env.state[:3] - np.asarray(gate['center_m'])
            yaw = gate['yaw_rad']
            normal = np.array([np.cos(yaw), np.sin(yaw), 0])
            side = np.array([-np.sin(yaw), np.cos(yaw), 0])
            nearest.append(dict(label=gate['label'], center_distance_m=float(np.linalg.norm(delta)),
                                normal_m=float(delta @ normal), side_m=float(delta @ side),
                                vertical_m=float(delta[2])))
        nearest.sort(key=lambda item: item['center_distance_m'])
        row = dict(seed=record['seed'], level=record['level'], recorded_outcome=record['outcome'],
                   replay_outcome=info['outcome'], gates_passed=info['gates_passed'],
                   target_label=min(env.target+1, len(env.course['gates'])),
                   elapsed_s=env.elapsed, terminal_state=env.state.tolist(), nearest_gates=nearest)
        rows.append(row)
        (directory / f'expert-failure-diagnostics-{version}.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k not in ('terminal_state', 'nearest_gates')}
                         | {'nearest_gate': nearest[0]}), flush=True)
        env.close()
        if len(rows)>=limit:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--version', choices=['v4', 'v5', 'obstacles'], default='v4')
    parser.add_argument('--limit', type=int, default=7)
    run(**vars(parser.parse_args()))
