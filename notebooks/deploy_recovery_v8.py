"""Export and atomically activate only a qualified run 008 policy."""
import hashlib
import json
from pathlib import Path
import os
import numpy as np
import torch
import onnx
import onnxruntime as ort
from stable_baselines3 import PPO
from training.learning.export import Actor
from notebooks.recovery_v8 import CONTRACT


def deploy(directory):
    directory=Path(directory).resolve()
    report=json.loads((directory/'report.json').read_text())
    from notebooks.train_recovery_v8 import SEEDS, qualifies
    if not report.get('release_ready') or sorted(r['seed'] for r in report['seeds'])!=SEEDS:
        raise ValueError('Deployment blocked: all three training seeds must qualify')
    if not all(qualifies(row['validation']) for row in report['seeds']):
        raise ValueError('Deployment blocked: held-out recovery or retention failure')
    evidence=json.loads((directory/'verification.json').read_text())
    if not evidence.get('python_tests_passed') or not evidence.get('browser_observation_parity_passed'):
        raise ValueError('Deployment blocked: missing contract verification')
    chosen=next(row for row in report['seeds'] if row['seed']==report['selected_seed'])
    if chosen['selected_model']!=report['selected_model']:
        raise ValueError('Selected checkpoint does not match the evaluated checkpoint')
    model=PPO.load(report['selected_model'],device='cpu')
    if model.observation_space.shape!=(74,): raise ValueError('Unexpected policy width')
    torch.set_num_threads(1)
    actor=Actor(model.policy).eval()
    target=directory/'recovery-008.onnx'
    torch.onnx.export(actor,torch.zeros(1,74),str(target),opset_version=17,
        input_names=['observation'],output_names=['action'],dynamic_axes={'observation':{0:'batch'},'action':{0:'batch'}},dynamo=False)
    onnx.checker.check_model(onnx.load(target))
    samples=[]
    for path in sorted((directory/f'seed-{report["selected_seed"]}'/'final-validation').glob('*.npz')):
        with np.load(path,allow_pickle=False) as trace:
            values=trace['observations'];samples.extend(values[::max(1,len(values)//64)])
    values=np.asarray(samples,dtype=np.float32)
    if values.ndim!=2 or values.shape[1]!=74 or len(values)<32:
        raise ValueError('Missing held-out observations for parity')
    session=ort.InferenceSession(str(target),providers=['CPUExecutionProvider'])
    actual=session.run(None,{'observation':values})[0]
    with torch.no_grad():expected=actor(torch.from_numpy(values)).numpy()
    error=float(np.max(np.abs(actual-expected)))
    if not np.isfinite(actual).all() or error>1e-5: raise ValueError(f'ONNX parity failure: {error}')
    # Also compare closed-loop actions for the exact chosen observations; browser
    # geometry is separately parity-tested before this pipeline is launched.
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    destination=Path(__file__).resolve().parents[1]/'web/public/models'
    filename=f'recovery-008-{digest[:12]}.onnx'
    (destination/filename).write_bytes(target.read_bytes())
    manifest=dict(file=filename,sha256=digest,input_dimension=74,observation_contract=CONTRACT,
        release_ready=True,checkpoint=report['selected_model'],max_absolute_error=error,
        parity_samples=len(values),training_seeds=SEEDS,collision_mode='non-solid-gates')
    old=destination/'active-policy.json'
    (directory/'previous-active-policy.json').write_bytes(old.read_bytes())
    temporary=destination/'active-policy.tmp'
    temporary.write_text(json.dumps(manifest,indent=2));os.replace(temporary,old)
    (directory/'deployment.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('directory')
    deploy(parser.parse_args().directory)
