"""Sustained PPO across independent seeds, with held-out recovery release gates."""
import argparse
import hashlib
import json
import os
import shutil
from functools import partial
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from notebooks.recovery_v8 import RecoveryEnv, RecoveryFeatures, RECOVERIES, CONTRACT, GAMMA
from training.learning.train import save_trace

SEEDS = [801, 802, 803]
GROUPS = [(f'course-{level}', level, None) for level in range(-1, 21)] + [
    (f'{kind}-{level}', level, kind) for kind in RECOVERIES for level in (18, 20)]


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def transfer(source, env, seed):
    old = PPO.load(source, device='cpu')
    model = PPO('MlpPolicy', env, device='cpu', seed=seed, n_steps=1024,
        batch_size=256, n_epochs=5, learning_rate=3e-5, gamma=GAMMA,
        gae_lambda=.95, clip_range=.1, target_kl=.01, ent_coef=0.,
        policy_kwargs=dict(features_extractor_class=RecoveryFeatures,
            net_arch=dict(pi=[256,256], vf=[256,256]), activation_fn=torch.nn.Tanh,
            log_std_init=-3))
    state = model.policy.state_dict()
    for key, value in old.policy.state_dict().items():
        if key not in state or key == 'log_std': continue
        if state[key].shape == value.shape:
            state[key] = value.clone()
        elif key.endswith('.0.weight') and value.ndim == 2 and value.shape[1] == 39:
            state[key].zero_()
            state[key][:,:39] = value
        else:
            raise ValueError(f'Unexpected transfer shape for {key}: {value.shape}')
    model.policy.load_state_dict(state)
    return model


def assess(model, output, cases, base, old_only=False):
    output.mkdir(parents=True, exist_ok=True)
    results=[]
    env=RecoveryEnv()
    for group_index,(name,level,recovery) in enumerate(GROUPS):
        if old_only and (level>12 or recovery): continue
        rows=[]
        for case in range(cases):
            obs,_=env.reset(seed=base+group_index*10000+case,
                options=dict(level=level,recovery=recovery,difficulty=2))
            while True:
                obs,_,done,_,info=env.step(model.predict(obs,deterministic=True)[0])
                if done: break
            rows.append({k:info[k] for k in ('seed','outcome','reacquired_s','altitude_loss_m','remaining_passed')})
            if case==0 or (info['outcome']!='success' and sum(r['outcome']!='success' for r in rows)<=2):
                save_trace(output/f'{name}-{case}.npz',env.completed)
        results.append(dict(name=name,level=level,recovery=recovery,cases=cases,
            successes=sum(r['outcome']=='success' for r in rows),results=rows))
        dump(output/'scores.json',results)
        print(json.dumps({k:v for k,v in results[-1].items() if k!='results'}),flush=True)
    env.close()
    return results


def retains(scores, baseline):
    by_name={row['name']:row for row in scores}
    return all(all(a['outcome']!='success' or b['outcome']=='success'
        for a,b in zip(row['results'],by_name[row['name']]['results'])) for row in baseline)


def qualifies(scores):
    return (len(scores)==len(GROUPS) and {r['name'] for r in scores}=={n for n,_,_ in GROUPS}
            and all(row['cases']>=32 and row['successes']/row['cases']>=.95 for row in scores))


def warm_value(model, vec, steps=512):
    """Fit the new reward's critic from rollouts; never change actor weights."""
    obs=vec.reset();xs=[];rewards=[];dones=[]
    for _ in range(steps):
        xs.append(obs.copy())
        obs,reward,done,_=vec.step(model.predict(obs,deterministic=True)[0])
        rewards.append(reward.copy());dones.append(done.copy())
    with torch.no_grad(): value=model.policy.predict_values(torch.tensor(obs)).numpy().ravel()
    returns=[]
    for reward,done in zip(reversed(rewards),reversed(dones)):
        value=reward+GAMMA*value*(1-done);returns.append(value.copy())
    x=torch.tensor(np.asarray(xs).reshape(-1,74),dtype=torch.float32)
    y=torch.tensor(np.asarray(list(reversed(returns))).reshape(-1),dtype=torch.float32)
    params=list(model.policy.mlp_extractor.value_net.parameters())+list(model.policy.value_net.parameters())
    optimizer=torch.optim.Adam(params,lr=3e-4)
    for _ in range(10):
        for indices in torch.randperm(len(x)).split(256):
            loss=((model.policy.predict_values(x[indices]).flatten()-y[indices])**2).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,.5);optimizer.step()
    # Reset again before PPO so its first observation and environment agree.
    model._last_obs=None
    return steps*vec.num_envs


class Progress(BaseCallback):
    def __init__(self, directory, seed, budget):
        super().__init__(); self.directory=directory; self.seed=seed; self.budget=budget
    def _on_step(self):
        if self.num_timesteps%4096==0:
            difficulty=min(2,int(self.num_timesteps/max(1,self.budget)*3))
            self.training_env.set_attr('difficulty',difficulty)
            dump(self.directory.parent/'status.json',dict(phase='ppo',seed=self.seed,
                steps=self.num_timesteps,seed_budget=self.budget,difficulty=difficulty))
        return True


def train_seed(output, source, seed, steps, workers, interval, cases):
    directory=output/f'seed-{seed}';directory.mkdir()
    factories=[partial(RecoveryEnv,seed=seed,worker=i,output=directory/'rollouts',training_mix=True) for i in range(workers)]
    vec=SubprocVecEnv(factories,start_method='spawn') if workers>1 else DummyVecEnv(factories)
    model=transfer(source,vec,seed)
    model.save(directory/'initial.zip')
    dump(output/'status.json',dict(phase='baseline',seed=seed,steps=0))
    baseline=assess(model,directory/'baseline',cases,510000000,old_only=True)
    dump(output/'status.json',dict(phase='critic-initialization',seed=seed,steps=0))
    critic_steps=warm_value(model,vec)
    best=-1.;selected=directory/'initial.zip';history=[]
    try:
        while model.num_timesteps<steps:
            # Continue the latest learning state even after a failed evaluation;
            # only promotion is gated. PPO sees its own actions and consequences.
            model.learn(min(interval,steps-model.num_timesteps),reset_num_timesteps=False,
                callback=Progress(directory,seed,steps))
            checkpoint=directory/f'step-{model.num_timesteps:012d}.zip';model.save(checkpoint)
            dump(output/'status.json',dict(phase='selection',seed=seed,steps=model.num_timesteps))
            scores=assess(model,directory/f'selection-{model.num_timesteps}',cases,510000000)
            retained=retains(scores,baseline)
            score=float(np.mean([r['successes']/r['cases'] for r in scores]))
            if retained and score>best:
                best=score;selected=checkpoint
            history.append(dict(steps=model.num_timesteps,checkpoint=str(checkpoint),retained=retained,score=score))
            dump(directory/'history.json',history)
            dump(directory/'selection.json',dict(selected_model=str(selected),score=best))
        return dict(seed=seed,steps=model.num_timesteps,critic_preparation_transitions=critic_steps,selected_model=str(selected),history=history)
    finally:
        model.save(directory/'latest-resumable.zip');vec.close()


def run(output,source,steps=1048576,workers=4,interval=262144,cases=4,final_cases=32,deploy=False):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('Use a new output directory')
    if steps<=0 or workers<=0 or interval<=0 or cases<=0 or final_cases<32:
        raise ValueError('Positive budgets and at least 32 final cases are required')
    output.mkdir(parents=True,exist_ok=True)
    verification=Path('runs/recovery-v8-verification.json')
    evidence=json.loads(verification.read_text())
    if not evidence.get('python_tests_passed') or not evidence.get('browser_observation_parity_passed'):
        raise ValueError('Run 008 preflight verification has not passed')
    shutil.copyfile(verification,output/'verification.json')
    shutil.copyfile(source,output/'source.zip')
    source=output/'source.zip'
    snapshots=output/'sources';snapshots.mkdir()
    for name in ('recovery_v8.py','train_recovery_v8.py','deploy_recovery_v8.py','broad_maneuver.py','maneuver.py','agile_v4.py','agile_v2.py','agile_curriculum.py'):
        shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
    dump(output/'configuration.json',dict(contract=CONTRACT,width=74,seeds=SEEDS,steps_per_seed=steps,
        workers=workers,interval=interval,selection_cases=cases,final_cases=final_cases,
        training_seed_range=[400000000,500000000],selection_seed=510000000,final_seed=610000000,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),deploy=deploy,
        recovery_mix=.4,retention_mix=.4,normal_broad_mix=.2,gamma=GAMMA,
        minimum_success_rate=.95,release_requires_all_three_seeds=True))
    torch.set_num_threads(1)
    rows=[]
    try:
        for seed in SEEDS:
            rows.append(train_seed(output,source,seed,steps,workers,interval,cases))
            dump(output/'training-summary.json',rows)
        # Training and checkpoint selection finish before any final cases run.
        # Deployment choice is fixed from selection; final cases only gate release.
        chosen=max(rows,key=lambda r:json.loads((output/f'seed-{r["seed"]}'/'selection.json').read_text())['score'])
        for row in rows:
            dump(output/'status.json',dict(phase='final-validation',seed=row['seed']))
            model=PPO.load(row['selected_model'],device='cpu')
            scores=assess(model,output/f'seed-{row["seed"]}'/'final-validation',final_cases,610000000)
            row.update(validation=scores,qualified=qualifies(scores))
            dump(output/'training-summary.json',rows)
        ready=all(r['qualified'] for r in rows)
        report=dict(release_ready=ready,selected_model=chosen['selected_model'],selected_seed=chosen['seed'],
            seeds=rows,total_ppo_transitions=sum(r['steps'] for r in rows),observation_contract=CONTRACT)
        dump(output/'report.json',report)
        if ready and deploy:
            from notebooks.deploy_recovery_v8 import deploy as publish
            publish(output)
        dump(output/'status.json',dict(phase='deployed' if ready and deploy else 'complete',release_ready=ready))
    except BaseException as error:
        dump(output/'status.json',dict(phase='failed',error=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source',default='runs/agile-curriculum-007/accepted-model.zip')
    for key,default in [('steps',1048576),('workers',4),('interval',262144),('cases',4),('final-cases',32)]:
        parser.add_argument('--'+key,type=int,default=default)
    parser.add_argument('--deploy',action='store_true')
    run(**vars(parser.parse_args()))
