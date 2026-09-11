"""Balanced maneuver distillation with retention gates and guarded PPO."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from notebooks.maneuver import DroneEnv, PROFILES, reset_course, expert
from notebooks.train_agile_obstacles import record, warm_critic
from training.learning.train import save_trace, fingerprint

def dump(path,value):
    path.write_text(json.dumps(value,indent=2))

def reset(env,level,seed):
    if level<0:
        env.frontier=0
        return env.reset(seed=seed)[0]
    obs,_=reset_course(env,level,seed)
    env.stage={1:1,3:4,10:6}[len(env.course['gates'])]
    return obs

def episode(model,level,seed,output,phase,teacher=False,labels=False):
    env=DroneEnv(evaluation=True,record=True)
    obs=reset(env,level,seed)
    xs,ys=[],[]
    while True:
        action=expert(env) if teacher else model.predict(obs,deterministic=True)[0]
        if labels and env.steps%3==0:
            xs.append(obs.copy())
            ys.append(expert(env) if phase!='retention-anchor' else model.predict(obs,deterministic=True)[0])
        obs,_,done,_,info=env.step(action)
        if done: break
    record(env,output,phase,level,seed)
    summary=dict(seed=seed,outcome=info['outcome'],gates_passed=info['gates_passed'],
                 duration=env.elapsed,steps=env.steps,reward=env.total_reward)
    if labels:
        indices=np.linspace(0,len(xs)-1,min(512,len(xs)),dtype=int)
        data=(np.asarray(xs)[indices],np.asarray(ys)[indices])
    else: data=None
    trace=env.completed
    env.close()
    return summary,data,trace

def assess(model,levels,cases,base,output,phase):
    rows=[]
    for level in levels:
        results=[]
        for case in range(cases):
            result,_,trace=episode(model,level,base+(level+1)*10000+case,output,phase)
            results.append(result)
            if case==0: save_trace(output/phase/f'level-{level}.npz',trace)
        row=dict(level=level,name='hover' if level<0 else PROFILES[level]['name'],
                 successes=sum(r['outcome']=='success' for r in results),cases=cases,results=results)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='results'}),flush=True)
    return rows

def retains(scores,reference):
    lookup={r['level']:r for r in scores}
    return all(all(a['outcome']!='success' or b['outcome']=='success'
                   for a,b in zip(row['results'],lookup[row['level']]['results'])) for row in reference)

def fit(model,anchors,pools,epochs,seed,learning_rate=1e-4):
    rng=np.random.default_rng(seed)
    ax,ay=[np.concatenate([item[i] for item in anchors]) for i in (0,1)]
    groups=[tuple(np.concatenate([item[i] for item in data]) for i in (0,1)) for data in pools.values() if data]
    optimizer=torch.optim.Adam(list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters()),lr=learning_rate)
    batches=max(1,int(np.ceil(sum(len(x) for x,_ in groups)/256)))
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        losses=[]
        for _ in range(batches):
            ai=rng.integers(len(ax),size=256)
            xparts,yparts=[ax[ai]],[ay[ai]]
            # 256 new/corrected examples, distributed equally across lessons.
            assignments=rng.integers(len(groups),size=256)
            for index,(x,y) in enumerate(groups):
                indices=rng.integers(len(x),size=int(np.sum(assignments==index)))
                xparts.append(x[indices]);yparts.append(y[indices])
            x=torch.tensor(np.concatenate(xparts),dtype=torch.float32)
            y=torch.tensor(np.concatenate(yparts),dtype=torch.float32)
            features=model.policy.extract_features(x)
            prediction=model.policy.action_net(model.policy.mlp_extractor.forward_actor(features))
            loss=((prediction-y)**2).mean()
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5)
            optimizer.step();losses.append(float(loss.detach()))
        if (epoch+1)%10==0: print(f'Epoch {epoch+1}/{epochs}: balanced action MSE {np.mean(losses):.8f}',flush=True)
    model.policy.set_training_mode(False)
    model.policy.optimizer.state.clear()
    return dict(epochs=epochs,optimizer_steps=epochs*batches,loss=float(np.mean(losses)))

class MixedEnv(DroneEnv):
    def __init__(self,levels,output):
        super().__init__(evaluation=True,record=True)
        self.levels,self.output=levels,output
        self.rng=np.random.default_rng(265000000+levels[-1])
        self.record_phase='ppo'
    def reset(self,*,seed=None,options=None):
        self.level=int(self.rng.choice(self.levels));self.chosen_seed=int(self.rng.integers(265000000,266000000))
        helper=DroneEnv(evaluation=True)
        reset(helper,self.level,self.chosen_seed)
        super().reset(seed=self.chosen_seed,options={'course':helper.course})
        self.stage,self.timeout=helper.stage,helper.timeout
        helper.close()
        return self._observation(),{}
    def step(self,action):
        result=super().step(action)
        if result[2]: record(self,self.output,self.record_phase,self.level,self.chosen_seed)
        return result

def run(output,episodes=16,epochs=60,rounds=4,ppo_steps=8192,cases=8,final_cases=16,
        source='runs/agile-curriculum-004/accepted-model.zip',lessons=None,old_levels=None,
        learning_rate=1e-4,continue_on_failure=False,seed_offset=0):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('Use a new empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(2042)
    source=Path(source)
    lessons=list(lessons if lessons is not None else [9,8,10,11,12])
    old=list(old_levels if old_levels is not None else [-1]+list(range(8)))
    def status(phase,**details):
        dump(output/'status.json',dict(phase=phase,**details))
    for name in ('train_maneuver.py','maneuver.py','agile_v4.py','agile_v2.py','agile_curriculum.py','train_agile_obstacles.py'):
        (output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    dump(output/'configuration.json',dict(profiles=PROFILES,collision_mode='non-solid-gates',
         source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         original_fingerprint=fingerprint(),episodes=episodes,epochs=epochs,rounds=rounds,
         ppo_steps=ppo_steps,cases=cases,final_cases=final_cases,lesson_order=lessons,
         old_levels=old,learning_rate=learning_rate,continue_on_failure=continue_on_failure,
         selection_seed=261000000+seed_offset,final_seed=270000000+seed_offset,seed_offset=seed_offset,anchor_batch_fraction=.5,
         observation_dimension=46,teacher='memoryless entrance staging; no avoidance'))
    model=PPO.load(source,env=DroneEnv(),device='cpu')
    frozen=PPO.load(source,device='cpu')
    model.save(output/'accepted-model.zip')
    learned=[];history=[];anchors=[];pools={}
    total_ppo=0;last_candidate=None
    status('retention-baseline')
    reference=assess(frozen,old,cases,261000000+seed_offset,output,'retention-baseline')
    dump(output/'retention-baseline.json',reference)
    # Preserve successful behavior across all earlier lessons, not just hover.
    for level in old:
        status('retention-collection',level=level)
        pools[level]=[]
        for case in range(4):
            seed=260000000+seed_offset+(level+1)*10000+case
            result,data,_=episode(frozen,level,seed,output,'retention-anchor',labels=True)
            if result['outcome']=='success': anchors.append(data)
            result,data,_=episode(model,level,seed+1000,output,'demonstration',teacher=True,labels=True)
            pools[level].append(data)
    if not anchors: raise RuntimeError('No successful retention anchors')
    np.savez_compressed(output/'anchors.npz',observations=np.concatenate([x for x,y in anchors]),actions=np.concatenate([y for x,y in anchors]))
    try:
        for level in lessons:
            pools[level]=[]
            accepted=False
            for round_index in range(rounds):
                status('collection',level=level,round=round_index,learned_levels=learned)
                # Restart each attempt from the accepted model, preventing drift
                # across rejected fits while retaining their corrective dataset.
                model=PPO.load(output/'accepted-model.zip',env=DroneEnv(),device='cpu')
                # Collect corrections from the rejected policy's actual state
                # distribution, while still fitting from the protected model.
                collector=PPO.load(last_candidate,device='cpu') if round_index and last_candidate else model
                outcomes=Counter()
                for case in range(episodes):
                    teacher=round_index==0 or case%2==0
                    result,data,_=episode(collector,level,262000000+seed_offset+level*10000+round_index*100+case,
                        output,'demonstration' if teacher else 'dagger',teacher=teacher,labels=True)
                    pools[level].append(data);outcomes[result['outcome']]+=1
                    print(f'Collect {PROFILES[level]["name"]} round {round_index}: {case+1}/{episodes} {result["outcome"]}',flush=True)
                # Refresh old-skill corrections on fresh training seeds each
                # round. Selection and final seeds are never used as labels.
                if continue_on_failure:
                    for prior in old+learned:
                        for offset,teacher in enumerate((True,False)):
                            _,data,_=episode(collector,prior,290000000+seed_offset+level*100000+round_index*1000+(prior+1)*2+offset,
                                output,'retention-demonstration' if teacher else 'retention-dagger',teacher=teacher,labels=True)
                            pools.setdefault(prior,[]).append(data)
                status('fitting',level=level,round=round_index,learned_levels=learned)
                fit_result=fit(model,anchors,pools,epochs,2042+level*10+round_index,learning_rate)
                candidate=output/f'level-{level}-round-{round_index}';candidate.mkdir()
                model.save(candidate/'model.zip');last_candidate=str(candidate/'model.zip')
                status('selection',level=level,round=round_index,learned_levels=learned)
                scores=assess(model,old+learned+[level],cases,261000000+seed_offset,output,f'selection-{level}-{round_index}')
                accepted=retains(scores,reference) and all(r['successes']==cases for r in scores if r['level'] in learned+[level])
                history.append(dict(level=level,round=round_index,collection=dict(outcomes),fit=fit_result,selection=scores,accepted=accepted))
                dump(output/'history.json',history)
                if accepted: break
            if not accepted:
                print('Promotion blocked; keeping the accepted checkpoint',flush=True)
                if continue_on_failure: continue
                break
            protected=copy.deepcopy(model.policy.state_dict())
            if ppo_steps:
                status('ppo',level=level,learned_levels=learned)
                vec=DummyVecEnv([lambda:MixedEnv(old+learned+[level],output)])
                model.set_env(vec);vec.envs[0].record_phase='critic_preparation'
                warm_critic(model,vec);vec.envs[0].record_phase='ppo'
                before=model.num_timesteps;model.learn(ppo_steps,reset_num_timesteps=False)
                total_ppo+=model.num_timesteps-before
                model.save(candidate/'ppo-candidate.zip')
                scores=assess(model,old+learned+[level],cases,261000000+seed_offset,output,f'ppo-selection-{level}')
                keep=retains(scores,reference) and all(r['successes']==cases for r in scores if r['level'] in learned+[level])
                if not keep: model.policy.load_state_dict(protected);model.policy.optimizer.state.clear()
                history[-1].update(ppo_accepted=keep,ppo_selection=scores)
                vec.close()
            learned.append(level);model.save(output/'accepted-model.zip');dump(output/'history.json',history)
        model=PPO.load(output/'accepted-model.zip',device='cpu')
        status('final-validation',learned_levels=learned)
        validation=assess(model,old+lessons,final_cases,270000000+seed_offset,output,'final-validation')
        candidate_validation=None
        if continue_on_failure and last_candidate:
            status('candidate-validation',learned_levels=learned)
            candidate_model=PPO.load(last_candidate,device='cpu')
            candidate_validation=assess(candidate_model,old+lessons,final_cases,270000000+seed_offset,output,'candidate-validation')
        dump(output/'report.json',dict(selected_model=str(output/'accepted-model.zip'),last_candidate=last_candidate,
             mastered_level=max(learned,default=7),learned_levels=learned,history=history,validation=validation,
             total_ppo_transitions=total_ppo,release_ready=len(learned)==len(lessons) and all(r['successes']/r['cases']>=.95 for r in validation),
             collision_mode='non-solid-gates',last_candidate_validation=candidate_validation))
        status('complete',learned_levels=learned,total_ppo_transitions=total_ppo)
    except BaseException as error:
        status('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),learned_levels=learned)
        raise
    finally:
        model.save(output/'interruption-model.zip')
        dump(output/'history.json',history)
        for level,data in pools.items():
            if data: np.savez_compressed(output/f'labels-{level}.npz',observations=np.concatenate([x for x,y in data]),actions=np.concatenate([y for x,y in data]))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    for name,default in [('episodes',16),('epochs',60),('rounds',4),('ppo-steps',8192),('cases',8),('final-cases',16)]:
        parser.add_argument('--'+name,type=int,default=default)
    run(**vars(parser.parse_args()))
