import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from notebooks.recovery_v8 import RecoveryEnv, RECOVERIES
from notebooks.maneuver import expert
from notebooks.train_recovery_v8 import transfer, qualifies, retains, GROUPS
from notebooks.train_recovery_v8 import warm_value
from stable_baselines3.common.vec_env import DummyVecEnv


def test_far_geometry_preserves_distance_and_direction():
    env=RecoveryEnv();env.reset(seed=701000001,options={'level':13})
    env.state[:3]=[-50,-50,3];env.state[3:7]=[1,0,0,0]
    env.course['gates'][0]['center_m']=[45,10,16]
    a=env._observation()
    env.course['gates'][0]['center_m']=[25,10,16]
    b=env._observation()
    assert a[9]==b[9]==1
    assert a[46]!=b[46] and a[52]!=b[52]
    np.testing.assert_allclose(a[49:52],np.array([95,60,13])/np.linalg.norm([95,60,13]),atol=1e-6)
    assert a.shape==(74,) and np.isfinite(a).all()


def test_transfer_is_exact_and_uses_new_features():
    torch.set_num_threads(1)
    source='runs/agile-curriculum-006/level-9-round-3/model.zip'
    env=RecoveryEnv();obs,_=env.reset(seed=701000002,options={'level':18,'recovery':'combined','difficulty':2})
    model=transfer(source,env,801);old=PPO.load(source,device='cpu')
    np.testing.assert_allclose(model.predict(obs,deterministic=True)[0],old.predict(obs[:46],deterministic=True)[0],atol=1e-6)
    features=model.policy.extract_features(torch.tensor(obs[None]))
    np.testing.assert_allclose(features.detach().numpy()[0,39:],obs)
    assert features.shape==(1,113)


@pytest.mark.parametrize('kind',RECOVERIES)
@pytest.mark.parametrize('level',[18,20])
def test_recovery_teacher_can_reacquire_and_finish(kind,level):
    env=RecoveryEnv();obs,_=env.reset(seed=702000000+level*100+RECOVERIES.index(kind),options=dict(level=level,recovery=kind,difficulty=2))
    first=obs.copy();course=env.course
    obs,_=env.reset(seed=702000000+level*100+RECOVERIES.index(kind),options=dict(level=level,recovery=kind,difficulty=2))
    np.testing.assert_array_equal(first,obs);assert course==env.course
    while True:
        _,reward,done,_,info=env.step(expert(env))
        assert np.isfinite(reward)
        if done:break
    assert info['outcome']=='success', info
    assert info['reacquired_s'] is not None
    assert info['remaining_passed']==len(env.course['gates'])-env.start_target


def test_release_gate_cannot_pass_missing_or_failed_groups():
    rows=[dict(name=n,cases=32,successes=32) for n,_,_ in GROUPS]
    assert qualifies(rows)
    assert not qualifies(rows[:-1])
    rows[-1]['successes']=30
    assert not qualifies(rows)
    baseline=[dict(name='x',results=[dict(outcome='success')])]
    assert not retains([dict(name='x',results=[dict(outcome='timeout')])],baseline)


def test_short_ppo_changes_actor_without_teacher_labels():
    torch.set_num_threads(1)
    env=RecoveryEnv(training_mix=True)
    model=PPO('MlpPolicy',env,n_steps=32,batch_size=32,n_epochs=1,device='cpu',seed=805)
    before=model.policy.action_net.weight.detach().clone()
    model.learn(64)
    assert model.num_timesteps==64
    assert not torch.equal(before,model.policy.action_net.weight)


def test_critic_initialization_preserves_actor():
    torch.set_num_threads(1)
    vec=DummyVecEnv([lambda:RecoveryEnv(training_mix=True)])
    model=transfer('runs/agile-curriculum-006/level-9-round-3/model.zip',vec,806)
    actor={k:v.clone() for k,v in model.policy.state_dict().items() if 'policy_net' in k or 'action_net' in k}
    assert warm_value(model,vec,steps=8)==8
    assert all(torch.equal(value,model.policy.state_dict()[key]) for key,value in actor.items())
    vec.close()
