import numpy as np
import pytest
from notebooks.maneuver import DroneEnv,reset_course,expert
from training.physics.collision import CollisionWorld,gate_basis,gate_crossing
from notebooks.train_maneuver import retains

def test_only_gate_collision_is_disabled():
    env=DroneEnv(evaluation=True,record=True);reset_course(env,0,280000001)
    gate=env.course['gates'][0];c=np.array(gate['center_m']);b=gate_basis(gate)
    a=c+b@[-2,1.3,0];z=c+b@[2,1.3,0]
    assert CollisionWorld(env.course['gates'],env.vehicle,env.rules).first_collision(a,z)[1]=='frame_collision'
    assert env.world.first_collision(a,z) is None
    assert gate_crossing(a,z,gate,.35)[1]=='gate_miss_forward'
    assert gate_crossing(c-b[:,0],c+b[:,0],gate,.35)[1]=='gate_pass'
    assert gate_crossing(c+b[:,0],c-b[:,0],gate,.35)[1]=='gate_cross_backward'
    assert env.world.first_collision(c,np.array([c[0],c[1],-1]))[1]=='ground_collision'

@pytest.mark.parametrize('level',[8,9,10,11,12])
def test_teacher_completes_seeded_vertical_lessons(level):
    env=DroneEnv(evaluation=True,record=True);reset_course(env,level,280000000+level)
    heights=[env.course['start_reference_m'][2]]+[g['center_m'][2] for g in env.course['gates']]
    if level==8: assert np.all(np.diff(heights)>0)
    if level in (9,11): assert np.all(np.diff(heights)<0)
    while True:
        _,_,done,_,info=env.step(expert(env))
        if done: break
    assert info['outcome']=='success'
    assert info['collision_mode']=='non-solid-gates'
    assert info['gates_passed']==len(env.course['gates'])

def test_retention_preserves_individual_successes():
    reference=[dict(level=0,results=[dict(outcome='success'),dict(outcome='timeout')])]
    swapped=[dict(level=0,results=[dict(outcome='timeout'),dict(outcome='success')])]
    assert not retains(swapped,reference)
    assert retains(reference,reference)
