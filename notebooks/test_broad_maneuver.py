import numpy as np
import pytest
from notebooks.broad_maneuver import DroneEnv, PROFILES, LESSONS, reset_course, expert
from training.physics.quaternion import rotation


@pytest.mark.parametrize('level',LESSONS)
def test_generator_is_reproducible_and_inside_workspace(level):
    env=DroneEnv(evaluation=True)
    for seed in range(310000000,310000024):
        obs,_=reset_course(env,level,seed)
        course=env.course
        again,_=reset_course(env,level,seed)
        assert course==env.course
        np.testing.assert_array_equal(obs,again)
        assert obs.shape==(46,) and np.isfinite(obs).all()
        assert len(course['gates'])==PROFILES[level]['count']
        assert [g['label'] for g in course['gates']]==list(range(1,len(course['gates'])+1))
        points=np.array([course['start_reference_m']]+[g['center_m'] for g in course['gates']])
        assert np.max(np.abs(points[:,:2]))<51
        assert np.min(points[:,2])>2 and np.max(points[:,2])<17
        if level in (13,14,15):
            body=rotation(course['initial_state']['quaternion_wxyz']).T
            relative=np.diff(points,axis=0)@body.T
            lo,hi=PROFILES[level]['span']
            assert np.all(np.abs(relative[:,1])>=lo-1e-6)
            assert np.all(np.abs(relative[:,1])<=hi+1e-6)
            assert np.all(relative[1:,1]*relative[:-1,1]<0)
    env.close()


@pytest.mark.parametrize('level',LESSONS)
@pytest.mark.parametrize('case',[0,1])
def test_teacher_can_complete_stress_courses(level,case):
    env=DroneEnv(evaluation=True)
    reset_course(env,level,320000000+level*100+case)
    while True:
        _,_,done,_,info=env.step(expert(env))
        if done: break
    env.close()
    assert info['outcome']=='success', (level,case,info)
