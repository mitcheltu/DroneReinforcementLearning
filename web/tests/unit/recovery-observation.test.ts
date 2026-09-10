import {describe,it,expect} from 'vitest';
import fixtures from '../fixtures/recovery-observation.json';
import {FlightEnvironment} from '../../src/simulation/environment';
import type {CourseV1} from '../../src/contracts/types';
describe('Run 008 observation contract',()=>{
  it('matches Python for distant gates, recovery attitudes, previous gates and hover',()=>{
    for(const fixture of fixtures){
      const env=new FlightEnvironment(fixture.course as CourseV1,'agile',false);
      env.state=[...fixture.state];env.target=fixture.target;env.previous=[...fixture.previous];env.elapsed=fixture.elapsed;
      const actual=env.recoveryObservation();expect(actual.length).toBe(74);
      actual.forEach((v,i)=>expect(v).toBeCloseTo(fixture.observation[i]!,6));
    }
  });
});
