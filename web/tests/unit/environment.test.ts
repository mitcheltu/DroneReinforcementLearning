import {describe,it,expect} from "vitest";
import fixtures from "../fixtures/flight-parity.json";
import type {CourseV1} from "../../src/contracts/types";
import {FlightEnvironment,crossing,collision} from "../../src/simulation/environment";

describe("Complete Python/browser flight parity",()=>{
  for(const fixture of fixtures)it(`stage ${fixture.stage}, failure=${fixture.failure}`,()=>{
    const env=new FlightEnvironment(fixture.course as CourseV1);
    for(const step of fixture.steps){
      const observation=env.observation();
      step.observation.forEach((v,i)=>expect(Math.abs(observation[i]!-v)).toBeLessThan(2e-6));
      const previous=env.reward;env.step(step.action);
      step.state.forEach((v,i)=>expect(Math.abs(env.state[i]!-v)).toBeLessThan(1e-7));
      expect(Math.abs(env.reward-previous-step.reward)).toBeLessThan(1e-7);
      expect(env.target).toBe(step.target);expect(Math.abs(env.elapsed-step.time)).toBeLessThan(1e-8);
    }
    expect(env.outcome).toBe(fixture.outcome);
    expect(Math.abs(env.reward-fixture.reward)).toBeLessThan(1e-7);
  });
  it("distinguishes forward, backward and frame hits",()=>{
    const gate={...(fixtures[1]!.course.gates[0] as CourseV1["gates"][number]),center_m:[0,0,5] as [number,number,number],yaw_rad:0};
    expect(crossing([-1,0,5],[1,0,5],gate)?.type).toBe("gate_pass");
    expect(crossing([1,0,5],[-1,0,5],gate)?.type).toBe("gate_cross_backward");
    expect(collision([-1,1.3,5],[1,1.3,5],[gate])?.type).toBe("frame_collision");
    expect(collision([0,0,1],[0,0,0],[])?.type).toBe("ground_collision");
  });
});
