import {describe,it,expect} from "vitest";
import fixtures from "../fixtures/flight-parity.json";
import type {CourseV1} from "../../src/contracts/types";
import {courseErrors,editGate} from "../../src/editor/course";
import {FlightEnvironment} from "../../src/simulation/environment";
import {validateArtifact} from "../../src/contracts/loader";
import {alignAttempt,fromReplay,poseAt,type Attempt} from "../../src/montage/data";

const original=fixtures[1]!.course as CourseV1;
describe("Custom gate layout",()=>{
  it("extends only the browser editor contract, retaining original curriculum validation",()=>{
    const changed=editGate(original,0,[0,0,5],0);
    expect(courseErrors(changed)).toEqual([]);
    expect(()=>validateArtifact("course",changed)).toThrow();
    expect(courseErrors({...changed,mode:"curriculum"})).not.toEqual([]);
    expect(courseErrors({...changed,mode:"standard"})).not.toEqual([]);
  });
  it("changes the actual flight target without changing the preset or start",()=>{
    const position=[...original.gates[0]!.center_m] as [number,number,number];position[2]+=.4;
    const changed=editGate(original,0,position,Math.PI/2);
    expect(changed.mode).toBe("experimental");expect(changed.generator).toBeNull();
    expect(changed.initial_state).toEqual(original.initial_state);
    expect(original.gates[0]!.center_m[2]).not.toBe(changed.gates[0]!.center_m[2]);
    expect(courseErrors(changed)).toEqual([]);
    expect(Array.from(new FlightEnvironment(changed).observation())).not.toEqual(Array.from(new FlightEnvironment(original).observation()));
  });
  it("rejects nonfinite input and complete frames outside the workspace",()=>{
    expect(()=>editGate(original,0,[NaN,0,5],0)).toThrow("finite");
    expect(courseErrors(editGate(original,0,[60,0,5],Math.PI/2))).not.toEqual([]);
    expect(courseErrors(editGate(original,0,[0,0,.5],0))).not.toEqual([]);
    expect(editGate(original,0,[0,0,5],3*Math.PI).gates[0]!.yaw_rad).toBeCloseTo(-Math.PI);
  });
});
describe("Montage alignment and sampling",()=>{
  it("aligns both flight positions and gate positions without changing source data",()=>{
    const course=structuredClone(original);course.start_reference_m=[10,20,5];
    course.initial_state.quaternion_wxyz=[Math.SQRT1_2,0,0,Math.SQRT1_2];course.gates[0]!.center_m=[10,27,5];course.gates[0]!.yaw_rad=Math.PI/2;
    const attempt:Attempt={id:"a",source:"imported",run:"test",stage:1,outcome:"success",duration:1,course,provenance:"test",samples:[[0,10,20,5,Math.SQRT1_2,0,0,Math.SQRT1_2],[1,10,27,5,Math.SQRT1_2,0,0,Math.SQRT1_2]]};
    const aligned=alignAttempt(attempt,true);
    expect(aligned.samples[1]![1]).toBeCloseTo(7);expect(aligned.samples[1]![2]).toBeCloseTo(0);
    expect(aligned.course.gates[0]!.center_m[0]).toBeCloseTo(7);
    expect(aligned.course.gates[0]!.yaw_rad).toBeCloseTo(0);
    expect(attempt.samples[1]![2]).toBe(27);
    expect(poseAt(aligned.samples,.5).fraction).toBeCloseTo(.5);
    expect(poseAt(aligned.samples,50).a).toEqual(aligned.samples[1]);
  });
  it("preserves terminal poses while decimating long traces",()=>{
    const states=Float64Array.from({length:18018},(_,i)=>i);
    const attempt=fromReplay({course:original,states,count:1001,duration:18000,outcome:"success",stage:1,reward:0,events:[]},"long");
    expect(attempt.samples.length).toBeLessThanOrEqual(601);
    expect(attempt.samples.at(-1)).toEqual(Array.from(states.slice(-18,-10)));
  });
});
