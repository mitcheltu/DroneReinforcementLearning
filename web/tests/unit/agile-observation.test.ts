import {describe,it,expect} from "vitest";
import fixtures from "../fixtures/agile-observation.json";
import {FlightEnvironment} from "../../src/simulation/environment";
import type {CourseV1} from "../../src/contracts/types";
import {readReplay} from "../../src/replay/npz";
import {readFileSync} from "node:fs";
import {inflateRawSync} from "node:zlib";

describe("Agile observation contract",()=>{
  it("reads an extended-duration experimental trace without weakening array checks",async()=>{
    const replay=await readReplay(new Uint8Array(readFileSync(new URL("../fixtures/agile-long.npz",import.meta.url))),async bytes=>new Uint8Array(inflateRawSync(bytes)));
    expect(replay.duration).toBe(60);expect(replay.course.mode).toBe("experimental");
    expect(replay.count).toBe(2);expect(replay.outcome).toBe("timeout");
  });
  it("matches Python including previous-gate transforms and extended timing",()=>{
    for(const row of fixtures){
      const env=new FlightEnvironment(row.course as CourseV1,"agile");
      env.state=[...row.state];env.previous=[...row.previous];env.target=row.target;env.elapsed=row.elapsed;
      expect(env.timeout).toBe(row.timeout);
      const actual=env.agileObservation();expect(actual.length).toBe(46);
      actual.forEach((v,i)=>expect(v).toBeCloseTo(row.observation[i]!,6));
      expect(Array.from(actual.slice(0,40))).toEqual(Array.from(env.observation()));
    }
  });
  it("preserves the original default timeout and observation size",()=>{
    const env=new FlightEnvironment(fixtures[0]!.course as CourseV1);
    expect(env.timeout).toBe(12);expect(env.observation().length).toBe(40);
    expect(Array.from(env.agileObservation().slice(40))).toEqual([0,0,0,0,0,0]);
  });
});
