import {it,expect} from "vitest";
import fixtures from "../fixtures/obstacle-observation.json";
import {FlightEnvironment} from "../../src/simulation/environment";
import type {CourseV1} from "../../src/contracts/types";

it("matches all 109 Python obstacle inputs including ordering and masks",()=>{
  for(const row of fixtures){
    const env=new FlightEnvironment(row.course as CourseV1,"agile");
    env.state=[...row.state];env.target=row.target;env.previous=[...row.previous];env.elapsed=row.elapsed;
    const actual=env.obstacleObservation();expect(actual.length).toBe(109);
    actual.forEach((value,index)=>expect(value).toBeCloseTo(row.observation[index]!,6));
  }
});
