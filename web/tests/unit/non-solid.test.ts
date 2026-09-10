import {it,expect} from "vitest";
import fixtures from "../fixtures/agile-observation.json";
import {FlightEnvironment,collision,crossing} from "../../src/simulation/environment";
import type {CourseV1} from "../../src/contracts/types";

it("removes only frame collisions while preserving directional aperture checks",()=>{
  const course=fixtures[0]!.course as CourseV1,g=course.gates[0]!;
  const point=(x:number,y:number)=>[g.center_m[0]+x*Math.cos(g.yaw_rad)-y*Math.sin(g.yaw_rad),g.center_m[1]+x*Math.sin(g.yaw_rad)+y*Math.cos(g.yaw_rad),g.center_m[2]];
  const a=point(-2,1.3),b=point(2,1.3);
  expect(collision(a,b,course.gates)?.type).toBe("frame_collision");
  expect(collision(a,b,[])).toBeNull();
  expect(crossing(a,b,g)?.type).toBe("gate_miss_forward");
  expect(crossing(point(-2,0),point(2,0),g)?.type).toBe("gate_pass");
  expect(crossing(point(2,0),point(-2,0),g)?.type).toBe("gate_cross_backward");
  expect(collision(a,[a[0]!,a[1]!,-1],[])?.type).toBe("ground_collision");
  expect(new FlightEnvironment(course,"agile",false).solidGates).toBe(false);
  const solid=new FlightEnvironment(course,"agile",true),ghost=new FlightEnvironment(course,"agile",false);
  for(const env of [solid,ghost]){
    env.state.splice(0,3,...point(-.01,1.3));
    env.state.splice(7,3,3*Math.cos(g.yaw_rad),3*Math.sin(g.yaw_rad),0);
    env.step([-.019,0,0,0]);
  }
  expect(solid.outcome).toBe("frame_collision");
  expect(ghost.outcome).toBe("running");
  expect(ghost.target).toBe(0);
  expect(ghost.events.some(event=>event.type==="gate_miss_forward")).toBe(true);
});
