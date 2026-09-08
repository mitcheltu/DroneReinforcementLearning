import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { describe,expect,it } from "vitest";
import vehicle from "../../../shared/vehicle.v1.json";
import { Quadrotor,actionCommand,motorCommand,normalize,rotation } from "../../src/physics/dynamics";
import fixtures from "../fixtures/physics-parity.json";

describe("Python / TypeScript dynamics parity",()=>{
  it("uses the exact same vehicle configuration",()=>{
    const bytes=readFileSync(new URL("../../../shared/vehicle.v1.json",import.meta.url));
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(fixtures.vehicle_sha256);
  });
  for(const scenario of fixtures.scenarios) it(`matches 240 physics ticks: ${scenario.name}`,()=>{
    const physics=new Quadrotor(vehicle);let state=scenario.initial;
    for(const expected of scenario.transitions){
      const command=actionCommand(expected.action,vehicle);
      const {motors,scale}=motorCommand(state,command.collective,command.rates,physics);
      expect(Math.abs(scale-expected.scale)).toBeLessThan(1e-9);
      motors.forEach((m,i)=>expect(Math.abs(m-expected.motors[i]!)).toBeLessThan(1e-9));
      state=physics.step(state,motors);
      state.forEach((v,i)=>expect(Math.abs(v-expected.state[i]!)).toBeLessThan(1e-9));
    }
  });
  it("rejects nonfinite actions and invalid integration requests",()=>{
    expect(()=>actionCommand([NaN,0,0,0],vehicle)).toThrow("finite");
    expect(()=>normalize([0,0,0,0])).toThrow("quaternion");
    expect(()=>new Quadrotor(vehicle).step(new Array(17).fill(0),[0,0,0,0])).toThrow("quaternion");
    expect(rotation([1,0,0,0])).toEqual([[1,0,0],[0,1,0],[0,0,1]]);
  });
});
