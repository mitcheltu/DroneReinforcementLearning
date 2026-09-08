/** Numeric body-rate quadrotor core. State layout matches the Python 17-field state. */
import type { VehicleConfig, Vector3, Vector4 } from "../contracts/types";

export const clamp = (value:number, low:number, high:number) => Math.max(low,Math.min(value,high));
export function normalize(q: readonly number[]): Vector4 {
  const norm=Math.hypot(...q);
  if(q.length!==4 || !Number.isFinite(norm) || norm<1e-12) throw new Error("Invalid quaternion");
  return q.map(x=>x/norm) as Vector4;
}
export function multiply(q:readonly number[],r:readonly number[]):Vector4 {
  const [w,x,y,z]=q as Vector4,[a,b,c,d]=r as Vector4;
  return [w*a-x*b-y*c-z*d,w*b+x*a+y*d-z*c,w*c-x*d+y*a+z*b,w*d+x*c-y*b+z*a];
}
export function cross(a:readonly number[],b:readonly number[]):Vector3 {
  return [a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,a[0]!*b[1]!-a[1]!*b[0]!];
}
export function rotation(q:readonly number[]):number[][] {
  const [w,x,y,z]=normalize(q);
  return [[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
    [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
    [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]];
}

export class Quadrotor {
  readonly dt:number;
  readonly b:number;
  constructor(readonly config:VehicleConfig){this.dt=1/config.physics_hz;this.b=config.arm_length_m/Math.sqrt(2);}

  torque(f:readonly number[]):Vector3 {
    const [a,b,c,d]=f as Vector4,k=this.config.reaction_torque_per_thrust_m;
    return [this.b*(a+b-c-d),this.b*(-a+b+c-d),k*(a-b+c-d)];
  }

  derivative(state:readonly number[],commands:readonly number[]):number[] {
    const q=state.slice(3,7),v=state.slice(7,10),omega=state.slice(10,13),motors=state.slice(13);
    const result=new Array<number>(17).fill(0),matrix=rotation(q),config=this.config;
    const torque=this.torque(motors),gyro=cross(omega,omega.map((x,i)=>x*config.inertia_kgm2[i]!));
    const dq=multiply(q,[0,...omega]);
    for(let i=0;i<3;i++) {
      result[i]=v[i]!;
      result[7+i]=matrix[i]![2]!*motors.reduce((a,b)=>a+b,0)/config.mass_kg-config.linear_drag_kgps*v[i]!/config.mass_kg;
      result[10+i]=(torque[i]!-gyro[i]!)/config.inertia_kgm2[i]!;
    }
    result[9]=result[9]!-config.gravity_mps2;
    for(let i=0;i<4;i++){result[3+i]=.5*dq[i]!;result[13+i]=(commands[i]!-motors[i]!)/config.motor_time_constant_s;}
    return result;
  }

  step(state:readonly number[],commands:readonly number[],h=this.dt):number[] {
    if(state.length!==17 || commands.length!==4 || ![...state,...commands,h].every(Number.isFinite) || h<=0 || h>this.dt+1e-12) throw new Error("Invalid physics input");
    if(commands.some(x=>x<0 || x>this.config.rotor_max_thrust_n)) throw new Error("Motor command out of bounds");
    const add=(delta:number[],scale:number)=>state.map((x,i)=>x+scale*delta[i]!);
    const a=this.derivative(state,commands),b=this.derivative(add(a,h/2),commands);
    const c=this.derivative(add(b,h/2),commands),d=this.derivative(add(c,h),commands);
    const next=state.map((x,i)=>x+h*(a[i]!+2*b[i]!+2*c[i]!+d[i]!)/6);
    if(!next.every(Number.isFinite)) throw new Error("Nonfinite RK4 state");
    next.splice(3,4,...normalize(next.slice(3,7)));
    const tolerance=this.config.motor_roundoff_tolerance_n,max=this.config.rotor_max_thrust_n;
    for(let i=13;i<17;i++){if(next[i]!< -tolerance || next[i]!>max+tolerance) throw new Error("Motor state exceeded physical bounds");next[i]=clamp(next[i]!,0,max);}
    return next;
  }
}

export function actionCommand(action:readonly number[],config:VehicleConfig):{action:Vector4;collective:number;rates:Vector3} {
  if(action.length!==4 || !action.every(Number.isFinite)) throw new Error("Action must contain four finite numbers");
  const u=action.map(x=>clamp(x,-1,1)) as Vector4;
  return {action:u,collective:(u[0]+1)*config.action.max_collective_n/2,
    rates:u.slice(1).map((x,i)=>x*config.action.max_body_rates_radps[i]!) as Vector3};
}

export function motorCommand(state:readonly number[],collective:number,rates:readonly number[],physics:Quadrotor):{motors:Vector4;scale:number} {
  if(state.length!==17 || rates.length!==3 || ![...state,...rates,collective].every(Number.isFinite) || collective<0 || collective>4*physics.config.rotor_max_thrust_n) throw new Error("Invalid controller input");
  const config=physics.config,omega=state.slice(10,13);
  const gyro=cross(omega,omega.map((x,i)=>x*config.inertia_kgm2[i]!));
  const [tx,ty,tz]=omega.map((x,i)=>config.rate_gains[i]!*(rates[i]!-x)+gyro[i]!) as Vector3;
  const b=4*physics.b,k=4*config.reaction_torque_per_thrust_m,center=collective/4;
  const delta=[tx/b-ty/b+tz/k,tx/b+ty/b-tz/k,-tx/b+ty/b+tz/k,-tx/b-ty/b-tz/k];
  let scale=1;
  for(const d of delta){if(d>0)scale=Math.min(scale,(config.rotor_max_thrust_n-center)/d);else if(d<0)scale=Math.min(scale,center/-d);}
  scale=clamp(scale,0,1);
  return {motors:delta.map(d=>clamp(center+scale*d,0,config.rotor_max_thrust_n)) as Vector4,scale};
}
