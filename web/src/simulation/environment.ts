/** Browser equivalent of DroneEnv. All integration uses fixed 120 Hz ticks. */
import vehicle from "../../../shared/vehicle.v1.json";
import rules from "../../../shared/course-rules.v1.json";
import fields from "../../../shared/observation.v1.json";
import type { CourseV1 } from "../contracts/types";
import { actionCommand, motorCommand, normalize, Quadrotor, rotation } from "../physics/dynamics";

type Gate = CourseV1["gates"][number];
export type Event = {time:number;type:string;label?:number};
const sub=(a:readonly number[],b:readonly number[])=>a.map((v,i)=>v-b[i]!);
const norm=(a:readonly number[])=>Math.hypot(...a);
const lerp=(a:readonly number[],b:readonly number[],t:number)=>a.map((v,i)=>v+t*(b[i]!-v));
const local=(p:readonly number[],g:Gate)=>{const d=sub(p,g.center_m),c=Math.cos(g.yaw_rad),s=Math.sin(g.yaw_rad);return [c*d[0]!+s*d[1]!,-s*d[0]!+c*d[1]!,d[2]!];};

export function slerp(a:readonly number[],b:readonly number[],t:number){
  a=normalize(a);b=normalize(b);let dot=a.reduce((s,v,i)=>s+v*b[i]!,0);
  if(dot<0){b=b.map(v=>-v);dot=-dot;} dot=Math.min(1,Math.max(0,dot));
  if(dot>.9995)return normalize(lerp(a,b,t));
  const theta=Math.acos(dot);return normalize(a.map((v,i)=>(Math.sin((1-t)*theta)*v+Math.sin(t*theta)*b[i]!)/Math.sin(theta)));
}

export function segmentBox(a:number[],b:number[],lo:number[],hi:number[]):number|null{
  let enter=0,leave=1;
  for(let i=0;i<3;i++){
    const d=b[i]!-a[i]!;
    if(Math.abs(d)<1e-12){if(a[i]!<lo[i]!-1e-9||a[i]!>hi[i]!+1e-9)return null;continue;}
    const x=(lo[i]!-a[i]!)/d,y=(hi[i]!-a[i]!)/d;
    enter=Math.max(enter,Math.min(x,y));leave=Math.min(leave,Math.max(x,y));
    if(enter>leave+1e-9)return null;
  }
  return enter>1||leave<0?null:Math.max(0,enter);
}

export function crossing(a:number[],b:number[],gate:Gate):{fraction:number;type:string}|null{
  a=local(a,gate);b=local(b,gate);
  if(a[0]!<0&&b[0]!>=0){const fraction=-a[0]!/(b[0]!-a[0]!),p=lerp(a,b,fraction);
    return {fraction,type:Math.abs(p[1]!)<=gate.width_m/2-.35&&Math.abs(p[2]!)<=gate.height_m/2-.35?"gate_pass":"gate_miss_forward"};}
  if(a[0]!>0&&b[0]!<=0)return {fraction:-a[0]!/(b[0]!-a[0]!),type:"gate_cross_backward"};
  return null;
}

export function collision(a:number[],b:number[],gates:Gate[]):{fraction:number;type:string}|null{
  let best:{fraction:number;type:string}|null=null;
  const add=(fraction:number,type:string)=>{if(best===null||fraction<best.fraction)best={fraction,type};};
  for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){
    const boundary=sign===-1?rules.workspace_min_m[axis]!+.3:rules.workspace_max_m[axis]!-.3;
    if(sign*(a[axis]!-boundary)>=0)add(0,axis===2&&sign===-1?"ground_collision":"workspace_exit");
    else if(sign*(b[axis]!-boundary)>=0)add((boundary-a[axis]!)/(b[axis]!-a[axis]!),axis===2&&sign===-1?"ground_collision":"workspace_exit");
  }
  const midpoint=lerp(a,b,.5),half=norm(sub(b,a))/2;
  for(const gate of gates){
    if(norm(sub(gate.center_m,midpoint))>half+2.5)continue;
    const x=local(a,gate),y=local(b,gate),w=gate.width_m/2,h=gate.height_m/2,d=gate.frame_depth_m/2,k=gate.frame_bar_m;
    const boxes=[ [[-d,-w-k,-h-k],[d,-w,h+k]], [[-d,w,-h-k],[d,w+k,h+k]], [[-d,-w,-h-k],[d,w,-h]], [[-d,-w,h],[d,w,h+k]] ];
    for(const [lo,hi] of boxes){const f=segmentBox(x,y,lo!.map(v=>v-.3),hi!.map(v=>v+.3));if(f!==null)add(f,"frame_collision");}
  }
  return best;
}

export class FlightEnvironment{
  state:number[];previous=[0,0,0,0];elapsed=0;tick=0;target=0;reward=0;outcome="running";
  events:Event[]=[];physics=new Quadrotor(vehicle);timeout:number;
  constructor(readonly course:CourseV1,profile:"original"|"agile"="original",readonly solidGates=true){
    const s=course.initial_state;
    this.state=[...s.position_m,...s.quaternion_wxyz,...s.velocity_world_mps,...s.omega_body_radps,...s.motor_thrust_n];
    this.timeout=({0:5,1:12,3:20,10:45} as Record<number,number>)[course.gates.length]!;
    if(profile==="agile")this.timeout=({0:5,1:60,3:180,10:600} as Record<number,number>)[course.gates.length]!;
    if(!this.timeout||this.state.length!==17||!this.state.every(Number.isFinite))throw new Error("Unsupported course or initial state");
  }
  agileObservation():Float32Array{
    const values=Array.from(this.observation()),gate=this.course.gates[this.target-1];
    if(!gate)return Float32Array.from([...values,0,0,0,0,0,0]);
    const r=rotation(this.state.slice(3,7));
    const body=(v:readonly number[])=>[0,1,2].map(i=>r.reduce((sum,row,j)=>sum+row[i]!*v[j]!,0));
    const previous=[...body(sub(gate.center_m,this.state.slice(0,3))).map(v=>v/40),...body([Math.cos(gate.yaw_rad),Math.sin(gate.yaw_rad),0])];
    return Float32Array.from([...values,...previous.map(v=>Math.max(-1,Math.min(1,v)))]);
  }
  obstacleObservation():Float32Array{
    const values=new Float32Array(109);values.set(this.agileObservation());
    const r=rotation(this.state.slice(3,7));
    const body=(v:readonly number[])=>[0,1,2].map(i=>r.reduce((sum,row,j)=>sum+row[i]!*v[j]!,0));
    const gates=this.course.gates.map((gate,index)=>({gate,index,distance:norm(sub(gate.center_m,this.state.slice(0,3)))}))
      .filter(({index})=>index!==this.target).sort((a,b)=>a.distance-b.distance||a.index-b.index).slice(0,9);
    gates.forEach(({gate},slot)=>values.set([...body(sub(gate.center_m,this.state.slice(0,3))).map(v=>v/200),
      ...body([Math.cos(gate.yaw_rad),Math.sin(gate.yaw_rad),0]),1].map(v=>Math.max(-1,Math.min(1,v))),46+slot*7));
    return values;
  }
  recoveryObservation():Float32Array{
    const r=rotation(this.state.slice(3,7));
    const body=(v:readonly number[])=>[0,1,2].map(i=>r.reduce((sum,row,j)=>sum+row[i]!*v[j]!,0));
    const extra:number[]=[];
    for(const index of [this.target,this.target+1,this.target-1]){
      const gate=this.course.gates[index];
      if(!gate){extra.push(0,0,0,0,0,0,0);continue;}
      const delta=body(sub(gate.center_m,this.state.slice(0,3))),distance=norm(delta);
      extra.push(...delta.map(v=>v/200),...delta.map(v=>v/Math.max(distance,1e-9)),distance/200);
    }
    extra.push((this.state[2]!-.3)/20);
    for(let axis=0;axis<3;axis++){
      const scale=axis===2?20:120;
      extra.push((this.state[axis]!-.3-rules.workspace_min_m[axis]!)/scale,
        (rules.workspace_max_m[axis]!-this.state[axis]!-.3)/scale);
    }
    return Float32Array.from([...this.agileObservation(),...extra.map(v=>Math.max(-1,Math.min(1,v)))]);
  }
  observation():Float32Array{
    const raw=new Array<number>(40).fill(0),s=this.state,R=rotation(s.slice(3,7));
    const body=(v:readonly number[])=>[0,1,2].map(i=>R.reduce((sum,row,j)=>sum+row[i]!*v[j]!,0));
    raw.splice(0,3,...body(s.slice(7,10)));raw.splice(3,3,...s.slice(10,13));raw.splice(6,3,...body([0,0,-1]));
    if(!this.course.gates.length){
      raw.splice(9,3,...body(sub(this.course.start_reference_m,s.slice(0,3))));
      const initial=rotation(this.course.initial_state.quaternion_wxyz),nose=[initial[0]![0]!,initial[1]![0]!,0],length=norm(nose);
      raw.splice(12,3,...body(nose.map(v=>v/length)));raw[30]=1;
    }
    for(let offset=0;offset<2;offset++){
      const gate=this.course.gates[this.target+offset];if(!gate)continue;
      raw.splice(9+6*offset,3,...body(sub(gate.center_m,s.slice(0,3))));
      raw.splice(12+6*offset,3,...body([Math.cos(gate.yaw_rad),Math.sin(gate.yaw_rad),0]));raw[30+offset]=1;
    }
    raw[21]=s[2]!-.3;raw.splice(22,4,...this.previous);raw.splice(26,4,...s.slice(13));
    for(let i=0;i<3;i++){raw[32+2*i]=s[i]!-.3-rules.workspace_min_m[i]!;raw[33+2*i]=rules.workspace_max_m[i]!-s[i]!-.3;}
    raw[38]=Math.max(0,this.course.gates.length-this.target);raw[39]=Math.max(0,1-this.elapsed/this.timeout);
    if(!raw.every(Number.isFinite))throw new Error("Nonfinite observation");
    return Float32Array.from(raw.map((v,i)=>Math.min(fields.fields[i]!.maximum,Math.max(fields.fields[i]!.minimum,v/fields.fields[i]!.scale))));
  }
  step(action:readonly number[]):number[][]{
    if(this.outcome!=="running")throw new Error("Flight has ended");
    const cmd=actionCommand(action,vehicle),gates=this.course.gates,rows:number[][]=[];
    let reward=-.02*cmd.action.reduce((sum,v,i)=>sum+(v-this.previous[i]!)**2,0),reason:string|null=null;
    for(let i=0;i<2;i++){
      const before=this.state.slice(),h=this.physics.dt,motor=motorCommand(before,cmd.collective,cmd.rates,this.physics);
      const after=this.physics.step(before,motor.motors),a=before.slice(0,3),b=after.slice(0,3);
      const hit=collision(a,b,this.solidGates?gates:[]),gate=gates[this.target],cross=gate?crossing(a,b,gate):null;
      let fraction=hit?.fraction??1;reason=hit?.type??null;
      if(cross&&cross.type!=="gate_pass"&&cross.fraction<fraction)this.events.push({time:this.elapsed+h*cross.fraction,type:cross.type,label:this.target+1});
      if(cross?.type==="gate_pass"&&(!hit||cross.fraction<hit.fraction-1e-9)){
        const point=lerp(a,b,cross.fraction),center=gate!.center_m;
        reward+=.5*(norm(sub(a,center))-norm(sub(point,center)))+10;
        this.target++;this.events.push({time:this.elapsed+h*cross.fraction,type:"gate_pass",label:this.target});
        if(this.target===gates.length){fraction=cross.fraction;reason="success";reward+=50;}
        else{const next=gates[this.target]!.center_m;reward+=.5*(norm(sub(point,next))-norm(sub(lerp(a,b,fraction),next)));}
      }else if(gate)reward+=.5*(norm(sub(a,gate.center_m))-norm(sub(lerp(a,b,fraction),gate.center_m)));
      this.state=lerp(before,after,fraction);this.state.splice(3,4,...slerp(before.slice(3,7),after.slice(3,7),fraction));
      const duration=h*fraction;this.elapsed+=duration;this.tick++;if(fraction===1)this.elapsed=this.tick*h;
      if(reason===null&&(norm(this.state.slice(7,10))>vehicle.state_limit_speed_mps||norm(this.state.slice(10,13))>vehicle.state_limit_rate_radps))reason="state_limit";
      if(!gates.length){const distance=norm(sub(this.state.slice(0,3),this.course.start_reference_m)),tilt=rotation(this.state.slice(3,7))[2]![2]!;
        reward+=duration*(1-.5*distance**2-.1*norm(this.state.slice(7,10))**2-.1*(1-tilt)-.02*norm(this.state.slice(10,13))**2);
        if(reason===null&&(distance>1||tilt<Math.cos(Math.PI/6)))reason="hover_departure";
      }else reward-=.2*duration;
      rows.push([this.elapsed,...this.state]);
      if(reason===null&&this.tick>=Math.round(this.timeout/h)){reason=gates.length?"timeout":"success";if(!gates.length)reward+=5;}
      if(reason!==null)break;
    }
    if(reason!==null&&reason!=="success")reward-=gates.length?50:5;
    this.previous=[...cmd.action];this.reward+=reward;
    if(reason!==null){this.outcome=reason;this.events.push({time:this.elapsed,type:reason});}
    return rows;
  }
}
