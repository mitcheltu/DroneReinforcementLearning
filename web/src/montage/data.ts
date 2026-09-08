import type {CourseV1} from "../contracts/types";
import type {Replay} from "../replay/npz";
import {multiply,rotation} from "../physics/dynamics";

export interface Attempt {id:string;source:string;run:string;stage:number;outcome:string;duration:number;course:CourseV1;samples:number[][];provenance:string;}
export interface MontageData {version:number;traces:Attempt[];episode_logs:{run:string;episodes_by_stage:Record<string,number>}[];note:string;}
export function fromReplay(replay:Replay,id:string):Attempt{
  const stride=Math.max(1,Math.ceil(replay.count/600)),samples:number[][]=[];
  for(let i=0;i<replay.count;i+=stride)samples.push(Array.from(replay.states.slice(i*18,i*18+8)));
  if((replay.count-1)%stride!==0)samples.push(Array.from(replay.states.slice(-18,-10)));
  return {id,source:"imported",run:"Local imports",stage:replay.stage,outcome:replay.outcome,duration:replay.duration,course:replay.course,samples,provenance:"user-imported trace"};
}
export function alignAttempt(attempt:Attempt,aligned:boolean):Attempt{
  if(!aligned)return attempt;
  const r=rotation(attempt.course.initial_state.quaternion_wxyz),yaw=Math.atan2(r[1]![0]!,r[0]![0]!);
  const c=Math.cos(yaw),s=Math.sin(yaw),origin=attempt.course.start_reference_m;
  const point=(p:readonly number[]):[number,number,number]=>{const x=p[0]!-origin[0],y=p[1]!-origin[1];return [c*x+s*y,-s*x+c*y,p[2]!-origin[2]];};
  const q=[Math.cos(yaw/2),0,0,-Math.sin(yaw/2)];
  return {...attempt,samples:attempt.samples.map(row=>[row[0]!,...point(row.slice(1,4)),...multiply(q,row.slice(4,8))]),
    course:{...attempt.course,start_reference_m:[0,0,0],gates:attempt.course.gates.map(g=>({...g,center_m:point(g.center_m),yaw_rad:g.yaw_rad-yaw}))}};
}
export function poseAt(samples:number[][],time:number):{a:number[];b:number[];fraction:number}{
  let lo=0,hi=samples.length-1;
  while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(samples[mid]![0]!<=time)lo=mid;else hi=mid-1;}
  const a=samples[lo]!,b=samples[Math.min(lo+1,samples.length-1)]!;
  return {a,b,fraction:a===b?0:Math.max(0,Math.min(1,(time-a[0]!)/(b[0]!-a[0]!)))};
}
