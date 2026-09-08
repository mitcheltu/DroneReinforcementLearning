import type { CourseV1, Vector3 } from "../contracts/types";
import { validateArtifact } from "../contracts/loader";

export const wrapYaw=(value:number)=>((value+Math.PI)%(2*Math.PI)+2*Math.PI)%(2*Math.PI)-Math.PI;
export function editGate(course:CourseV1,index:number,position:Vector3,yaw:number):CourseV1{
  if(!course.gates[index]||![...position,yaw].every(Number.isFinite))throw new Error("Enter finite gate coordinates and heading");
  return {...course,mode:"experimental",generator:null,gates:course.gates.map((gate,i)=>i===index?{...gate,center_m:[...position],yaw_rad:wrapYaw(yaw)}:gate)};
}
export function courseErrors(course:CourseV1):string[]{
  const errors:string[]=[];
  try{validateArtifact("editable-course",course);}catch(e){errors.push(e instanceof Error?e.message:"Invalid course");return errors;}
  if(![0,1,3,10].includes(course.gates.length))errors.push("Choose a supported course with 0, 1, 3 or 10 gates");
  for(const g of course.gates){
    const c=Math.abs(Math.cos(g.yaw_rad)),s=Math.abs(Math.sin(g.yaw_rad));
    const extent=[c*g.frame_depth_m/2+s*(g.width_m/2+g.frame_bar_m),s*g.frame_depth_m/2+c*(g.width_m/2+g.frame_bar_m),g.height_m/2+g.frame_bar_m];
    if(g.center_m.some((v,i)=>v-extent[i]!<[-60,-60,0][i]!||v+extent[i]!>[60,60,20][i]!))errors.push(`Gate ${g.label}: keep its entire frame inside X/Y ±60 m and Z 0–20 m`);
  }
  return errors;
}

