"use client";
import {useRef,useState} from "react";
import type {CourseV1,Vector3} from "../contracts/types";
import {courseErrors} from "./course";

export default function GatePanel({course,selected,onSelect,onChange,mode,onMode,onInvalid}:{
  course:CourseV1;selected:number;onSelect:(n:number)=>void;onChange:(position:Vector3,yaw:number)=>void;
  mode:"translate"|"rotate";onMode:(mode:"translate"|"rotate")=>void;onInvalid:(invalid:boolean)=>void;
}){
  const invalid=useRef(new Set<number>());
  const [error,setError]=useState("");
  const gate=course.gates[selected];
  const check=(axis:number,value:string)=>{
    if(value.trim()===""||!Number.isFinite(Number(value)))invalid.current.add(axis);else invalid.current.delete(axis);
    const bad=invalid.current.size>0;setError(bad?"Enter a valid position before flying.":"");onInvalid(bad);
  };
  const commit=(axis:number,value:string)=>{
    check(axis,value);const n=Number(value);if(!gate||value.trim()===""||!Number.isFinite(n))return;
    const position=[...gate.center_m] as Vector3;if(axis<3)position[axis]=n;
    onChange(position,axis===3?n*Math.PI/180:gate.yaw_rad);
  };
  if(!gate)return null;
  return <section className="minimal-editor" aria-label="Gate editor">
    <select aria-label="Selected gate" value={selected} onChange={e=>onSelect(Number(e.target.value))}>{course.gates.map((g,i)=><option key={g.id} value={i}>Gate {g.label}</option>)}</select>
    <div className="edit-modes" role="group" aria-label="Gate manipulation">
      <button aria-pressed={mode==="translate"} onClick={()=>onMode("translate")}>Move</button>
      <button aria-pressed={mode==="rotate"} onClick={()=>onMode("rotate")}>Rotate</button>
    </div>
    <details className="gate-position"><summary>Position</summary><div className="position-fields">
      {[...gate.center_m,gate.yaw_rad*180/Math.PI].map((v,i)=><label key={gate.id+"-"+i}>
        {["X (m)","Y (m)","Height (m)","Direction (°)"][i]}
        <input key={gate.id+"-"+i+"-"+v} aria-label={["Gate X","Gate Y","Gate Z","Gate heading"][i]} type="number" step={i===3?1:.1} defaultValue={Number(v.toFixed(4))}
          onChange={e=>check(i,e.target.value)} onBlur={e=>commit(i,e.target.value)} onKeyDown={e=>{if(e.key==="Enter")e.currentTarget.blur();}}/>
      </label>)}
    </div></details>
    {[error,...courseErrors(course)].filter(Boolean).map((message,i)=><p className="editor-error" role="alert" key={i}>{message}</p>)}
  </section>;
}
