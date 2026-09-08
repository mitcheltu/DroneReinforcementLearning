"use client";
import {useRef,useState} from "react";
import type {CourseV1,Vector3} from "../contracts/types";
import {courseErrors} from "./course";

export default function GatePanel({course,selected,onSelect,onChange,mode,onMode,onReset,onSave,onInvalid}:{
  course:CourseV1;selected:number;onSelect:(n:number)=>void;onChange:(position:Vector3,yaw:number)=>void;
  mode:"translate"|"rotate";onMode:(mode:"translate"|"rotate")=>void;onReset:()=>void;onSave:()=>void;onInvalid:(invalid:boolean)=>void;
}){
  const invalid=useRef(new Set<number>());
  const check=(axis:number,value:string)=>{if(value.trim()===""||!Number.isFinite(Number(value)))invalid.current.add(axis);else invalid.current.delete(axis);const bad=invalid.current.size>0;setError(bad?"Enter valid numbers before flying":"");onInvalid(bad);};
  const [error,setError]=useState("");const gate=course.gates[selected];
  const commit=(axis:number,value:string)=>{
    check(axis,value);const n=Number(value);if(value.trim()===""||!Number.isFinite(n)){return;}
    if(!gate)return;const position=[...gate.center_m] as Vector3;
    if(axis<3)position[axis]=n;
    onChange(position,axis===3?n*Math.PI/180:gate.yaw_rad);
  };
  return <section className="gate-editor" aria-label="Gate editor">
    <div className="editor-heading"><h2>Configure gates</h2><button onClick={onReset}>Reset layout</button><button onClick={onSave}>Save course JSON</button></div>
    {!gate?<p>Hover has no gates. Choose a gate course to edit.</p>:<>
      <label>Selected gate <select aria-label="Selected gate" value={selected} onChange={e=>onSelect(Number(e.target.value))}>{course.gates.map((g,i)=><option key={g.id} value={i}>Gate {g.label}</option>)}</select></label>
      <div className="coordinate-fields">{[...gate.center_m,gate.yaw_rad*180/Math.PI].map((v,i)=><label key={`${gate.id}-${i}`}>{["X (m)","Y (m)","Z (m)","Heading (degrees)"][i]}<input key={`${gate.id}-${i}-${v}`} aria-label={["Gate X","Gate Y","Gate Z","Gate heading"][i]} type="number" step={i===3?1:.1} defaultValue={Number(v.toFixed(4))} onChange={e=>check(i,e.target.value)} onBlur={e=>commit(i,e.target.value)} onKeyDown={e=>{if(e.key==="Enter")e.currentTarget.blur();}} /></label>)}</div>
      <button aria-pressed={mode==="translate"} onClick={()=>onMode("translate")}>Move handles</button> <button aria-pressed={mode==="rotate"} onClick={()=>onMode("rotate")}>Rotate heading</button>
      <p className="hint">Select a gate in 3D and drag its handles, or enter coordinates and press Enter. Arrows show the required crossing direction. Heading 0° points along +X; 90° along +Y. Gates remain upright and are passed in label order.</p>
      <p className="hint">The drone’s starting pose is unchanged. Custom layouts can exceed the policy’s trained range and may fail.</p>
    </>}
    {[error,...courseErrors(course)].filter(Boolean).map((message,i)=><p className="error" role="alert" key={i}>{message}</p>)}
  </section>;
}


