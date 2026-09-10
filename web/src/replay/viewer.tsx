"use client";
import dynamic from "next/dynamic";
import Link from "next/link";
import {useEffect,useRef,useState} from "react";
import type {Replay} from "./npz";
import SceneBoundary from "./scene-boundary";
import type {CourseV1,Vector3} from "../contracts/types";
import GatePanel from "../editor/panel";
import {editGate,courseErrors} from "../editor/course";

const Scene=dynamic(()=>import("./scene"),{ssr:false});
function preview(course:CourseV1):Replay{const s=course.initial_state;return {course,stage:6,states:Float64Array.from([0,...s.position_m,...s.quaternion_wxyz,...s.velocity_world_mps,...s.omega_body_radps,...s.motor_thrust_n]),count:1,duration:0,reward:0,outcome:"course_preview",events:[]};}

export default function Viewer(){
  const worker=useRef<Worker|null>(null);
  const [course,setCourse]=useState<CourseV1|null>(null);
  const [preset,setPreset]=useState<CourseV1|null>(null);
  const [replay,setReplay]=useState<Replay|null>(null);
  const [editing,setEditing]=useState(true);
  const [live,setLive]=useState(false);
  const [busy,setBusy]=useState(false);
  const [invalid,setInvalid]=useState(false);
  const [selected,setSelected]=useState(0);
  const [mode,setMode]=useState<"translate"|"rotate">("translate");
  const [time,setTime]=useState(0);
  const [reset,setReset]=useState(0);
  const [thirdPerson,setThirdPerson]=useState(false);
  const [controls,setControls]=useState([0,0,0,0]);
  const [helpOpen,setHelpOpen]=useState(false);
  const [error,setError]=useState("");
  useEffect(()=>{
    const controller=new AbortController();
    fetch("/courses.json",{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error("Could not load the gates. Please reload.");return r.json();})
      .then((courses:{stage:number;variant:number;course:CourseV1}[])=>{
        const initial=courses.find(c=>c.stage===6&&c.variant===1)?.course;
        if(!initial)throw new Error("The ten-gate course is unavailable.");
        setPreset(initial);setCourse(initial);setReplay(preview(initial));
      }).catch(e=>{if(!controller.signal.aborted)setError(String(e));});
    return()=>{controller.abort();worker.current?.terminate();};
  },[]);
  const showCourse=(next:CourseV1)=>{setCourse(next);setReplay(preview(next));setTime(0);setError("");};
  const stop=()=>{worker.current?.terminate();worker.current=null;setLive(false);setBusy(false);if(course)showCourse(course);setEditing(true);};
  const fly=()=>{
    if(!course||invalid)return;
    const problems=courseErrors(course);if(problems.length){setError(problems.join("; "));return;}
    const chosen=structuredClone(course);
    worker.current?.terminate();setEditing(false);setLive(true);setBusy(true);setError("");setTime(0);setReplay(preview(chosen));
    const current=new Worker(new URL("../simulation/flight.worker.ts",import.meta.url),{type:"module"});worker.current=current;
    let states=new Float64Array(0);
    const finish=()=>{setLive(false);setBusy(false);current.terminate();worker.current=null;};
    current.onmessage=({data})=>{
      if(worker.current!==current)return;
      if(data.type==="frame"){
        const appended=new Float64Array(states.length+data.rows.length*18);appended.set(states);appended.set(data.rows.flat(),states.length);states=appended;
        setReplay({course:chosen,states,count:states.length/18,duration:data.elapsed,outcome:data.outcome,stage:6,reward:data.reward,events:data.events});setControls(data.action??[0,0,0,0]);setTime(data.elapsed);setBusy(false);
      }else if(data.type==="error"){setError("Flight stopped: "+data.message);finish();}
      else if(data.type==="done")finish();
    };
    current.onerror=event=>{if(worker.current===current){setError(event.message||"Unable to start the flight.");finish();}};
    current.postMessage({type:"start",course:chosen,policy:"active"});
  };
  const changeGate=(position:Vector3,yaw:number)=>{if(course)showCourse(editGate(course,selected,position,yaw));};
  return <main className="minimal-flight">
    <p className="flight-intro">Move the gates around - The Reinforcement Learning Drone adapts.</p>
    <Link className="training-link" href="/montage">Training montage ↗</Link>
    <button className="help-widget" aria-label="About this reinforcement learning drone" onClick={()=>setHelpOpen(true)}>?</button>
    {helpOpen&&<div className="help-backdrop" role="presentation" onClick={()=>setHelpOpen(false)}>
      <section className="help-popup" role="dialog" aria-modal="true" aria-labelledby="help-title" onClick={event=>event.stopPropagation()}>
        <button className="help-close" aria-label="Close summary" onClick={()=>setHelpOpen(false)}>×</button>
        <h1 id="help-title">About this drone</h1>
        <p>This is a local 3D flight simulator. Move or rotate the ten gates, then press Fly to run the neural policy in your browser.</p>
        <p>The policy was trained in simulation with reinforcement learning. It learned from earlier gate flights, then practiced wide spacing, different heights, turns, reversals, and recovery from drift, overshoot, wrong altitude, and wrong heading.</p>
        <p>Training keeps earlier successful skills through retention tests. New models must also pass held-out recovery tests across independent training seeds before they can replace the website model.</p>
        <Link href="/montage" onClick={()=>setHelpOpen(false)}>View the training montage →</Link>
      </section>
    </div>}
    <div className="flight-canvas" aria-label="Interactive drone course">
      {replay&&<SceneBoundary><Scene replay={replay} time={time} chase={thirdPerson} reset={reset} editor={editing&&course?{selected,mode,cameraCourse:preset??course,onSelect:n=>{setSelected(n);setInvalid(false);},onChange:changeGate}:undefined}/></SceneBoundary>}
    </div>
    {live&&<section className="control-monitor" aria-label="Live policy outputs">
      <div className="control-monitor-heading"><span>Live outputs</span><small>normalized</small></div>
      {["Throttle","Roll","Pitch","Yaw"].map((label,index)=>{
        const value=controls[index]??0;const width=Math.abs(value)*50;const left=value<0?50-width:50;
        return <div className="control-row" key={label}>
          <div className="control-label"><span>{label}</span><strong>{value.toFixed(2)}</strong></div>
          <div className="control-track"><i className="control-midpoint"/><i className="control-fill" style={{left:`${left}%`,width:`${width}%`,background:value<0?"#e88367":"#5fc7b0"}}/></div>
        </div>;
      })}
    </section>}
    <div className="minimal-controls">
      {editing&&course&&<GatePanel key={selected+":"+reset} course={course} selected={selected} onSelect={n=>{setSelected(n);setInvalid(false);}} onChange={changeGate} mode={mode} onMode={setMode} onInvalid={setInvalid}/>}
      <div className="flight-actions">
        {live?<button className="fly-button" onClick={stop}>Stop</button>:<button className="fly-button" disabled={!course||invalid} onClick={fly}>Fly</button>}
        {!editing&&!live&&<button onClick={stop}>Edit gates</button>}
        {editing&&<button disabled={!preset} onClick={()=>{if(preset)showCourse(structuredClone(preset));setSelected(0);setInvalid(false);setReset(n=>n+1);}}>Reset gates</button>}
        <button aria-pressed={thirdPerson} onClick={()=>setThirdPerson(value=>!value)}>{thirdPerson?"Course view":"Third person"}</button>
        <button onClick={()=>setReset(n=>n+1)}>Reset view</button>
      </div>
    </div>
    {(!course||busy)&&!error&&<p className="minimal-status" role="status">{busy?"Preparing flight…":"Loading…"}</p>}
    {error&&<p className="minimal-error" role="alert">{error}</p>}
  </main>;
}
