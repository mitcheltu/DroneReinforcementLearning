"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import { readReplay, type Replay } from "./npz";
import { sample } from "./timeline";
import SceneBoundary from "./scene-boundary";
import type { CourseV1, Vector3 } from "../contracts/types";
import GatePanel from "../editor/panel";
import {editGate,courseErrors} from "../editor/course";
import {loadBytes} from "../contracts/loader";

function preview(course:CourseV1,stage:number):Replay{const s=course.initial_state;return {course,stage,states:Float64Array.from([0,...s.position_m,...s.quaternion_wxyz,...s.velocity_world_mps,...s.omega_body_radps,...s.motor_thrust_n]),count:1,duration:0,reward:0,outcome:"course_preview",events:[]};}

const Scene = dynamic(() => import("./scene"), { ssr: false, loading: () => <p className="canvas-message">Preparing 3D view…</p> });
const label = (value: string) => value.replaceAll("_", " ");

export default function Viewer() {
  const worker = useRef<Worker|null>(null);
  const [live,setLive]=useState(false);
  const [courses,setCourses]=useState<{stage:number;variant:number;course:CourseV1}[]>([]);
  const [stage,setStage]=useState(6);
  const [variant,setVariant]=useState(1);
  const [drafts,setDrafts]=useState<Record<string,CourseV1>>({});
  const [editing,setEditing]=useState(false);
  const [invalidEdit,setInvalidEdit]=useState(false);
  const [selectedGate,setSelectedGate]=useState(0);
  const [editMode,setEditMode]=useState<"translate"|"rotate">("translate");
  const courseFile=useRef<HTMLInputElement>(null);
  const courseKey=`${stage}:${variant}`;
  const preset=courses.find(c=>c.stage===stage&&c.variant===variant);
  const activeCourse=drafts[courseKey]??preset?.course;
  const [replay,setReplay] = useState<Replay|null>(null);
  const [name,setName] = useState("");
  const [error,setError] = useState("");
  const [busy,setBusy] = useState(false);
  const [playing,setPlaying] = useState(false);
  const [time,setTime] = useState(0);
  const [speed,setSpeed] = useState(1);
  const [chase,setChase] = useState(false);
  const [reset,setReset] = useState(0);
  const generation = useRef(0);
  const file = useRef<HTMLInputElement>(null);
  const load = useCallback(async (source: File | string, title: string) => {
    worker.current?.terminate();worker.current=null;setLive(false);setEditing(false);setInvalidEdit(false);
    const request = ++generation.current;
    setPlaying(false); setBusy(true); setError("");
    try {
      if (source instanceof File && source.size > 32*1024*1024) throw new Error("Select a trace smaller than 32 MiB.");
      const buffer = source instanceof File ? await source.arrayBuffer() : await fetch(source).then(r => { if(!r.ok) throw new Error("Example could not be loaded."); return r.arrayBuffer(); });
      const parsed = await readReplay(new Uint8Array(buffer));
      if (request !== generation.current) return;
      setReplay(parsed); setName(title); setTime(0); setReset(v=>v+1);
    } catch (e) { if(request === generation.current) setError(e instanceof Error?e.message:"Replay could not be read."); }
    finally { if(request === generation.current) setBusy(false); }
  },[]);
  useEffect(()=>{
    const controller=new AbortController();
    fetch("/courses.json",{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error("Courses could not be loaded");return r.json();}).then(setCourses).catch(e=>{if(!controller.signal.aborted)setError(String(e));});
    return()=>{controller.abort();worker.current?.terminate();};
  },[]);
  const startFlight=()=>{
    if(!activeCourse||invalidEdit)return;const problems=courseErrors(activeCourse);if(problems.length){setError(problems.join("; "));return;}
    const chosen={stage,course:structuredClone(activeCourse)};setEditing(false);
    generation.current++;worker.current?.terminate();setPlaying(false);setBusy(true);setLive(true);setError("");
    setName(`Neural pilot · stage ${stage} · variation ${variant}`);setTime(0);setReset(v=>v+1);
    const initial=chosen.course.initial_state;
    setReplay({course:chosen.course,states:Float64Array.from([0,...initial.position_m,...initial.quaternion_wxyz,...initial.velocity_world_mps,...initial.omega_body_radps,...initial.motor_thrust_n]),count:1,duration:0,outcome:"loading",stage:chosen.stage,reward:0,events:[]});
    const current=new Worker(new URL("../simulation/flight.worker.ts",import.meta.url),{type:"module"});worker.current=current;
    let states=new Float64Array(0);
    current.onmessage=({data})=>{
      if(worker.current!==current)return;
      if(data.type==="frame"){
        const appended=new Float64Array(states.length+data.rows.length*18);appended.set(states);appended.set(data.rows.flat(),states.length);states=appended;
        setReplay({course:chosen.course,states,count:states.length/18,duration:data.elapsed,outcome:data.outcome,stage:chosen.stage,reward:data.reward,events:data.events});
        setTime(data.elapsed);setBusy(false);
      }else if(data.type==="error"){setError(`Neural flight stopped: ${data.message}`);setLive(false);setBusy(false);current.terminate();worker.current=null;}
      else if(data.type==="done"){setLive(false);setBusy(false);current.terminate();worker.current=null;}
    };
    current.onerror=(event)=>{if(worker.current===current){setError(event.message||"Flight worker failed");setLive(false);setBusy(false);current.terminate();worker.current=null;}};
    current.postMessage({type:"start",course:chosen.course});
  };
  const showDraft=(course:CourseV1)=>{generation.current++;setDrafts(old=>({...old,[courseKey]:course}));setReplay(preview(course,stage));setName(`Course editor · stage ${stage} · variation ${variant}`);setTime(0);setPlaying(false);setChase(false);setError("");};
  const beginEditing=()=>{if(!activeCourse)return;showDraft(drafts[courseKey]??{...structuredClone(activeCourse),course_id:crypto.randomUUID(),mode:"experimental",generator:null});setSelectedGate(0);setInvalidEdit(false);setEditing(true);setReset(n=>n+1);};
  const changeGate=(position:Vector3,yaw:number)=>{if(activeCourse)showDraft(editGate(activeCourse,selectedGate,position,yaw));};
  const resetLayout=()=>{if(!preset)return;showDraft({...structuredClone(preset.course),course_id:crypto.randomUUID(),mode:"experimental",generator:null});setSelectedGate(0);setInvalidEdit(false);setReset(n=>n+1);};
  const saveCourse=()=>{if(!activeCourse)return;const url=URL.createObjectURL(new Blob([JSON.stringify(activeCourse,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="aerorl-custom-course.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  const importCourse=async(file:File)=>{const request=++generation.current;setBusy(true);try{if(file.size>1024*1024)throw new Error("Course JSON must be smaller than 1 MiB");const {data}=await loadBytes<CourseV1>("editable-course",new Uint8Array(await file.arrayBuffer()));if(request!==generation.current)return;const errors=courseErrors(data);if(errors.length)throw new Error(errors.join("; "));const nextStage=({0:0,1:1,3:4,10:6} as Record<number,number>)[data.gates.length]!;setStage(nextStage);setVariant(1);setDrafts(old=>({...old,[`${nextStage}:1`]:data}));setEditing(true);setSelectedGate(0);setInvalidEdit(false);setReset(n=>n+1);setReplay(preview(data,nextStage));setName("Imported custom course");setTime(0);setPlaying(false);setChase(false);setError("");}catch(e){if(request===generation.current)setError(e instanceof Error?e.message:String(e));}finally{if(request===generation.current)setBusy(false);}};
  const stopFlight=()=>{worker.current?.terminate();worker.current=null;setLive(false);setBusy(false);setReplay(r=>r&&(r.outcome==="running"||r.outcome==="loading")?{...r,outcome:"stopped"}:r);};
  const invalidate = useCallback(()=>{generation.current++;},[]);
  useEffect(() => { const timer=window.setTimeout(()=>void load("/examples/reference.npz","Reference controller · three gates"),0); return () => {clearTimeout(timer);invalidate();}; },[load,invalidate]);
  useEffect(() => {
    const pause = () => { if(document.hidden) setPlaying(false); };
    document.addEventListener("visibilitychange",pause); return () => document.removeEventListener("visibilitychange",pause);
  },[]);
  const atEnd = !live && !!replay && time >= replay.duration;
  useEffect(() => {
    if(!playing || !replay || atEnd) return;
    let frame=0, previous=performance.now();
    const animate = (now:number) => { const dt=Math.min((now-previous)/1000,.1); previous=now; setTime(t=>Math.min(replay.duration,t+dt*speed)); frame=requestAnimationFrame(animate); };
    frame=requestAnimationFrame(animate); return ()=>cancelAnimationFrame(frame);
  },[playing,replay,speed,atEnd]);
  const row = replay ? sample(replay,time) : null;
  const passed = replay?.events.filter(e=>e.type==="gate_pass" && e.time<=time).length ?? 0;
  const seek = (value:number) => {if(live)return;setPlaying(false);setTime(value);};
  return <main className="workbench">
    <section className="live-controls" aria-label="Neural flight controls">
      <strong>Neural pilot</strong><label>Course <select aria-label="Course stage" disabled={live||busy} value={stage} onChange={e=>{setStage(Number(e.target.value));setEditing(false);setInvalidEdit(false);}}>{["Hover","One gate","One gate · varied","Three gates","Three gates · turns","Ten gates","Ten gates · varied"].map((title,i)=><option value={i} key={i}>{title}</option>)}</select></label>
      <label>Variation <select aria-label="Course variation" disabled={live||busy} value={variant} onChange={e=>{setVariant(Number(e.target.value));setEditing(false);setInvalidEdit(false);}}>{Array.from({length:16},(_,i)=><option key={i} value={i+1}>{i+1}</option>)}</select></label>
      <button className="primary" disabled={live||!courses.length||busy||invalidEdit} onClick={startFlight}>Fly with ONNX</button><button disabled={!live} onClick={stopFlight}>Stop flight</button>
      <button disabled={live||busy||!activeCourse} onClick={beginEditing}>Edit gates</button><button disabled={live||busy} onClick={()=>courseFile.current?.click()}>Load course JSON</button>
      <input ref={courseFile} hidden type="file" accept=".json" onChange={e=>{const f=e.target.files?.[0];if(f)void importCourse(f);e.target.value="";}}/>
      <a className="nav-link" href="/montage">Training montage ↗</a>
      <span role="status">{live?(busy?"Loading neural pilot…":"Live browser simulation") : "Runs locally in your browser"}</span>
    </section>
    {editing&&activeCourse&&<GatePanel key={`${courseKey}:${selectedGate}:${reset}`} course={activeCourse} selected={selectedGate} onSelect={n=>{setSelectedGate(n);setInvalidEdit(false);}} onChange={changeGate} mode={editMode} onMode={setEditMode} onReset={resetLayout} onSave={saveCourse} onInvalid={setInvalidEdit}/>}
    <header className="topbar"><div className="brand">AERO<span>RL</span><small>FLIGHT LAB</small></div><div className="mode">{live?"Live neural flight":"Flight replay"} <span>LOCAL FLIGHT LAB</span></div><button className="primary" onClick={()=>file.current?.click()} disabled={busy||live}>Import training attempt <span>↗</span></button><input ref={file} type="file" accept=".npz" hidden onChange={e=>{const selected=e.target.files?.[0]; if(selected) void load(selected,selected.name); e.target.value="";}} /></header>
    <div className="workspace">
      <section className="flight-panel" aria-label="Flight replay">
        <div className="view-heading"><div><p className="eyebrow">ATTEMPT EXPLORER</p><h1>{name || "Loading flight"}</h1></div><span className="badge">{replay?label(replay.outcome):"Loading"}</span></div>
        {error && <div role="alert" className="error">{error} </div>}
        <div className="viewport">{replay && <SceneBoundary key={replay.course.course_id}><Scene replay={replay} time={time} chase={chase} reset={reset} editor={editing&&activeCourse?{selected:selectedGate,mode:editMode,cameraCourse:preset?.course??activeCourse,onSelect:n=>{setSelectedGate(n);setInvalidEdit(false);},onChange:changeGate}:undefined} /></SceneBoundary>}{busy && <div className="loading">Reading flight…</div>}<div className="view-tools"><button onClick={()=>setChase(v=>!v)} aria-pressed={chase}>{chase?"Chase camera":"Orbit camera"}</button><button onClick={()=>setReset(v=>v+1)} aria-label="Reset camera">Reset view</button></div><div className="view-note">Z ↑ <span>meters</span> · {chase?"Following drone":"Drag to orbit · Scroll to zoom"}</div></div>
        <div className="transport"><div className="timeline-label"><span>{time.toFixed(2)} s</span><span>{replay?.duration.toFixed(2) ?? "0.00"} s</span></div><input aria-label="Replay time" type="range" min={0} max={replay?.duration ?? 1} step={.001} value={time} disabled={!replay || busy||live||replay.outcome==="course_preview"} onChange={e=>seek(Number(e.target.value))} /><div className="transport-buttons"><button className="play" disabled={!replay || busy||live||replay.outcome==="course_preview"} onClick={()=>{if(atEnd){setTime(0);setPlaying(true);}else setPlaying(v=>!v);}}>{playing&&!atEnd?"Ⅱ Pause":atEnd?"↺ Replay":"▶ Play"}</button><button disabled={!replay || busy||live||replay.outcome==="course_preview"} onClick={()=>seek(0)}>Restart</button><label>Replay speed <select disabled={live} value={speed} onChange={e=>setSpeed(Number(e.target.value))}>{[.25,.5,1,2].map(v=><option value={v} key={v}>{v}×</option>)}</select></label><span className="transport-status">{live?"Flying with neural policy":editing?"Configure gates before flying":atEnd?"Attempt ended":playing?"Playing recording":"Paused"}</span></div></div>
        <div className="telemetry">{[["ALTITUDE",`${row?.[3]?.toFixed(2) ?? "—"} m`],["SPEED",`${row?Math.hypot(row[8]!,row[9]!,row[10]!).toFixed(2):"—"} m/s`],["GATES PASSED",`${passed} / ${replay?.course.gates.length ?? 0}`],["EPISODE REWARD",replay?.reward.toFixed(2) ?? "—"]].map(([title,value])=><div key={title}><span>{title}</span><strong>{value}</strong></div>)}</div>
      </section>
      <aside><section className="panel-section"><p className="eyebrow">FLIGHT LIBRARY</p><button className="example" disabled={busy||live} onClick={()=>void load("/examples/reference.npz","Reference controller · three gates")}><strong>Three-gate reference</strong><span>Controller baseline · Successful flight</span></button><button className="example" disabled={busy||live} onClick={()=>void load("/examples/failure.npz","Training attempt · hover departure")}><strong>Early training failure</strong><span>Actual PPO attempt · Hover stage</span></button><p className="hint">Open an <code>.npz</code> from your run’s <code>training-traces</code> folder. Files stay in this browser.</p></section>
        <section className="panel-section"><div className="section-title"><p className="eyebrow">COURSE PROGRESS</p><span>Stage {replay?.stage ?? "—"}</span></div>{replay?.course.gates.length===0?<p className="hint">Hover task · Hold the starting position for 5 seconds.</p>:<ol className="gates">{replay?.course.gates.map((g,i)=><li className={i<passed?"complete":i===passed?"target":""} key={g.id}><b>{i<passed?"✓":g.label}</b><div><strong>Gate {g.label}</strong><small>{g.center_m[2].toFixed(1)} m altitude</small></div><span>{i<passed?"Passed":i===passed?"Target":"Ahead"}</span></li>)}</ol>}</section>
        <section className="panel-section events"><p className="eyebrow">EVENTS</p>{replay?.events.length===0?<p className="hint">No gate events in this attempt.</p>:replay?.events.map((e,i)=><button key={i} onClick={()=>seek(e.time)}><span>{e.time.toFixed(2)} s</span>{label(e.type)}{e.label?` · ${e.label}`:""}</button>)}{replay && !live && <button onClick={()=>seek(replay.duration)}><span>{replay.duration.toFixed(2)} s</span>End · {label(replay.outcome)}</button>}</section>
      </aside>
    </div><footer><span>120 Hz physics · 60 Hz neural control</span><span>Live flights use the neural actor. Library reference flights use the controller baseline.</span></footer>
  </main>;
}




