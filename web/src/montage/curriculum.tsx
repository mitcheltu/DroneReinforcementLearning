"use client";
import {useEffect,useMemo,useState} from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import type {Attempt} from "./data";
import SceneBoundary from "../replay/scene-boundary";
const Scene=dynamic(()=>import("./scene"),{ssr:false});
type Trace=Attempt&{phase:string};
type Lesson={id:string;label:string;count:number;url:string};
type Index={runs:{id:string;lessons:Lesson[]}[]};
export default function Curriculum(){
 const [index,setIndex]=useState<Index|null>(null),[run,setRun]=useState("all"),[lesson,setLesson]=useState("gentle"),[outcome,setOutcome]=useState("all"),[phase,setPhase]=useState("all");
 const [loaded,setLoaded]=useState<{key:string;traces:Trace[]}|null>(null),[error,setError]=useState("");
 const [time,setTime]=useState(0),[playing,setPlaying]=useState(true),[reset,setReset]=useState(0),[gates,setGates]=useState(false),[aligned,setAligned]=useState(true),[normalized,setNormalized]=useState(true);
 useEffect(()=>{const abort=new AbortController();fetch("/curriculum-montage/index.json",{signal:abort.signal,cache:"no-store"}).then(r=>{if(!r.ok)throw new Error("Recordings unavailable");return r.json();}).then(setIndex).catch(e=>{if(!abort.signal.aborted)setError(String(e));});return()=>abort.abort();},[]);
 const lessons=useMemo(()=>{const result=new Map<string,Lesson>();index?.runs.filter(r=>run==="all"||r.id===run).forEach(r=>r.lessons.forEach(l=>{if(l.count)result.set(l.id,l);}));return [...result.values()];},[index,run]);
 const selected=lessons.find(l=>l.id===lesson)??lessons[0];
 const key=JSON.stringify(index?.runs.filter(r=>run==="all"||r.id===run).flatMap(r=>r.lessons.filter(l=>l.id===selected?.id&&l.count).map(l=>l.url))??[]);
 useEffect(()=>{const abort=new AbortController();const urls=JSON.parse(key) as string[];Promise.all(urls.map(url=>fetch(url,{signal:abort.signal,cache:"no-store"}).then(r=>{if(!r.ok)throw new Error("Could not load scenario recordings");return r.json() as Promise<Trace[]>;}))).then(groups=>{setLoaded({key,traces:groups.flat()});setError("");}).catch(e=>{if(!abort.signal.aborted)setError(String(e));});return()=>abort.abort();},[key]);
 const attempts=useMemo(()=>loaded?.key===key?loaded.traces.filter(a=>(phase==="all"||a.phase===phase)&&(outcome==="all"||(outcome==="success"?a.outcome==="success":a.outcome!=="success"))):[],[loaded,key,phase,outcome]);
 const maximum=normalized?1:Math.max(0,...attempts.map(a=>a.duration));
 useEffect(()=>{if(!playing||!attempts.length)return;let frame=0,last=performance.now();const tick=(now:number)=>{const dt=Math.min(.1,(now-last)/1000);last=now;setTime(t=>t>=maximum?0:Math.min(maximum,t+dt*(normalized?1/20:1)));frame=requestAnimationFrame(tick);};frame=requestAnimationFrame(tick);return()=>cancelAnimationFrame(frame);},[playing,maximum,normalized,attempts.length]);
 const rewind=()=>{setTime(0);setReset(n=>n+1);};
 return <main className="minimal-flight montage-minimal">
  <div className="flight-canvas"><SceneBoundary><Scene attempts={attempts} aligned={aligned} opacity={.15} time={Math.min(time,maximum)} normalized={normalized} showGates={gates} reset={reset}/></SceneBoundary></div>
  <div className="montage-top"><Link href="/">← Fly</Link><select aria-label="Training scenario" value={selected?.id??""} onChange={e=>{setLesson(e.target.value);rewind();}}>{lessons.map(l=><option key={l.id} value={l.id}>{l.label}</option>)}</select><details className="montage-options"><summary>Filters</summary><div>
   <label>Run<select aria-label="Training run" value={run} onChange={e=>{setRun(e.target.value);rewind();}}><option value="all">All runs</option>{index?.runs.map(r=><option key={r.id}>{r.id}</option>)}</select></label>
   <label>Attempts<select aria-label="Attempt type" value={phase} onChange={e=>{setPhase(e.target.value);rewind();}}><option value="all">All recorded attempts</option><option value="ppo">PPO & critic preparation</option><option value="demonstration">Demonstrations & anchors</option><option value="dagger">Learner corrections</option><option value="evaluation">Evaluations</option><option value="baseline">Baselines</option></select></label>
   <label>Outcome<select aria-label="Attempt outcome" value={outcome} onChange={e=>{setOutcome(e.target.value);rewind();}}><option value="all">Successes & failures</option><option value="success">Successes</option><option value="failure">Failures</option></select></label>
   <label><input type="checkbox" checked={aligned} onChange={e=>{setAligned(e.target.checked);rewind();}}/>Align starts and headings</label>
   <label><input type="checkbox" checked={gates} onChange={e=>setGates(e.target.checked)}/>Show each gate layout</label>
   <label><input type="checkbox" checked={normalized} onChange={e=>{setNormalized(e.target.checked);rewind();}}/>Synchronize flight progress</label>
  </div></details></div>
  <p className="montage-caption">Recorded flights · <span className="success-key">success</span> / <span className="failure-key">failure</span></p>
  <div className="minimal-controls montage-playback"><button onClick={()=>setPlaying(v=>!v)}>{playing?"Pause":"Play"}</button><input aria-label="Montage progress" type="range" min={0} max={maximum||1} step={.001} value={Math.min(time,maximum)} onChange={e=>{setPlaying(false);setTime(Number(e.target.value));}}/><button onClick={rewind}>Restart</button></div>
  {error?<p className="minimal-error" role="alert">{error}</p>:(!index||loaded?.key!==key)?<p className="minimal-status" role="status">Loading recordings…</p>:!attempts.length&&<p className="minimal-status" role="status">No recordings match these filters.</p>}
 </main>;
}
