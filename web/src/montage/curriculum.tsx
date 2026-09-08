"use client";
import {useEffect,useMemo,useState} from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import type {Attempt} from "./data";
import SceneBoundary from "../replay/scene-boundary";

const Scene=dynamic(()=>import("./scene"),{ssr:false});
type TrainingAttempt=Attempt&{phase:string};
type Index={generated_at:string;runs:{id:string;status:string;qualified:boolean;lessons:{id:string;label:string;count:number;url:string}[];validation:{name:string;successes:number;cases:number}[]}[]};
export default function Curriculum(){
  const [index,setIndex]=useState<Index|null>(null),[run,setRun]=useState(""),[lesson,setLesson]=useState(""),[phase,setPhase]=useState("training");
  const [data,setData]=useState<{url:string;traces:TrainingAttempt[]}|null>(null),[error,setError]=useState("");
  const [time,setTime]=useState(0),[playing,setPlaying]=useState(false),[opacity,setOpacity]=useState(.25),[aligned,setAligned]=useState(true),[gates,setGates]=useState(false),[reset,setReset]=useState(0);
  useEffect(()=>{const abort=new AbortController();fetch("/curriculum-montage/index.json",{signal:abort.signal,cache:"no-store"}).then(r=>{if(!r.ok)throw new Error("Curriculum snapshot is unavailable");return r.json();}).then(setIndex).catch(e=>{if(!abort.signal.aborted)setError(String(e));});return()=>abort.abort();},[]);
  const selected=index?.runs.find(r=>r.id===run)??index?.runs.at(-1);
  const group=selected?.lessons.find(l=>l.id===lesson)??selected?.lessons[0];
  const url=String(group?.url??"");
  useEffect(()=>{if(!url)return;const abort=new AbortController();fetch(url,{signal:abort.signal,cache:"no-store"}).then(r=>{if(!r.ok)throw new Error("Lesson recordings could not be loaded");return r.json();}).then(traces=>setData({url,traces})).catch(e=>{if(!abort.signal.aborted)setError(String(e));});return()=>abort.abort();},[url]);
  const attempts=useMemo(()=>data&&data.url===url?data.traces.filter(a=>phase==="all"||(phase==="training"?["ppo","demonstration","dagger"].includes(a.phase):a.phase===phase)):[],[data,url,phase]);
  const maximum=Math.max(0,...attempts.map(a=>a.duration)),ended=time>=maximum;
  useEffect(()=>{if(!playing||ended)return;let frame=0,last=performance.now();const tick=(now:number)=>{setTime(t=>Math.min(maximum,t+Math.min(.1,(now-last)/1000)));last=now;frame=requestAnimationFrame(tick);};frame=requestAnimationFrame(tick);return()=>cancelAnimationFrame(frame);},[playing,ended,maximum]);
  const rewind=()=>{setPlaying(false);setTime(0);};
  return <main className="montage-page"><header className="topbar"><div className="brand">AERO<span>RL</span><small>CURRICULUM RECORDINGS</small></div><Link className="nav-link" href="/montage">Original training montage</Link><Link className="nav-link" href="/">Gate editor</Link></header>
    <section className="montage-controls">
      <label>Run <select aria-label="Curriculum run" value={selected?.id??""} onChange={e=>{setRun(e.target.value);setLesson("");rewind();}}>{index?.runs.map(r=><option key={r.id}>{r.id}</option>)}</select></label>
      <label>Lesson <select aria-label="Curriculum lesson" value={group?.id??""} onChange={e=>{setLesson(e.target.value);rewind();}}>{selected?.lessons.map(l=><option key={l.id} value={l.id}>{l.label}</option>)}</select></label>
      <label>Attempts <select aria-label="Curriculum phase" value={phase} onChange={e=>{setPhase(e.target.value);rewind();}}><option value="training">All training rollouts</option><option value="ppo">PPO & critic preparation</option><option value="demonstration">Expert demonstrations</option><option value="dagger">Learner correction rollouts</option><option value="evaluation">Candidate evaluations</option><option value="baseline">Original policy comparison</option><option value="all">All recorded attempts</option></select></label>
    </section>
    <section className="montage-body"><h1>{group?.label??"Training recordings"} · {attempts.length} attempts</h1>
      <p className="hint">{selected?.status}. {selected?.qualified?"Passed the run’s development screen.":"Not qualified for release."} Snapshot: {index?new Date(index.generated_at).toLocaleString():"loading"}. Every matching recorded attempt is overlaid; green is success and coral is failure. Demonstrations and learner corrections are training data, not proof of policy success. PPO recordings include deterministic critic-preparation rollouts.</p>
      {error&&<p className="error" role="alert">{error}</p>}
      <div className="montage-settings"><label>Opacity {Math.round(opacity*100)}% <input aria-label="Curriculum opacity" type="range" min={.03} max={1} step={.01} value={opacity} onChange={e=>setOpacity(Number(e.target.value))}/></label><label><input type="checkbox" checked={aligned} onChange={e=>setAligned(e.target.checked)}/> Align starts</label><label><input type="checkbox" checked={gates} onChange={e=>setGates(e.target.checked)}/> Show gate layouts</label><button onClick={()=>setReset(n=>n+1)}>Reset camera</button></div>
      <div className="viewport montage-viewport"><SceneBoundary><Scene attempts={attempts} aligned={aligned} opacity={opacity} time={Math.min(time,maximum)} normalized={false} showGates={gates} reset={reset}/></SceneBoundary>{!attempts.length&&<div className="loading">{data?.url!==url?"Loading lesson recordings…":"No attempts for these filters"}</div>}</div>
      <div className="montage-settings"><button disabled={!attempts.length} onClick={()=>{if(ended){setTime(0);setPlaying(true);}else setPlaying(v=>!v);}}>{playing&&!ended?"Pause":"Play all attempts"}</button><button onClick={rewind}>Restart</button><span>{Math.min(time,maximum).toFixed(2)} / {maximum.toFixed(2)} s</span></div>
      <input aria-label="Curriculum time" type="range" min={0} max={maximum||1} step={.01} value={Math.min(time,maximum)} onChange={e=>{setTime(Number(e.target.value));setPlaying(false);}}/>
      <p className="hint">Full paths remain visible; bodies freeze at episode endpoints. Display recordings retain at most 601 poses per attempt. Sources and configurations differ between experiments; compare each candidate with its own paired baseline evaluation.</p>
    </section></main>;
}
