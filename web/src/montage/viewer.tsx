"use client";
import dynamic from "next/dynamic";
import Link from "next/link";
import {useEffect,useMemo,useRef,useState} from "react";
import {fromReplay,type Attempt,type MontageData} from "./data";
import {readReplay} from "../replay/npz";
import SceneBoundary from "../replay/scene-boundary";

const Scene=dynamic(()=>import("./scene"),{ssr:false});
const stages=["Hover","One gate","One gate · varied","Three gates","Three gates · turns","Ten gates","Ten gates · varied"];
export default function Montage(){
  const [data,setData]=useState<MontageData|null>(null),[imported,setImported]=useState<Attempt[]>([]);
  const [source,setSource]=useState("ppo"),[stage,setStage]=useState(0),[outcome,setOutcome]=useState("all");
  const [aligned,setAligned]=useState(true),[opacity,setOpacity]=useState(.25),[gates,setGates]=useState(false);
  const [time,setTime]=useState(0),[playing,setPlaying]=useState(false),[normalized,setNormalized]=useState(false),[reset,setReset]=useState(0);
  const [error,setError]=useState(""),[loading,setLoading]=useState(false);
  const files=useRef<HTMLInputElement>(null),serial=useRef(0);
  useEffect(()=>{const controller=new AbortController();fetch("/montage.json",{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error("Montage data could not be loaded");return r.json();}).then(setData).catch(e=>{if(!controller.signal.aborted)setError(String(e));});return()=>controller.abort();},[]);
  const attempts=useMemo(()=>[...(data?.traces??[]),...imported].filter(a=>a.source===source&&a.stage===stage&&(outcome==="all"||(outcome==="success"?a.outcome==="success":a.outcome!=="success"))),[data,imported,source,stage,outcome]);
  const max=normalized?1:Math.max(0,...attempts.map(a=>a.duration));const visibleTime=Math.min(time,max);
  const ended=visibleTime>=max;
  useEffect(()=>{if(!playing||!max||ended)return;let frame=0,last=performance.now();const tick=(now:number)=>{const dt=Math.min(.1,(now-last)/1000);last=now;setTime(t=>Math.min(max,t+dt*(normalized?.1:1)));frame=requestAnimationFrame(tick);};frame=requestAnimationFrame(tick);return()=>cancelAnimationFrame(frame);},[playing,max,normalized,ended]);
  const resetPlayback=()=>{setTime(0);setPlaying(false);};
  const importFiles=async(list:FileList)=>{setLoading(true);setError("");try{
    const selected=Array.from(list);if(imported.length+selected.length>1000)throw new Error("Load at most 1,000 attempts per montage");
    if(selected.reduce((sum,f)=>sum+f.size,0)>256*1024*1024)throw new Error("Each import batch must total at most 256 MiB");
    const added:Attempt[]=[];for(const f of selected){if(f.size>32*1024*1024)throw new Error(`${f.name}: file exceeds 32 MiB`);const replay=await readReplay(new Uint8Array(await f.arrayBuffer()));added.push(fromReplay(replay,`import-${serial.current++}-${f.name}`));}
    setImported(old=>[...old,...added]);setSource("imported");if(added[0])setStage(added[0].stage);resetPlayback();
  }catch(e){setError(e instanceof Error?e.message:String(e));}finally{setLoading(false);}};
  const totalLogged=data?.episode_logs.reduce((n,run)=>n+(run.episodes_by_stage[String(stage)]??0),0)??0;
  return <main className="montage-page">
    <header className="topbar"><div className="brand">AERO<span>RL</span><small>TRAINING MONTAGE</small></div><Link className="nav-link" href="/curriculum">New curriculum recordings</Link><Link className="nav-link" href="/">← Gate editor & live flight</Link></header>
    <section className="montage-controls">
      <label>Source <select aria-label="Montage source" value={source} onChange={e=>{setSource(e.target.value);resetPlayback();}}><option value="ppo">PPO training · saved attempts</option><option value="demonstrations">Imitation · reconstructed demonstrations</option><option value="validation">Repaired policy · validation recordings</option><option value="imported">Imported traces</option></select></label>
      <label>Training type <select aria-label="Training type" value={stage} onChange={e=>{setStage(Number(e.target.value));resetPlayback();}}>{stages.map((s,i)=><option key={i} value={i}>{s}</option>)}</select></label>
      <label>Outcomes <select aria-label="Montage outcomes" value={outcome} onChange={e=>{setOutcome(e.target.value);resetPlayback();}}><option value="all">All attempts</option><option value="success">Successes</option><option value="failure">Failures</option></select></label>
      <button disabled={loading} onClick={()=>files.current?.click()}>{loading?"Reading attempts…":"Import multiple NPZ traces"}</button><input hidden multiple type="file" accept=".npz" ref={files} onChange={e=>{if(e.target.files)void importFiles(e.target.files);e.target.value="";}}/>
      {!!imported.length&&<button onClick={()=>{setImported([]);resetPlayback();}}>Clear imported traces</button>}
    </section>
    <section className="montage-body"><h1>{stages[stage]} · {attempts.length} attempts overlaid</h1>
      <p className="hint">{source==="ppo"?`All retained PPO trajectories for this type are available here; the source logs contain ${totalLogged.toLocaleString()} episodes. Most historical trajectories were not recorded.`:source==="demonstrations"?"All 16 original demonstration flights per trained type, reconstructed from their seeds and reference controller. Perturbed label states are not flight trajectories.":source==="validation"?"Saved validation flights of the repaired neural policy. These are evaluation recordings, not training episodes.":"All successfully imported trajectories matching these filters are shown together."}</p>
      {error&&<p className="error" role="alert">{error}</p>}
      <div className="montage-settings"><label><input type="checkbox" checked={aligned} onChange={e=>setAligned(e.target.checked)}/> Align starts & headings</label><label><input type="checkbox" checked={gates} onChange={e=>setGates(e.target.checked)}/> Show every gate layout</label><label>Opacity {Math.round(opacity*100)}% <input aria-label="Montage opacity" type="range" min={.03} max={1} step={.01} value={opacity} onChange={e=>setOpacity(Number(e.target.value))}/></label><button onClick={()=>setReset(n=>n+1)}>Reset montage camera</button></div>
      <div className="viewport montage-viewport"><SceneBoundary><Scene attempts={attempts} aligned={aligned} opacity={opacity} time={visibleTime} normalized={normalized} showGates={gates} reset={reset}/></SceneBoundary>{!attempts.length&&<div className="loading">{data?"No trajectories for these filters. Choose another source/type or import recordings.":"Loading recorded trajectories…"}</div>}</div>
      <div className="montage-settings"><button disabled={!attempts.length} onClick={()=>{if(visibleTime>=max){setTime(0);setPlaying(true);}else setPlaying(v=>!v);}}>{playing&&visibleTime<max?"Pause montage":"Play all attempts"}</button><button onClick={resetPlayback}>Restart montage</button><label><input type="checkbox" checked={normalized} onChange={e=>{setNormalized(e.target.checked);resetPlayback();}}/> Synchronize by progress</label><span>{normalized?`${Math.round(visibleTime*100)}%`:`${visibleTime.toFixed(2)} / ${max.toFixed(2)} s`}</span></div>
      <input aria-label="Montage time" type="range" min={0} max={max||1} step={normalized?.001:.01} value={visibleTime} onChange={e=>{setPlaying(false);setTime(Number(e.target.value));}}/>
      <p className="hint"><span className="success-key">Green: success</span> · <span className="failure-key">Coral: failure</span>. Full paths stay visible together; moving bodies share the selected clock and freeze at their endpoints. Opacity 100% is opaque. Alignment shifts each start reference to the origin and rotates its initial heading to +X; turn it off for original world coordinates. All attempts are retained, with at most 601 display poses per attempt.</p>
    </section>
  </main>;
}


