import * as ort from "onnxruntime-web/wasm";
import type { CourseV1 } from "../contracts/types";
import { FlightEnvironment } from "./environment";

let flight:FlightEnvironment|null=null,session:ort.InferenceSession|null=null;
let active=false,generation=0;
let sessionModel="";
const sleep=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));
const basePath=process.env.NEXT_PUBLIC_BASE_PATH??"";
const staticDeployment=process.env.NEXT_PUBLIC_STATIC_DEPLOYMENT==="true";
self.onmessage=async(event:MessageEvent<{type:string;course?:CourseV1;fast?:boolean;policy?:string}>)=>{
  if(event.data.type!=="start"){active=false;generation++;return;}
  const token=++generation;active=true;
  try{
    if(!event.data.course)throw new Error("No course selected");
    const experimental=event.data.policy==="obstacles-005";
    const managed=event.data.policy==="active";
    let recovery=false;
    let activeVersion="";
    if(managed){
      const response=await fetch(staticDeployment?`${basePath}/models/active-policy.json`:`${self.location.origin}/api/policy`,{cache:"no-store"});
      if(!response.ok)throw new Error("The active policy could not be loaded");
      const manifest=await response.json();
      recovery=manifest.observation_contract==="recovery-workspace-v8"&&manifest.input_dimension===74;
      if(!recovery&&!(manifest.observation_contract==="agile-previous-gate-v2"&&manifest.input_dimension===46))throw new Error("Unsupported policy observation contract");
      activeVersion=manifest.sha256;
    }
    const maneuver=managed||event.data.policy==="maneuver-006";
    const model=managed&&!staticDeployment?`/api/policy/model?version=${activeVersion}`:`${basePath}/models/${managed?JSON.parse(await (await fetch(`${basePath}/models/active-policy.json`)).text()).file:maneuver?"maneuver-006-round-3.onnx":experimental?"obstacles-005-round-3.onnx":"actor.onnx"}`;
    self.postMessage({type:"loading"});
    if(!session||sessionModel!==model){
      ort.env.wasm.numThreads=1;
      ort.env.wasm.wasmPaths=`${basePath}/onnx/`;
      const loaded=await ort.InferenceSession.create(staticDeployment?model:`${self.location.origin}${model}`,{executionProviders:["wasm"]});
      if(token!==generation){await loaded.release();return;}
      if(session)await session.release();
      session=loaded;sessionModel=model;
    }
    if(token!==generation)return;
    flight=new FlightEnvironment(event.data.course,experimental||maneuver?"agile":"original",false);
    self.postMessage({type:"frame",rows:[[0,...flight.state]],action:[0,0,0,0],elapsed:0,reward:0,events:[],outcome:"running"});
    let rows:number[][]=[],steps=0;
    let latestAction=[0,0,0,0];
    const start=performance.now();
    while(active&&token===generation&&flight.outcome==="running"){
      const observation=recovery?flight.recoveryObservation():maneuver?flight.agileObservation():experimental?flight.obstacleObservation():flight.observation();
      const output=await session.run({observation:new ort.Tensor("float32",observation,[1,observation.length])});
      if(token!==generation)return;
      const action=Array.from(output.action!.data as Float32Array);
      latestAction=action;
      rows.push(...flight.step(action));steps++;
      if(steps%4===0||flight.outcome!=="running"){
        self.postMessage({type:"frame",rows,action:latestAction,elapsed:flight.elapsed,reward:flight.reward,events:flight.events,outcome:flight.outcome});rows=[];
        if(!event.data.fast)await sleep(Math.max(0,flight.elapsed*1000-(performance.now()-start)));
        else await sleep(0);
      }
    }
    if(token===generation){active=false;self.postMessage({type:"done"});}
  }catch(error){if(token===generation){active=false;self.postMessage({type:"error",message:error instanceof Error?error.message:String(error)});}}
};
