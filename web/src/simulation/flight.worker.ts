import * as ort from "onnxruntime-web/wasm";
import type { CourseV1 } from "../contracts/types";
import { FlightEnvironment } from "./environment";

let flight:FlightEnvironment|null=null,session:ort.InferenceSession|null=null;
let active=false,generation=0;
const sleep=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));
self.onmessage=async(event:MessageEvent<{type:string;course?:CourseV1;fast?:boolean}>)=>{
  if(event.data.type!=="start"){active=false;generation++;return;}
  const token=++generation;active=true;
  try{
    if(!event.data.course)throw new Error("No course selected");
    self.postMessage({type:"loading"});
    if(!session){
      ort.env.wasm.numThreads=1;
      ort.env.wasm.wasmPaths=`${self.location.origin}/onnx/`;
      session=await ort.InferenceSession.create(`${self.location.origin}/models/actor.onnx`,{executionProviders:["wasm"]});
    }
    if(token!==generation)return;
    flight=new FlightEnvironment(event.data.course);
    self.postMessage({type:"frame",rows:[[0,...flight.state]],elapsed:0,reward:0,events:[],outcome:"running"});
    let rows:number[][]=[],steps=0;
    const start=performance.now();
    while(active&&token===generation&&flight.outcome==="running"){
      const output=await session.run({observation:new ort.Tensor("float32",flight.observation(),[1,40])});
      if(token!==generation)return;
      const action=Array.from(output.action!.data as Float32Array);
      rows.push(...flight.step(action));steps++;
      if(steps%4===0||flight.outcome!=="running"){
        self.postMessage({type:"frame",rows,elapsed:flight.elapsed,reward:flight.reward,events:flight.events,outcome:flight.outcome});rows=[];
        if(!event.data.fast)await sleep(Math.max(0,flight.elapsed*1000-(performance.now()-start)));
        else await sleep(0);
      }
    }
    if(token===generation){active=false;self.postMessage({type:"done"});}
  }catch(error){if(token===generation){active=false;self.postMessage({type:"error",message:error instanceof Error?error.message:String(error)});}}
};
