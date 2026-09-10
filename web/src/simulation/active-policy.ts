import {readFile} from 'node:fs/promises';
import path from 'node:path';

export async function activePolicy(){
  const directory=path.join(process.cwd(),'public','models');
  const value=JSON.parse(await readFile(path.join(directory,'active-policy.json'),'utf8')) as {
    file:string;sha256:string;input_dimension:number;observation_contract:string;
  };
  if(!/^[a-zA-Z0-9-]+\.onnx$/.test(value.file)||!/^[a-f0-9]{64}$/.test(value.sha256)||
    !((value.input_dimension===74&&value.observation_contract==='recovery-workspace-v8')||
      (value.input_dimension===46&&value.observation_contract==='agile-previous-gate-v2')))
    throw new Error('Invalid active policy manifest');
  return {value,directory};
}
