import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {activePolicy} from '../../../../simulation/active-policy';
export const dynamic='force-dynamic';
export async function GET(request:Request){
  try{
    const {value,directory}=await activePolicy();
    if(new URL(request.url).searchParams.get('version')!==value.sha256)return new Response('Policy changed; start again',{status:409});
    const data=await readFile(path.join(directory,value.file));
    if(createHash('sha256').update(data).digest('hex')!==value.sha256)throw new Error('Policy checksum mismatch');
    return new Response(new Uint8Array(data),{headers:{'Content-Type':'application/octet-stream','Cache-Control':'no-store'}});
  }catch{return new Response('Policy unavailable',{status:503});}
}
