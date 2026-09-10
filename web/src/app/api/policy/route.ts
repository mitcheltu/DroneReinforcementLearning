import {activePolicy} from '../../../simulation/active-policy';
export const dynamic='force-dynamic';
export async function GET(){
  try{const {value}=await activePolicy();return Response.json(value,{headers:{'Cache-Control':'no-store'}});}
  catch{return Response.json({error:'Policy unavailable'},{status:503});}
}
