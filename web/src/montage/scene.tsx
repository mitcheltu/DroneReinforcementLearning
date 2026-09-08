"use client";
import {Canvas,useFrame,useThree} from "@react-three/fiber";
import {Line,OrbitControls} from "@react-three/drei";
import {useEffect,useMemo,useRef} from "react";
import {Box3,Color,InstancedMesh,Object3D,Quaternion,Vector3} from "three";
import {alignAttempt,poseAt,type Attempt} from "./data";

function Ghosts({attempts,time,normalized,opacity}:{attempts:Attempt[];time:number;normalized:boolean;opacity:number}){
  const mesh=useRef<InstancedMesh>(null!);const dummy=useMemo(()=>new Object3D(),[]);
  useFrame(()=>{if(!mesh.current)return;attempts.forEach((attempt,i)=>{
    const {a,b,fraction}=poseAt(attempt.samples,normalized?time*attempt.duration:time);
    dummy.position.set(...([1,2,3].map(k=>a[k]!+(b[k]!-a[k]!)*fraction) as [number,number,number]));
    dummy.quaternion.set(a[5]!,a[6]!,a[7]!,a[4]!);dummy.quaternion.slerp(new Quaternion(b[5],b[6],b[7],b[4]),fraction);
    dummy.updateMatrix();mesh.current.setMatrixAt(i,dummy.matrix);
    mesh.current.setColorAt(i,new Color(attempt.outcome==="success"?"#59dcc0":"#ff8669"));
  });mesh.current.instanceMatrix.needsUpdate=true;if(mesh.current.instanceColor)mesh.current.instanceColor.needsUpdate=true;});
  return <instancedMesh key={attempts.length} ref={mesh} args={[undefined,undefined,attempts.length]} frustumCulled={false}><boxGeometry args={[.48,.25,.12]}/><meshBasicMaterial transparent opacity={opacity} depthWrite={false}/></instancedMesh>;
}
function View({attempts,reset}:{attempts:Attempt[];reset:number}){
  const {camera}=useThree();const framing=useMemo(()=>{
    const box=new Box3();attempts.forEach(a=>a.samples.forEach(p=>box.expandByPoint(new Vector3(p[1],p[2],p[3]))));
    const center=box.isEmpty()?new Vector3():box.getCenter(new Vector3());const extent=box.isEmpty()?8:Math.max(5,box.getSize(new Vector3()).length()*.65);return {center,extent};
  },[attempts]);
  useEffect(()=>{camera.up.set(0,0,1);camera.position.copy(framing.center).add(new Vector3(framing.extent*.8,-framing.extent*1.2,framing.extent*.8));camera.lookAt(framing.center);},[camera,framing,reset]);
  return <OrbitControls makeDefault target={framing.center} maxDistance={500}/>;
}
export default function MontageScene({attempts,aligned,opacity,time,normalized,showGates,reset}:{attempts:Attempt[];aligned:boolean;opacity:number;time:number;normalized:boolean;showGates:boolean;reset:number}){
  const prepared=useMemo(()=>attempts.map(a=>alignAttempt(a,aligned)),[attempts,aligned]);
  return <Canvas camera={{position:[10,-15,10],near:.02,far:1000}} dpr={[1,1.5]}>
    <color attach="background" args={["#101925"]}/><gridHelper args={[120,60,"#41566a","#263441"]} rotation={[Math.PI/2,0,0]}/><axesHelper args={[3]}/>
    {prepared.map(a=><Line key={a.id} points={a.samples.map(p=>[p[1]!,p[2]!,p[3]!] as [number,number,number])} color={a.outcome==="success"?"#59dcc0":"#ff8669"} lineWidth={1.5} transparent opacity={opacity} depthWrite={false}/>)}
    {showGates&&prepared.flatMap(a=>a.course.gates.map(g=>{const w=g.width_m/2,h=g.height_m/2,s=Math.sin(g.yaw_rad),c=Math.cos(g.yaw_rad);return <Line key={`${a.id}-${g.id}`} points={[[-w,-h],[w,-h],[w,h],[-w,h],[-w,-h]].map(([u,v])=>[g.center_m[0]-s*u!,g.center_m[1]+c*u!,g.center_m[2]+v!] as [number,number,number])} color="#8ca5bf" transparent opacity={Math.min(opacity,.5)} depthWrite={false}/>;}))}
    {!!prepared.length&&<Ghosts attempts={prepared} time={time} normalized={normalized} opacity={opacity}/>}
    <View attempts={prepared} reset={reset}/>
  </Canvas>;
}
