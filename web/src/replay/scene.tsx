"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Line, OrbitControls, Html, TransformControls } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import { Group, Object3D, Vector3 } from "three";
import type { Replay } from "./npz";
import { sample } from "./timeline";
import type {CourseV1,Vector3 as Tuple3} from "../contracts/types";

export interface GateEditing {selected:number;mode:"translate"|"rotate";cameraCourse:CourseV1;onSelect:(index:number)=>void;onChange:(position:Tuple3,yaw:number)=>void;}

function Gate({gate,index,color,editor}:{gate:CourseV1["gates"][number];index:number;color:string;editor?:GateEditing}){
  const group=useRef<Group>(null!);const selected=editor?.selected===index;
  return <><group ref={group} position={gate.center_m} rotation={[0,0,gate.yaw_rad]} onClick={e=>{if(editor){e.stopPropagation();editor.onSelect(index);}}}>
    {[-1,1].map(side=><group key={side}>
      <mesh position={[0,side*(gate.width_m/2+.05),0]}><boxGeometry args={[.1,.1,gate.height_m+.2]}/><meshStandardMaterial color={selected?"#ffd875":color}/></mesh>
      <mesh position={[0,0,side*(gate.height_m/2+.05)]}><boxGeometry args={[.1,gate.width_m,.1]}/><meshStandardMaterial color={selected?"#ffd875":color}/></mesh>
    </group>)}
    {editor&&<arrowHelper args={[new Vector3(1,0,0),new Vector3(0,0,0),2,selected?0xffd875:0x61cfe0,.35,.2]}/>}
    <Html position={[0,0,gate.height_m/2+.6]} center occlude style={{color:"#e4e9f0",fontSize:16,pointerEvents:"none"}}>{gate.label}</Html>
  </group>{selected&&editor&&<TransformControls object={group} mode={editor.mode} space="world" showX={editor.mode==="translate"} showY={editor.mode==="translate"} showZ size={.85} onObjectChange={()=>{const g=group.current;editor.onChange([g.position.x,g.position.y,g.position.z],g.rotation.z);}}/>}</>;
}

Object3D.DEFAULT_UP.set(0, 0, 1);

function Drone({ replay, time }: { replay: Replay; time: number }) {
  const group = useRef<Group>(null);
  useFrame(() => {
    const row = sample(replay, time);
    group.current?.position.set(row[1]!, row[2]!, row[3]!);
    group.current?.quaternion.set(row[5]!, row[6]!, row[7]!, row[4]!);
  });
  return <group ref={group}>
    <mesh><boxGeometry args={[.24,.16,.09]} /><meshStandardMaterial color="#ec6d38" /></mesh>
    {[Math.PI/4,-Math.PI/4].map(angle => <mesh key={angle} rotation={[0,0,angle]}><boxGeometry args={[.4,.035,.035]} /><meshStandardMaterial color="#d2d9e1" /></mesh>)}
    {[-1,1].flatMap(x => [-1,1].map(y => <mesh key={`${x}:${y}`} position={[x*.1414,y*.1414,.035]} rotation={[Math.PI/2,0,0]}><cylinderGeometry args={[.095,.095,.012,20]} /><meshStandardMaterial color="#a4adbc" transparent opacity={.7} /></mesh>))}
    <mesh position={[.16,0,.025]}><boxGeometry args={[.06,.055,.05]} /><meshStandardMaterial color="#ffffff" /></mesh>
  </group>;
}

function Camera({ replay, time, chase, reset, frameCourse }: { replay: Replay; time: number; chase: boolean; reset: number; frameCourse?:CourseV1 }) {
  const { camera } = useThree();
  const course=frameCourse??replay.course;
  const target = useMemo(() => {
    const points = [course.start_reference_m, ...course.gates.map(g => g.center_m)];
    return points.reduce((a, p) => a.addScaledVector(new Vector3(...p),1/points.length),new Vector3());
  }, [course]);
  useEffect(() => {
    const extent = Math.max(8, ...course.gates.map(g => new Vector3(...g.center_m).distanceTo(target)));
    camera.up.set(0,0,1); camera.position.copy(target).add(new Vector3(extent*.8,-extent*1.5,extent)); camera.lookAt(target);
  }, [camera, course, target, reset, chase]);
  useFrame((_, dt) => {
    if (!chase) return;
    const row = sample(replay,time);
    const w=row[4]!,x=row[5]!,y=row[6]!,z=row[7]!;
    const yaw = Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));
    const desired = new Vector3(row[1]!-5*Math.cos(yaw),row[2]!-5*Math.sin(yaw),row[3]!+2);
    camera.position.lerp(desired,1-Math.exp(-Math.min(dt,.1)/.15));
    camera.lookAt(row[1]!+2*Math.cos(yaw),row[2]!+2*Math.sin(yaw),row[3]!);
  });
  return <OrbitControls key={`${reset}:${chase}`} enabled={!chase} target={target} minDistance={1} maxDistance={200} makeDefault />;
}

export default function Scene({ replay, time, chase, reset, editor }: { replay: Replay; time: number; chase: boolean; reset: number; editor?:GateEditing }) {
  const path = useMemo(() => Array.from({length: replay.count},(_,i) => new Vector3(replay.states[i*18+1],replay.states[i*18+2],replay.states[i*18+3])),[replay]);
  const passed = replay.events.filter(e => e.type === "gate_pass" && e.time <= time).length;
  return <Canvas camera={{position:[15,-20,18],fov:48,near:.05,far:500}} dpr={[1,1.5]}>
    <color attach="background" args={["#101925"]} /><ambientLight intensity={1.4} /><directionalLight position={[10,-15,30]} intensity={2} />
    <gridHelper args={[120,60,"#3b4c5b","#263441"]} rotation={[Math.PI/2,0,0]} />
    <mesh position={[0,0,-.02]}><planeGeometry args={[120,120]} /><meshStandardMaterial color="#15202c" /></mesh>
    {replay.course.gates.map((g,i)=><Gate key={g.id} gate={g} index={i} editor={editor} color={i<passed?"#50bdaa":i===passed?"#ed8d4b":"#8393a4"}/>)}
    <Line points={path.length===1?[path[0]!,path[0]!]:path} color="#e8975e" lineWidth={1.5} transparent opacity={.5} />
    <Drone replay={replay} time={time} /><Camera replay={replay} time={time} chase={chase} reset={reset} frameCourse={editor?.cameraCourse} />
  </Canvas>;
}
