import { readFileSync } from "node:fs";
import { inflateRawSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { crc32, readNpy, readReplay, unzip } from "../../src/replay/npz";
import { sample } from "../../src/replay/timeline";

const inflate = async (bytes: Uint8Array) => new Uint8Array(inflateRawSync(bytes));
const reference = new Uint8Array(readFileSync(new URL("../../public/examples/reference.npz", import.meta.url)));
const failure = new Uint8Array(readFileSync(new URL("../../public/examples/failure.npz", import.meta.url)));

function storedZip(entries: Map<string, Uint8Array>): Uint8Array {
  const locals: Uint8Array[] = [], directories: Uint8Array[] = [];
  let offset=0;
  for(const [name,data] of entries) {
    const nameBytes = new TextEncoder().encode(name);
    const local = new Uint8Array(30+nameBytes.length+data.length), lv=new DataView(local.buffer);
    lv.setUint32(0,0x04034b50,true); lv.setUint16(4,20,true);
    lv.setUint32(14,crc32(data),true); lv.setUint32(18,data.length,true);lv.setUint32(22,data.length,true);lv.setUint16(26,nameBytes.length,true);
    local.set(nameBytes,30);local.set(data,30+nameBytes.length);
    const directory=new Uint8Array(46+nameBytes.length),dv=new DataView(directory.buffer);
    dv.setUint32(0,0x02014b50,true);dv.setUint32(16,crc32(data),true);dv.setUint32(20,data.length,true);dv.setUint32(24,data.length,true);dv.setUint16(28,nameBytes.length,true);dv.setUint32(42,offset,true);directory.set(nameBytes,46);
    locals.push(local);directories.push(directory);offset+=local.length;
  }
  const size=directories.reduce((sum,d)=>sum+d.length,0),end=new Uint8Array(22),ev=new DataView(end.buffer);
  ev.setUint32(0,0x06054b50,true);ev.setUint16(8,entries.size,true);ev.setUint16(10,entries.size,true);ev.setUint32(12,size,true);ev.setUint32(16,offset,true);
  return new Uint8Array(Buffer.concat([...locals,...directories,end]));
}

describe("Notebook replay compatibility",()=>{
  it("uses native streaming decompression",async()=>{
    expect((await readReplay(failure)).outcome).toBe("hover_departure");
  });
  it("reads the real NumPy reference archive and ordered passes",async()=>{
    const replay=await readReplay(reference,inflate);
    expect(replay.course.gates).toHaveLength(3);
    expect(replay.outcome).toBe("success");
    expect(replay.events.filter(e=>e.type==="gate_pass").map(e=>e.label)).toEqual([1,2,3]);
    expect(replay.duration).toBeCloseTo(7.808265403726055,10);
    expect(replay.count).toBe(938);
    const start=sample(replay,-100),end=sample(replay,100);
    expect(start).toEqual(Array.from(replay.states.slice(0,18)));
    expect(end[0]).toBe(replay.duration);
    const mid=sample(replay,.1234);
    expect(mid[0]).toBeCloseTo(.1234,10);
    expect(Math.hypot(...mid.slice(4,8))).toBeCloseTo(1,12);
  });
  it("reads an actual failed stochastic training attempt",async()=>{
    const replay=await readReplay(failure,inflate);
    expect(replay.outcome).toBe("hover_departure");
    expect(replay.course.gates).toHaveLength(0);
    expect(replay.duration).toBeLessThan(5);
  });
  it("reads uncompressed entries too",async()=>{
    const entries=await unzip(failure,inflate);
    expect((await readReplay(storedZip(entries))).outcome).toBe("hover_departure");
  });
  it("rejects an invalid timestamp despite a valid ZIP checksum",async()=>{
    const entries=await unzip(failure,inflate);
    const states=entries.get("states.npy")!.slice();
    const v=new DataView(states.buffer),start=10+v.getUint16(8,true);
    v.setFloat64(start+18*8,0,true);entries.set("states.npy",states);
    await expect(readReplay(storedZip(entries))).rejects.toThrow("timestamps");
  });
  it("rejects corrupt payloads",async()=>{
    const zipped=storedZip(await unzip(failure,inflate));zipped[100]=zipped[100]!^1;
    await expect(readReplay(zipped)).rejects.toThrow("checksum");
  });
  it("rejects missing arrays and unrelated files",async()=>{
    await expect(readReplay(storedZip(new Map([["something.npy",new Uint8Array()]])))).rejects.toThrow("notebook training trace");
    await expect(readReplay(new Uint8Array(25))).rejects.toThrow("directory");
    expect(()=>readNpy(new Uint8Array(20))).toThrow("NPY");
  });
  it("rejects duplicate filenames and oversized files",async()=>{
    await expect(readReplay(new Uint8Array(32*1024*1024+1))).rejects.toThrow("32 MiB");
    const archive=storedZip(new Map([["same.npy",new Uint8Array()],["else.npy",new Uint8Array()]]));
    const text=new TextDecoder("latin1").decode(archive),at=text.lastIndexOf("else.npy");
    archive.set(new TextEncoder().encode("same.npy"),at);
    await expect(unzip(archive)).rejects.toThrow("duplicate");
  });
});
