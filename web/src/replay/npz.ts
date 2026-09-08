import { parseJson, validateArtifact } from "../contracts/loader";
import type { CourseV1 } from "../contracts/types";

export interface Replay {
  course: CourseV1;
  states: Float64Array;
  count: number;
  duration: number;
  outcome: string;
  stage: number;
  reward: number;
  events: { time: number; type: string; label?: number }[];
}

const LIMIT = 32 * 1024 * 1024;
function requireValue(value: unknown, message: string): asserts value {
  if (!value) throw new Error(message);
}

export function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

async function inflate(bytes: Uint8Array, expected: number): Promise<Uint8Array> {
  const stream = new Blob([Uint8Array.from(bytes)]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  const reader = stream.getReader();
  const output = new Uint8Array(expected);
  let offset = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      requireValue(offset + value.length <= expected, "Decompressed entry exceeds its declared size.");
      output.set(value, offset); offset += value.length;
    }
  } catch (error) { await reader.cancel(); throw error; }
  requireValue(offset === expected, "Truncated compressed entry.");
  return output;
}

export async function unzip(bytes: Uint8Array, decompress = inflate): Promise<Map<string, Uint8Array>> {
  requireValue(bytes.length <= LIMIT && bytes.length >= 22, "Replay must be a ZIP smaller than 32 MiB.");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let end = bytes.length - 22;
  while (end >= Math.max(0, bytes.length - 65557) && view.getUint32(end, true) !== 0x06054b50) end--;
  requireValue(end >= 0 && view.getUint32(end, true) === 0x06054b50, "ZIP directory is missing.");
  requireValue(end + 22 + view.getUint16(end + 20, true) === bytes.length, "Invalid ZIP trailer.");
  requireValue(view.getUint16(end+4, true) === 0 && view.getUint16(end+6, true) === 0, "Multipart ZIP is unsupported.");
  const count = view.getUint16(end+10, true);
  let offset = view.getUint32(end+16, true);
  const directoryEnd = offset + view.getUint32(end+12, true);
  requireValue(count > 0 && count <= 24 && directoryEnd === end, "Invalid ZIP directory size.");
  const entries = new Map<string, Uint8Array>();
  let total = 0;
  for (let i = 0; i < count; i++) {
    requireValue(offset + 46 <= directoryEnd && view.getUint32(offset, true) === 0x02014b50, "Invalid ZIP entry.");
    const flags = view.getUint16(offset+8, true), method = view.getUint16(offset+10, true);
    const checksum = view.getUint32(offset+16, true), compressed = view.getUint32(offset+20, true);
    const size = view.getUint32(offset+24, true), nameSize = view.getUint16(offset+28, true);
    const extra = view.getUint16(offset+30, true), comment = view.getUint16(offset+32, true);
    const local = view.getUint32(offset+42, true);
    total += size;
    requireValue(!(flags & 1) && (method === 0 || method === 8) && total <= LIMIT, "Unsupported or oversized ZIP entry.");
    requireValue(offset+46+nameSize+extra+comment <= directoryEnd, "Truncated ZIP filename.");
    const name = new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(offset+46, offset+46+nameSize));
    requireValue(/^[a-z_]+\.npy$/.test(name) && !entries.has(name), "Unexpected or duplicate replay entry.");
    requireValue(local+30 <= end && view.getUint32(local, true) === 0x04034b50, "Invalid local ZIP header.");
    const start = local+30+view.getUint16(local+26, true)+view.getUint16(local+28, true);
    requireValue(start+compressed <= view.getUint32(end+16, true), "ZIP entry overlaps its directory.");
    const payload = bytes.subarray(start, start+compressed);
    const decoded = method === 0 ? payload : await decompress(payload, size);
    requireValue(decoded.length === size && crc32(decoded) === checksum, "Replay checksum failed.");
    entries.set(name, decoded);
    offset += 46+nameSize+extra+comment;
  }
  requireValue(offset === directoryEnd, "ZIP entry count mismatch.");
  return entries;
}

export function readNpy(bytes: Uint8Array): { shape: number[]; values?: Float64Array; text?: string } {
  requireValue(bytes.length >= 10 && bytes[0] === 0x93 && new TextDecoder().decode(bytes.subarray(1, 6)) === "NUMPY", "Invalid NPY array.");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = bytes[6];
  requireValue(version === 1 || version === 2, "Unsupported NPY version.");
  const prefix = version === 1 ? 10 : 12;
  requireValue(bytes.length >= prefix, "Truncated NPY header.");
  const length = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const start = prefix + length;
  requireValue(length <= 16384 && start <= bytes.length, "Invalid NPY header size.");
  const header = new TextDecoder().decode(bytes.subarray(prefix, start));
  const dtype = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const dimensions = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  requireValue(dtype && dimensions !== undefined && /'fortran_order':\s*False/.test(header), "Unsupported NPY layout.");
  requireValue(/^[\d,\s]*$/.test(dimensions), "Invalid array dimensions.");
  const shape = dimensions.split(",").map(s => s.trim()).filter(Boolean).map(Number);
  const count = shape.reduce((a, b) => a*b, 1);
  requireValue(shape.length <= 2 && Number.isSafeInteger(count) && count <= LIMIT/4, "Array shape is too large.");
  if (dtype.startsWith("<U")) {
    const chars = Number(dtype.slice(2));
    requireValue(shape.length === 0 && Number.isSafeInteger(chars) && chars <= 1000000 && start+chars*4 === bytes.length, "Invalid metadata string.");
    let text = "";
    for (let i = 0; i < chars; i++) { const cp = view.getUint32(start+i*4, true); requireValue(cp <= 0x10ffff, "Invalid Unicode metadata."); text += String.fromCodePoint(cp); }
    return { shape, text };
  }
  requireValue(dtype === "<f8" || dtype === "<f4", "Expected a little-endian floating-point array.");
  const stride = dtype === "<f8" ? 8 : 4;
  requireValue(start+count*stride === bytes.length, "Array byte length does not match its shape.");
  const values = new Float64Array(count);
  for (let i = 0; i < count; i++) { const x = stride === 8 ? view.getFloat64(start+i*stride, true) : view.getFloat32(start+i*stride, true); requireValue(Number.isFinite(x), "Replay contains nonfinite state."); values[i] = x; }
  return { shape, values };
}

export async function readReplay(bytes: Uint8Array, decompress = inflate): Promise<Replay> {
  const entries = await unzip(bytes, decompress);
  const stateFile = entries.get("states.npy"), metaFile = entries.get("metadata_json.npy");
  requireValue(stateFile && metaFile, "Select a notebook training trace (.npz), not a checkpoint or run ZIP.");
  const array = readNpy(stateFile), metadata = readNpy(metaFile).text;
  requireValue(array.values && array.shape.length === 2 && array.shape[1] === 18 && array.shape[0]! >= 2 && array.shape[0]! <= 72002 && metadata, "Invalid trajectory dimensions.");
  const meta = parseJson(new TextEncoder().encode(metadata)) as Record<string, unknown>;
  requireValue(meta && meta.format === "aerorl-notebook-trace-v1", "Unsupported replay format.");
  const course = validateArtifact<CourseV1>("editable-course", meta.course);
  const maximumDuration=course.mode==="experimental"?600:45;
  requireValue(array.shape[0]!<=maximumDuration*120+2,"Invalid trajectory dimensions.");
  const metrics = meta.metrics as Record<string, unknown>;
  requireValue(metrics && typeof metrics.outcome === "string" && Number.isInteger(metrics.stage) && typeof metrics.episode === "object" && metrics.episode, "Replay metrics are missing.");
  const episode = metrics.episode as Record<string, unknown>;
  requireValue(typeof episode.r === "number" && Number.isFinite(episode.r), "Invalid episode reward.");
  requireValue(Array.isArray(meta.events) && meta.events.length <= 10000, "Invalid replay events.");
  const states = array.values, count = array.shape[0]!;
  const duration = states[(count-1)*18]!;
  requireValue(states[0] === 0 && duration > 0 && duration <= maximumDuration+.001, "Invalid replay duration.");
  for (let i = 0; i < count; i++) {
    const row = i*18;
    requireValue(i === 0 || states[row]! > states[row-18]!, "Replay timestamps must increase.");
    const norm = Math.hypot(states[row+4]!, states[row+5]!, states[row+6]!, states[row+7]!);
    requireValue(Math.abs(norm-1) < 1e-6, "Invalid drone orientation.");
    requireValue(Math.max(...states.subarray(row+1, row+4).map(Math.abs)) < 1000, "Position exceeds replay bounds.");
  }
  const events = meta.events.map((event: unknown) => {
    const e = event as Record<string, unknown>;
    requireValue(e && typeof e.time === "number" && e.time >= 0 && e.time <= duration+1e-8 && typeof e.type === "string", "Invalid replay event.");
    requireValue(e.label === undefined || (Number.isInteger(e.label) && Number(e.label) >= 1 && Number(e.label) <= course.gates.length), "Invalid gate label.");
    return { time: e.time, type: e.type, label: e.label as number | undefined };
  });
  let lastEvent = -1, lastGate = 0;
  for (const event of events) {
    requireValue(event.time >= lastEvent, "Replay events are not ordered by time.");
    lastEvent = event.time;
    if (event.type === "gate_pass") {
      requireValue(event.label === lastGate+1, "Replay gate passes are out of order.");
      lastGate++;
    }
  }
  return { course, states, count, duration, outcome: metrics.outcome, stage: Number(metrics.stage), reward: episode.r, events };
}
