import { Quaternion } from "three";
import type { Replay } from "./npz";

export function sample(replay: Replay, time: number): number[] {
  const t = Math.max(0, Math.min(time, replay.duration));
  let lo = 0, hi = replay.count-1;
  while (lo < hi) { const mid = Math.ceil((lo+hi)/2); if (replay.states[mid*18]! <= t) lo = mid; else hi = mid-1; }
  const next = Math.min(lo+1, replay.count-1), a = lo*18, b = next*18;
  const fraction = a === b ? 0 : (t-replay.states[a]!)/(replay.states[b]!-replay.states[a]!);
  const row = Array.from({ length: 18 }, (_, k) => replay.states[a+k]! + fraction*(replay.states[b+k]!-replay.states[a+k]!));
  const qa = new Quaternion(replay.states[a+5], replay.states[a+6], replay.states[a+7], replay.states[a+4]);
  const qb = new Quaternion(replay.states[b+5], replay.states[b+6], replay.states[b+7], replay.states[b+4]);
  qa.slerp(qb, fraction).normalize();
  row.splice(4, 4, qa.w, qa.x, qa.y, qa.z);
  return row;
}
