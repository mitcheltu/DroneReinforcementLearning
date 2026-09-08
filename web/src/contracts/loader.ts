import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { visit } from "jsonc-parser";
import { schemas, type SchemaName } from "./schemas";
import type {
  ConfigTypes, CourseRules, CourseV1, ReplayManifest, StateV1, TrainingConfig, VehicleConfig,
} from "./types";

export const CONFIG_NAMES = ["vehicle", "observation", "course-rules", "training"] as const;
export const MAX_JSON_BYTES = 16 * 1024 * 1024;

export class ContractError extends Error {
  constructor(public readonly code: string, message: string) { super(message); }
}

function requireContract(condition: boolean, code: string, message: string): asserts condition {
  if (!condition) throw new ContractError(code, message);
}

function checkNumbers(value: unknown, depth = 0): void {
  requireContract(depth <= 64, "depth", "JSON nesting exceeds 64 levels");
  if (typeof value === "number") {
    requireContract(Number.isFinite(value), "nonfinite", "Nonfinite JSON number");
    requireContract(!Number.isInteger(value) || Number.isSafeInteger(value),
      "unsafe_integer", "Large integers must be decimal strings");
  } else if (value !== null && typeof value === "object") {
    for (const item of Object.values(value)) checkNumbers(item, depth + 1);
  }
}

export function parseJson(raw: Uint8Array): unknown {
  requireContract(raw.byteLength <= MAX_JSON_BYTES, "size", "JSON artifact exceeds 16 MiB");
  let source: string;
  let value: unknown;
  try {
    source = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(raw);
    value = JSON.parse(source);
  } catch (error) {
    throw new ContractError("json", `Invalid UTF-8 JSON: ${String(error)}`);
  }
  const objects: Set<string>[] = [];
  checkNumbers(value);
  visit(source, {
    onObjectBegin: () => { objects.push(new Set()); },
    onObjectProperty: (property) => {
      const current = objects.at(-1)!;
      requireContract(!current.has(property), "duplicate_key", `Duplicate JSON property: ${property}`);
      current.add(property);
    },
    onObjectEnd: () => { objects.pop(); },
  });
  checkNumbers(value);
  return value;
}

function stateSemantics(state: StateV1): void {
  const norm = Math.sqrt(state.quaternion_wxyz.reduce((sum, x) => sum + x * x, 0));
  requireContract(Math.abs(norm - 1) <= 1e-6, "quaternion", "Quaternion norm differs from one");
  state.quaternion_wxyz = state.quaternion_wxyz.map(x => x / norm) as StateV1["quaternion_wxyz"];
}

function semantics(name: SchemaName, value: unknown): void {
  if (name === "state") stateSemantics(value as StateV1);
  if (name === "course" || name === "editable-course") {
    const course = value as CourseV1;
    stateSemantics(course.initial_state);
    requireContract(new Set(course.gates.map(g => g.id)).size === course.gates.length,
      "duplicate_id", "Gate IDs repeat");
    requireContract(course.gates.every((g, i) => g.label === i + 1),
      "invalid_label", "Gate labels must follow array order");
    if (course.generator) {
      for (const field of ["seed", "reset_seed"] as const) {
        requireContract(BigInt(course.generator[field]) <= 18446744073709551615n,
          "seed", "Seed exceeds uint64");
      }
      if (course.mode === "curriculum") {
        requireContract(course.gates.length === [0, 1, 1, 3, 3, 10, 10][course.generator.stage],
          "gate_count", "Stage and gate count disagree");
      }
    }
  }
  if (name === "vehicle") {
    const vehicle = value as VehicleConfig;
    requireContract(vehicle.physics_hz === 2 * vehicle.policy_hz, "timing", "Expected two ticks/action");
    requireContract(Math.abs(vehicle.action.max_collective_n - 4 * vehicle.rotor_max_thrust_n) <= 1e-9,
      "collective", "Collective differs from rotor capacity");
    const b = vehicle.arm_length_m / Math.sqrt(2);
    const positions = [[b, b, 0], [-b, b, 0], [-b, -b, 0], [b, -b, 0]];
    vehicle.rotors.forEach((rotor, i) => {
      requireContract(rotor.index === i, "rotors", "Rotor indices are not ordered");
      requireContract(rotor.reaction_sign === [1, -1, 1, -1][i] &&
        rotor.position_m.every((x, j) => Math.abs(x - positions[i]![j]!) <= 1e-12),
      "rotors", "Rotor geometry/signs differ from X-frame convention");
    });
  }
  if (name === "course-rules") {
    const rules = value as CourseRules;
    requireContract(rules.workspace_min_m.every((x, i) => x < rules.workspace_max_m[i]!),
      "bounds", "Workspace minimum must precede maximum");
    for (const field of ["horizontal_spacing_m", "gate_center_height_m"] as const) {
      requireContract(rules[field][0]! > 0 && rules[field][0]! <= rules[field][1]!,
        "bounds", `Invalid ${field}`);
    }
  }
  if (name === "training") {
    const { ppo, curriculum } = value as TrainingConfig;
    requireContract(ppo.gamma > 0 && ppo.gamma <= 1, "gamma", "Gamma must be in (0,1]");
    for (const n of ppo.environment_candidates) {
      requireContract([1, 2, 4, 8].includes(n) && ppo.n_steps * n % ppo.batch_size === 0,
        "rollout", "Every rollout must divide into complete minibatches");
    }
    requireContract(Math.abs(curriculum.frontier_probability + curriculum.previous_probability - 1) <= 1e-9,
      "probability", "Curriculum probabilities must sum to one");
    requireContract(curriculum.stages.every((stage, i) => stage.index === i),
      "stages", "Curriculum stages must be ordered");
  }
  if (name === "replay") {
    const replay = value as ReplayManifest;
    requireContract(replay.transition_end >= replay.transition_start, "order", "Reversed transitions");
    requireContract(replay.policy_revision_end >= replay.policy_revision_start, "order", "Reversed revisions");
    let total = 0;
    for (const [filename, entry] of Object.entries(replay.files)) {
      total += entry.byte_length;
      if (entry.dtype !== "json") {
        const width = { float64: 8, float32: 4, uint32: 4 }[entry.dtype];
        requireContract(entry.shape.reduce((a, b) => a * b, 1) * width === entry.byte_length,
          "byte_length", `Inconsistent array size for ${filename}`);
      }
    }
    requireContract(total <= 64 * 1024 * 1024, "size", "Replay exceeds uncompressed size budget");
  }
  if (name === "worker-message") {
    const message = value as Record<string, unknown>;
    if (message.type === "LOAD_COURSE") semantics("course", message.course);
    if (message.type === "SNAPSHOT") {
      stateSemantics(message.state as StateV1);
      stateSemantics(message.previous_state as StateV1);
    }
  }
}

// Only registered schemas are available; remote schema retrieval is never enabled.
const ajv = new Ajv2020({ allErrors: true, strict: true, strictRequired: false, strictTuples: false });
addFormats(ajv);
for (const schema of Object.values(schemas)) ajv.addSchema(schema);

export function validateArtifact<T = unknown>(name: SchemaName, value: unknown): T {
  const schema = schemas[name];
  requireContract(Boolean(schema), "schema", `Unknown schema: ${name}`);
  checkNumbers(value);
  const validate = ajv.getSchema(schema.$id)!;
  if (!validate(value)) throw new ContractError("schema", `${name}: ${ajv.errorsText(validate.errors)}`);
  const result: unknown = structuredClone(value);
  if (name === "config") {
    const kind = CONFIG_NAMES.find(k => ajv.getSchema(schemas[k].$id)!(result));
    if (kind) semantics(kind, result);
  } else semantics(name, result);
  return result as T;
}

export interface LoadedArtifact<T = unknown> { data: T; sha256: string; byte_length: number }

export async function loadBytes<T = unknown>(
  name: SchemaName, raw: Uint8Array, expectedSha256?: string,
): Promise<LoadedArtifact<T>> {
  requireContract(raw.byteLength <= MAX_JSON_BYTES, "size", "JSON artifact exceeds 16 MiB");
  const bytes = new Uint8Array(raw);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const sha256 = [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, "0")).join("");
  requireContract(expectedSha256 === undefined || sha256 === expectedSha256,
    "hash", "Artifact SHA-256 does not match expected bytes");
  return { data: validateArtifact<T>(name, parseJson(raw)), sha256, byte_length: raw.byteLength };
}

export async function loadConfig<K extends keyof ConfigTypes>(
  name: K, raw: Uint8Array, expectedSha256?: string,
): Promise<LoadedArtifact<ConfigTypes[K]>> {
  return loadBytes<ConfigTypes[K]>(name, raw, expectedSha256);
}

export type ConfigBundle = { [K in keyof ConfigTypes]: LoadedArtifact<ConfigTypes[K]> };

export function validateConfigBundle(bundle: ConfigBundle): void {
  const vehicle = bundle.vehicle.data;
  const observation = bundle.observation.data;
  const rules = bundle["course-rules"].data;
  requireContract(observation.fields.slice(26, 30).every(f => f.scale === vehicle.rotor_max_thrust_n),
    "compatibility", "Motor observation scaling differs from vehicle capacity");
  const clearance = vehicle.collision_radius_m + vehicle.gate_pass_margin_m;
  requireContract(Math.min(rules.gate.width_m, rules.gate.height_m) / 2 > clearance,
    "compatibility", "Drone cannot fit through the gate's pass aperture");
  for (const timeout of Object.values(rules.timeouts_s)) {
    const ticks = timeout * vehicle.physics_hz;
    requireContract(ticks > 0 && ticks % 2 === 0, "compatibility", "Timeout must end on a policy tick");
  }
}

export function verifyManifestConfigs(manifest: Record<string, unknown>, bundle: ConfigBundle): void {
  const fields = {
    vehicle_sha256: "vehicle", observation_sha256: "observation",
    rules_sha256: "course-rules", training_config_sha256: "training",
  } as const;
  for (const [field, name] of Object.entries(fields)) {
    requireContract(manifest[field] === bundle[name].sha256, "compatibility", `Mismatch: ${field}`);
  }
}

