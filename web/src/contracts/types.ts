import type vehicle from "../../../shared/vehicle.v1.json";
import type observation from "../../../shared/observation.v1.json";
import type rules from "../../../shared/course-rules.v1.json";
import type training from "../../../shared/training.v1.json";

export type VehicleConfig = typeof vehicle;
export type ObservationConfig = typeof observation;
export type CourseRules = typeof rules;
export type TrainingConfig = typeof training;
export type Vector3 = [number, number, number];
export type Vector4 = [number, number, number, number];

export interface StateV1 {
  position_m: Vector3;
  quaternion_wxyz: Vector4;
  velocity_world_mps: Vector3;
  omega_body_radps: Vector3;
  motor_thrust_n: Vector4;
}

export interface CourseV1 {
  schema_version: 1;
  course_id: string;
  name: string;
  mode: "standard" | "curriculum" | "experimental";
  gates: {
    id: string; label: number; center_m: Vector3; yaw_rad: number;
    width_m: 2.5; height_m: 2.5; frame_bar_m: 0.1; frame_depth_m: 0.1;
  }[];
  initial_state: StateV1;
  start_reference_m: Vector3;
  generator: null | {
    version: string; seed: string; reset_seed: string; stage: number; attempt_count: number;
  };
  rules_sha256: string;
}

export interface ReplayManifest {
  transition_start: number;
  transition_end: number;
  policy_revision_start: number;
  policy_revision_end: number;
  files: Record<string, {
    byte_length: number; sha256: string; shape: number[];
    dtype: "json" | "float64" | "float32" | "uint32";
  }>;
}

export interface ConfigTypes {
  vehicle: VehicleConfig;
  observation: ObservationConfig;
  "course-rules": CourseRules;
  training: TrainingConfig;
}
