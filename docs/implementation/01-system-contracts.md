# 01 — System contracts

## 1. Components and data flow

Training: shared JSON configuration -> Python simulator -> Gymnasium environment -> SB3 PPO -> checkpoint/evaluation -> ONNX actor + manifest.

Browser flight: validated course + initial state -> observation -> worker ONNX inference -> action clamp/map -> rate controller/mixer -> RK4 physics -> events -> render snapshots.

Replay: validated replay bundle -> recorded state/event timeline -> render adapter + telemetry charts. Replay never invokes ONNX or integrates physics.

The browser has a UI/main thread and one dedicated module worker for inference and physics. The worker owns simulation state. The main thread owns course editing while idle/paused and sends immutable complete course snapshots. Worker messages include `protocol_version: 1`, `generation_id`, `request_id`, and a discriminated `type`. Unknown versions/types are rejected with structured errors.

Python owns procedural training generation. Browser version 1 does not implement a second random generator. It loads pre-generated benchmark courses and supports user edits. This avoids claiming shared seeds reproduce Python-generated courses in JavaScript.

## 2. Units and frames

| Quantity | Convention |
|---|---|
| World axes | +X, +Y horizontal, +Z up; right-handed |
| Body axes | +X forward, +Y left, +Z up; right-handed |
| Gate axes | local +X forward traversal normal, +Y across width, +Z up |
| Position / velocity | world frame, meters / m/s |
| Angular velocity | body frame, rad/s |
| Attitude | Hamilton quaternion, body-to-world rotation, `[w,x,y,z]` |
| Rotor force | positive along body +Z, newtons |
| Torque | body frame, N m, right-hand rule |
| Time | integer physics tick plus optional event fraction; seconds derived |

For a unit quaternion q, R(q) maps body vectors to world vectors. Body-frame velocity is R(q)^T v. Body gravity direction is R(q)^T [0,0,-1]. Relative gate center is R(q)^T (c-p). Gate normal is [cos(yaw),sin(yaw),0] in world coordinates. Gate width vector is [-sin(yaw),cos(yaw),0].

Quaternions must be normalized on input when norm error is <=1e-6; larger error is an invalid input. Zero norm and nonfinite values are always invalid. Integrator normalization follows document 02. For comparison, q and -q are equivalent; rotation error is `2*acos(clamp(abs(dot(q1,q2)),0,1))`.

All JSON numbers must be finite; units never change according to display preferences. Display degrees are converted at UI boundaries. JSON yaw is normalized to [-pi,pi).

## 3. Required shared files

| File | Required contents |
|---|---|
| `shared/vehicle.v1.json` | All constants in document 02, controller gain vector, rotor positions/spins, collision radius |
| `shared/observation.v1.json` | Ordered 40 fields, scales, clip bounds, masks and schema ID |
| `shared/course-rules.v1.json` | Workspace, gate geometry, support constraints and timeout |
| `shared/training.v1.json` | PPO, curriculum, rewards, evaluation and retention defaults |
| `shared/schemas/*.schema.json` | JSON Schema draft 2020-12 for each artifact and message manifest |
| `shared/fixtures/*` | Golden physics, observation, event and action cases |

Configuration parsing must fail on missing required keys or unknown keys. No fallback physics values are permitted after parsing. SHA-256 hashes refer to the exact stored UTF-8 bytes; do not reserialize before verification. A release manifest includes hashes for every dependent configuration.

## 4. Course schema

`CourseV1` fields:

| Field | Type / rule |
|---|---|
| `schema_version` | integer 1 |
| `course_id` | UUID string |
| `name` | 1–80 printable characters, rendered as text |
| `mode` | `standard`, `curriculum`, or `experimental` |
| `gates` | ordered list; 10 standard/experimental, 0/1/3/10 curriculum |
| `initial_state` | StateV1 below |
| `start_reference_m` | three numbers; unperturbed start position used for course constraints |
| `generator` | object: version, seed decimal string, reset_seed decimal string, stage integer, attempt_count integer; null after any committed editor change |
| `rules_sha256` | 64 lowercase hex characters |

Each gate has `id` (unique UUID), `label` (array index+1), `center_m` (three numbers), `yaw_rad` (number), `width_m:2.5`, `height_m:2.5`, `frame_bar_m:0.10`, `frame_depth_m:0.10`. No independent gate quaternion is stored: derive it from yaw. The UI may reorder gate IDs but must rewrite contiguous labels and validate the complete order. Adding/removing gates is disabled in standard/experimental editor version 1. Imported curriculum courses are replay/development assets, not editable standard courses.

Any committed editor change creates a new course UUID and sets generator=null. Preserve gate UUIDs through moves/reordering. Reset state follows document 03. The canonical geometry/initial-state comparison hash is SHA-256 of compact sorted-key JSON containing only ordered gate numeric geometry, start_reference_m and initial_state; exclude IDs, labels, name, mode and generator metadata. Use this hash for duplicate-partition checks and replay comparisons, never the whole-file hash containing arbitrary IDs.

StateV1 fields are `position_m[3]`, `quaternion_wxyz[4]`, `velocity_world_mps[3]`, `omega_body_radps[3]`, `motor_thrust_n[4]`. Additional episode state is `physics_tick`, `target_gate_index`, `previous_action[4]`, `elapsed_policy_steps`, `status`, and `episode_id`. States are not interchangeable with observations.

## 5. Observation: `obs-bodyrate-v1`

Every feature is built from the state at the policy boundary before action selection. Perform frame transforms in float64, scale, clip, then cast to float32 once. The Gymnasium space is `Box(-1,1,shape=(40,),dtype=float32)`. Positive-only fields still use this enclosing space.

| Indices | Feature | Scaling and bounds |
|---|---|---|
| 0–2 | body linear velocity | divide by 15 m/s; clip [-1,1] |
| 3–5 | body angular velocity | divide by 12 rad/s; clip [-1,1] |
| 6–8 | body gravity unit direction | clip [-1,1] for roundoff only |
| 9–11 | current gate relative center, body frame | divide by 40 m; clip [-1,1] |
| 12–14 | current gate normal, body frame | unit vector; clip [-1,1] |
| 15–17 | following gate relative center, body frame | divide by 80 m; clip [-1,1] |
| 18–20 | following gate normal, body frame | unit vector; clip [-1,1] |
| 21 | sphere-bottom ground clearance, pz-radius | divide by 20 m; clip [0,1] |
| 22–25 | previous applied normalized action | clip [-1,1]; action meaning below |
| 26–29 | actual four motor thrusts | divide by 4 N; clip [0,1] |
| 30 | current gate present mask | 1 if present; else 0 |
| 31 | following gate present mask | 1 if present; else 0 |
| 32–37 | boundary clearance: xmin,xmax,ymin,ymax,zmin,zmax | sphere surface distances to workspace faces, divide respectively by 120,120,120,120,20,20 m; clip [0,1] |
| 38 | gates remaining including current | count/10; clip [0,1] |
| 39 | episode time remaining | max(0,1-elapsed_seconds/timeout_seconds) |

When current/following gate is absent, its six features are zero and its mask is zero. After final success both slots are absent. There is no wrapping from gate 10 to gate 1. For hover, current center is the hover target and normal is the initial heading normal, mask 30=1, following fields/mask zero, remaining count=0. Record `task_kind=hover` in metadata; the released policy is evaluated only on courses.

At reset, previous action is `[2*(m*g)/(4*fmax)-1,0,0,0]` (approximately -0.019 in its first component). This represents hover collective thrust and zero desired rates. Actual motor thrust is initially mg/4 per rotor. Boundary features are world-aligned: do not claim exact global yaw invariance of the entire observation. Gate geometry is fixed/upright, so width, height and roll need not be encoded in v1. Varying any of them requires a new observation schema and retraining.

Log per-feature clipping counts. Nonfinite state/features are infrastructure errors, not values to sanitize into a valid observation. Finite clipping is part of the policy contract, but out-of-bounds termination still applies before further actions.

## 6. Action: `collective-bodyrate-v1`

SB3 uses a four-dimensional continuous Box [-1,1]. PPO's Gaussian samples and actor means may lie outside that box before postprocessing. The training collector must capture raw sampled output and separately store the clamped command actually applied. Browser/evaluation raw action means the deterministic actor output before clamping.

For finite raw a, set u=clip(a,-1,1). Map:

- `F_des = 0.5*(u[0]+1)*16` N total collective thrust.
- `omega_des = [6*u[1],6*u[2],3*u[3]]` rad/s.
- Hold these commands for two physics steps. Recompute the rate controller from current omega at each physics tick.
- Store u as previous_action after the transition; the low-level mixer output is not previous_action.

Zero normalized collective action is 8 N, approximately hover; it is not zero thrust. Nonfinite actor output pauses browser flight with `inference_nonfinite`; training raises an error and saves diagnostics. No silent zero-action substitute is allowed.

## 7. Model release manifest

Required fields: `schema_version=1`, `model_id`, `created_utc`, `training_run_id`, `checkpoint_id`, `training_transitions`, `training_seed`, `code_commit`, `actor_sha256`, `actor_file`, `onnx_opset=17`, `input_name=obs`, `input_shape=[1,40]`, `output_name=action`, `output_shape=[1,4]`, `dtype=float32`, observation/action IDs, physics/controller/rules/config hashes, `physics_hz=120`, `policy_hz=60`, exact Python/JS dependency lock hashes, held-out evaluation report hash and license.

Shape, hash or configuration mismatch is a loading error. No automatic padding from the original 26-field schema or conversion from rotor-action models is permitted.

## 8. Errors and API boundaries

Structured error object: `code`, user-readable `message`, `recoverable` boolean, `details` object. UI messages must not expose stack traces. Developer console may contain local diagnostic details but no raw imported file content by default.

Python public modules expose pure functions for quaternion operations, derivatives, controller/mixer, observations, gate geometry and event detection. The Gymnasium wrapper owns RNG, episode state, reward accounting and recording. PPO never mutates the physics state directly. TypeScript mirrors the pure numerical interfaces. Frontend React components never calculate authoritative rewards or gate progression.
