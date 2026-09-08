# AeroRL 3D: Real-Time Interactive Web Simulator for Autonomous Drone Racing
**Revised Production Specification & Technical Architecture Document**

> **Implementation authority (2026-09-05):** The [implementation documentation](implementation/README.md) supersedes conflicting choices in this historical specification. It fixes ten ordered upright gates, fixed apertures, collective-thrust/body-rate control, a 40-field observation, exact course randomization, Kaggle continuation, actual training replays, and synchronized simulation-time policy/physics behavior. Use that document set to implement version 1; this file remains architectural background. See its [decision register](implementation/09-decisions-evidence.md) for the explicit change mapping.

---

## 1. Executive Summary

### 1.1 Overview

**AeroRL 3D** is a full-stack robotics and machine-learning project that trains an autonomous quadrotor racing policy with deep reinforcement learning in Python and deploys the trained policy directly in a client-side web application.

The system combines:

1. a custom **6-DOF quadrotor rigid-body simulator**;
2. a **Gymnasium** reinforcement-learning environment;
3. **Proximal Policy Optimization (PPO)** using Stable-Baselines3;
4. a PyTorch-to-**ONNX** export pipeline;
5. client-side inference with **ONNX Runtime Web** using WebGPU when available and WebAssembly as a fallback; and
6. an interactive **Three.js / React Three Fiber** race-track editor and visualizer.

Users can reposition and rotate race gates inside a defined feasible workspace, then run the trained controller on the edited course in real time. The application targets a fixed-rate physics simulation and a visually smooth browser render loop while keeping model inference entirely on the client.

The project is intentionally scoped as a **simulation and deployment project**, not as a claim of real-world sim-to-real flight performance.

### 1.2 Primary Engineering Goals

AeroRL 3D demonstrates capability across four engineering areas:

1. **Robotics & Control**
   - 6-DOF rigid-body dynamics
   - body-frame coordinate transforms
   - quaternion attitude representation
   - actuator limits and motor response

2. **Deep Reinforcement Learning**
   - continuous-control PPO
   - reward shaping
   - curriculum learning
   - procedural environment generation
   - held-out policy evaluation

3. **ML Deployment**
   - PyTorch policy export to ONNX
   - deterministic observation preprocessing
   - browser inference through WebGPU / WASM
   - reproducible latency and model-size benchmarking

4. **Interactive 3D Web Engineering**
   - React / Next.js UI
   - Three.js visualization
   - drag-and-drop track editing
   - decoupled inference, physics, and render loops

### 1.3 Scope Boundaries

The initial production version does **not** attempt to provide:

- real-world drone control;
- photorealistic aerodynamics;
- propeller wake interaction;
- battery-voltage modeling;
- sensor noise equivalent to a physical flight controller;
- guaranteed completion of arbitrary user-created tracks; or
- bit-for-bit deterministic trajectories across Python, JavaScript engines, CPUs, and GPUs.

Instead, the project defines a constrained track distribution and validates performance statistically on held-out procedural tracks.

---

## 2. System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE TRAINING PIPELINE                            │
└──────────────────────────────────────────────────────────────────────────────┘

Shared Vehicle / Observation Configuration
                  │
                  ▼
       ┌───────────────────────┐
       │ Python Quadrotor      │
       │ 6-DOF Dynamics        │
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │ Gymnasium Environment │
       │ Procedural Tracks     │
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │ Stable-Baselines3 PPO │
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │ PyTorch Policy        │
       │ Checkpoint            │
       └───────────┬───────────┘
                   │
            ONNX Export
                   │
                   ▼
       ┌───────────────────────┐
       │ Browser Policy Model  │
       │ .onnx                 │
       └───────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                         CLIENT-SIDE WEB RUNTIME                              │
└──────────────────────────────────────────────────────────────────────────────┘

 User Gate Editor ───────► Race Track State
                              │
                              ▼
                     Observation Builder
                              │
                              ▼
                ONNX Runtime Web Policy Loop
                  WebGPU → WASM fallback
                              │
                              ▼
                    Action Postprocessing
                              │
                              ▼
                    Fixed-Step Physics Loop
                         target: 120 Hz
                              │
                              ▼
                   Interpolated Render State
                              │
                              ▼
                     Three.js Render Loop
                      display-dependent FPS
```

### 2.1 Runtime Loop Separation

The browser architecture must not tie neural inference directly to `requestAnimationFrame()` or use asynchronous inference completion as the simulation clock.

The runtime is divided into three loops:

| Loop | Target Rate | Purpose |
|---|---:|---|
| **Physics** | 120 Hz | Fixed-step rigid-body integration |
| **Policy** | 60 Hz initially | ONNX inference and action update |
| **Rendering** | Display refresh rate | Three.js visualization |

The physics loop holds the latest completed policy action between policy updates. Rendering interpolates between physics states when appropriate.

If main-thread load becomes noticeable, inference and/or physics should move to a Web Worker.

---

## 3. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| RL Training | Python 3.11+, PyTorch 2.x, Gymnasium | Environment and policy training |
| RL Framework | Stable-Baselines3 PPO | Continuous-control learning |
| Experimentation | Optuna, TensorBoard | Hyperparameter search and training analysis |
| Model Export | ONNX | Browser-compatible policy serialization |
| Model Inspection | Netron | ONNX graph validation |
| Frontend | React 19, Next.js, TypeScript | Application shell and UI |
| 3D Rendering | Three.js, `@react-three/fiber`, `@react-three/drei` | Drone, gates, terrain, cameras |
| Client Inference | ONNX Runtime Web | WebGPU / WASM inference |
| Physics | Custom Python + TypeScript implementations | Matched 6-DOF dynamics |
| Deployment | Vercel | Static assets and frontend hosting |
| CI/CD | GitHub Actions | Tests, export checks, deployment |

### 3.1 Physics Ownership

The quadrotor state is integrated by the custom physics implementation.

A second rigid-body physics engine must **not** simultaneously integrate the drone state. If a library such as Rapier is introduced later, it should be limited to collision queries or non-drone scene objects unless the architecture is deliberately changed to make Rapier the authoritative rigid-body solver.

---

## 4. Quadrotor Dynamics

### 4.1 Degrees of Freedom and State Representation

The quadrotor is a **6-degree-of-freedom rigid body** represented by a quaternion-based state.

The minimum flight state is:

\[
\mathbf{x} =
\begin{bmatrix}
\mathbf{p} \\
\mathbf{q} \\
\mathbf{v} \\
\boldsymbol{\omega}
\end{bmatrix}
\]

where:

- \(\mathbf{p}=[x,y,z]^T\): world-frame position, 3 values;
- \(\mathbf{q}=[q_w,q_x,q_y,q_z]^T\): unit quaternion representing body-to-world attitude, 4 values;
- \(\mathbf{v}=[v_x,v_y,v_z]^T\): world-frame linear velocity, 3 values;
- \(\boldsymbol{\omega}=[\omega_x,\omega_y,\omega_z]^T\): body-frame angular velocity, 3 values.

This produces **13 numerical state values** while still representing a 6-DOF rigid body.

Optionally, four motor-thrust states may be appended if first-order actuator dynamics are modeled explicitly.

### 4.2 Translational Dynamics

The translational equations are:

\[
\dot{\mathbf{p}} = \mathbf{v}
\]

\[
m\dot{\mathbf{v}} =
\begin{bmatrix}
0\\
0\\
-mg
\end{bmatrix}
+
\mathbf{R}(\mathbf{q})
\begin{bmatrix}
0\\
0\\
U_1
\end{bmatrix}
-
\mathbf{K}_{drag}\mathbf{v}
\]

with:

\[
U_1=\sum_{i=1}^{4}f_i
\]

where \(f_i\) is the physical thrust produced by rotor \(i\).

Initial vehicle parameters are configuration values rather than hard-coded assumptions:

| Parameter | Initial Value | Notes |
|---|---:|---|
| Mass \(m\) | 0.8 kg | Tunable |
| Gravity \(g\) | 9.81 m/s² | Constant |
| Arm length \(l\) | 0.20 m | Vehicle geometry |
| \(I_{xx}\) | 0.005 kg·m² | Initial estimate |
| \(I_{yy}\) | 0.005 kg·m² | Initial estimate |
| \(I_{zz}\) | 0.009 kg·m² | Initial estimate |
| Linear drag | Tunable | Simplified aerodynamic model |
| Max rotor thrust \(f_{max}\) | Configured experimentally | Must exceed hover thrust with maneuvering headroom |

For a 0.8 kg vehicle, hover requires approximately \(7.85\,\text{N}\) of total thrust, or approximately \(1.96\,\text{N}\) per rotor under symmetric loading. The configured maximum thrust must provide sufficient headroom for racing maneuvers.

### 4.3 Rotational Dynamics

Rigid-body angular acceleration is calculated as:

\[
\mathbf{I}\dot{\boldsymbol{\omega}}
=
\boldsymbol{\tau}
-
\boldsymbol{\omega}\times(\mathbf{I}\boldsymbol{\omega})
\]

where:

\[
\mathbf{I}=\operatorname{diag}(I_{xx},I_{yy},I_{zz})
\]

and \(\boldsymbol{\tau}=[\tau_x,\tau_y,\tau_z]^T\) is the body torque produced by differential rotor thrust.

For the selected X-frame rotor convention, the mixer signs must be defined once in shared configuration and tested explicitly. Python and TypeScript must use the same rotor numbering, spin directions, and torque convention.

### 4.4 Quaternion Attitude Kinematics

Body angular velocity is **not** integrated by directly adding \(\boldsymbol{\omega}\Delta t\) to roll, pitch, and yaw.

Instead, quaternion attitude evolves according to:

\[
\dot{\mathbf{q}}
=
\frac{1}{2}
\mathbf{q}\otimes
\begin{bmatrix}
0\\
\boldsymbol{\omega}
\end{bmatrix}
\]

After every integration step:

\[
\mathbf{q} \leftarrow \frac{\mathbf{q}}{\|\mathbf{q}\|}
\]

This avoids Euler-angle singularities and provides a correct representation for aggressive 3D rotation.

### 4.5 Numerical Integration

The target implementation should use one of the following consistently in Python and TypeScript:

1. fixed-step RK4 for the coupled dynamics; or
2. another explicitly documented fixed-step integrator validated against RK4 reference trajectories.

The initial target physics timestep is:

\[
\Delta t = \frac{1}{120}\text{ s}
\]

The project must report the actual integrator used. Resume claims must match the implemented algorithm.

### 4.6 Optional Motor Dynamics

A more realistic actuator model can use first-order thrust response:

\[
\dot f_i = \frac{f_{cmd,i}-f_i}{\tau_m}
\]

with:

\[
0\le f_i\le f_{max}
\]

This is recommended if direct rotor-thrust control remains the policy action space.

---

## 5. Coordinate Frames and Observation Design

### 5.1 Coordinate Frames

AeroRL 3D uses:

- a fixed world frame;
- a body frame attached to the quadrotor; and
- a local coordinate frame for each gate.

For a gate with world position \(\mathbf{p}_{gate}\) and drone world position \(\mathbf{p}_{drone}\):

\[
\Delta\mathbf{p}_{world}
=
\mathbf{p}_{gate}-\mathbf{p}_{drone}
\]

The relative gate position in body coordinates is:

\[
\mathbf{p}_{gate}^{body}
=
\mathbf{R}(\mathbf{q})^T\Delta\mathbf{p}_{world}
\]

The gate traversal normal is transformed similarly:

\[
\mathbf{n}_{gate}^{body}
=
\mathbf{R}(\mathbf{q})^T\mathbf{n}_{gate}^{world}
\]

This encoding removes dependence on absolute world translation and greatly reduces dependence on absolute yaw. It should be described as **ego-centric/body-frame observation encoding**, not full arbitrary SE(3) invariance because gravity defines a privileged world direction.

### 5.2 Recommended Observation Vector

The initial policy observation is expanded beyond the original 18-element design so the task remains approximately Markov with respect to action smoothness and ground avoidance.

Recommended schema:

| Index | Field | Size | Normalization |
|---:|---|---:|---|
| `0..2` | Body-frame linear velocity | 3 | divide by configured velocity scale |
| `3..5` | Body angular velocity | 3 | divide by configured angular-rate scale |
| `6..8` | Gravity direction in body frame | 3 | already approximately `[-1,1]` |
| `9..11` | Next gate center in body frame | 3 | divide by next-gate distance scale |
| `12..14` | Next gate forward normal in body frame | 3 | unit vector |
| `15..17` | Following gate center in body frame | 3 | divide by lookahead distance scale |
| `18..20` | Following gate normal in body frame | 3 | unit vector |
| `21` | Ground clearance / altitude feature | 1 | scaled to operating envelope |
| `22..25` | Previous normalized action | 4 | already `[-1,1]` |

**Total recommended observation dimension: 26.**

The exact schema may change during experimentation, but once a model is exported, the schema must be versioned and shared by Python and TypeScript.

### 5.3 Observation Normalization

Observation preprocessing must be identical during training and browser deployment.

Preferred approach for the first version:

- use explicit feature scaling from configuration;
- clip physically impossible/extreme values to defined bounds; and
- store preprocessing constants in a shared JSON metadata file.

Example:

```json
{
  "schema_version": 1,
  "velocity_scale_mps": 15.0,
  "angular_rate_scale_radps": 12.0,
  "next_gate_distance_scale_m": 40.0,
  "lookahead_gate_distance_scale_m": 80.0,
  "altitude_scale_m": 20.0
}
```

If `VecNormalize` is used instead, the learned statistics must be exported and reproduced exactly in the browser.

---

## 6. Action Space and Low-Level Control

### 6.1 Version 1: Direct Rotor-Thrust Control

The PPO policy outputs:

\[
\mathbf{a}_t\in[-1,1]^4
\]

Each normalized action is converted to physical thrust:

\[
f_i = \operatorname{clip}\left(\frac{a_i+1}{2},0,1\right)f_{max}
\]

This conversion must occur identically in training evaluation and browser inference.

The browser must **never** pass raw negative PPO actions directly into the physics engine as rotor thrusts.

### 6.2 Alternative Version: Body-Rate Controller

If training direct rotor thrust proves unnecessarily difficult, the project may switch to a hierarchical controller:

```text
PPO Policy
   │
   ├── collective thrust command
   └── desired body rates (p, q, r)
                │
                ▼
       low-level PID/rate controller
                │
                ▼
            motor mixer
                │
                ▼
          rotor thrusts
```

This version is closer to common practical multirotor control architectures and may improve learning stability.

The final README must clearly document which controller architecture is actually implemented.

---

## 7. Reinforcement Learning Strategy

### 7.1 Algorithm

The initial agent uses **PPO** from Stable-Baselines3 with an MLP actor-critic policy.

Starting configuration:

| Hyperparameter | Initial Value | Status |
|---|---:|---|
| Network | 256 × 256 MLP | Starting point |
| Learning rate | `3e-4` linear schedule | Tune experimentally |
| Total training budget | up to `10M` environment steps | Budget, not guarantee |
| Vectorized environments | 16 | Hardware-dependent |
| GAE lambda | `0.95` | Starting point |
| PPO clip range | `0.2` | Starting point |
| Batch size | `512` | Validate against rollout size |
| Epochs/update | `10` | Starting point |
| Discount factor | `0.995` initial candidate | Tune for policy frequency |

The discount factor must be interpreted in relation to the policy update rate. At high control frequencies, `gamma = 0.99` can correspond to a relatively short real-time planning horizon, so `0.995`, `0.997`, `0.999`, and nearby values should be included in experiments rather than assuming one value is universally best.

### 7.2 Reward Function

The reward should encourage forward course progress, valid gate passage, speed, and stable control without creating incentives for reckless gate hits.

A candidate reward is:

\[
R_t =
R_{progress}
+
R_{alignment}
+
R_{passage}
-
R_{control}
-
R_{time}
-
R_{collision}
\]

#### Progress Reward

\[
R_{progress}
=
k_p(d_{t-1}-d_t)
\]

where \(d_t\) is distance to the next gate target.

#### Alignment Reward

\[
R_{alignment}
=
k_a
\frac{\mathbf{v}_t\cdot\mathbf{n}_{gate}}
{\|\mathbf{v}_t\|+\epsilon}
\]

Gate normals must always point in the intended forward traversal direction.

#### Valid Gate Passage

A gate pass occurs only when all of the following are true:

1. the drone crosses the gate plane from the valid side;
2. the crossing point lies inside the gate aperture with a drone-size safety margin; and
3. the drone has not collided with the gate frame.

A large positive sparse reward is applied only for a valid pass.

#### Control Smoothness

\[
R_{control}
=
k_s\|\mathbf{a}_t-\mathbf{a}_{t-1}\|^2
\]

Because this reward depends on the previous action, the previous action is included in the observation.

#### Time Penalty

A small step penalty encourages faster completion:

\[
R_{time}=k_t
\]

with \(k_t>0\) subtracted each policy step.

#### Collision Penalty

A collision or out-of-bounds condition terminates the episode with a penalty large enough that a policy is not rewarded overall for passing one gate and immediately crashing.

A high roll or pitch angle is **not automatically considered a crash**. Aggressive attitude is allowed as long as the vehicle remains dynamically valid and collision-free.

### 7.3 Curriculum Learning

Difficulty should increase across multiple dimensions, not merely gate distance.

Suggested curriculum:

1. hover and stabilization;
2. reach one large gate;
3. fly through one gate from randomized starts;
4. straight multi-gate sequences;
5. moderate horizontal turns;
6. altitude changes;
7. smaller apertures;
8. faster initial conditions;
9. sharper combined 3D turns;
10. fully randomized tracks within the defined operating envelope.

### 7.4 Procedural Track Distribution

The policy is evaluated only on tracks satisfying documented feasibility constraints.

Example initial envelope:

| Property | Initial Constraint |
|---|---|
| Workspace | bounded rectangular volume |
| Gate spacing | minimum and maximum distance |
| Gate aperture | fixed or bounded size range |
| Ground clearance | minimum altitude margin |
| Turn severity | bounded using gate geometry and spacing |
| Vertical displacement | bounded between consecutive gates |
| Initial drone state | sampled from defined distribution |

Exact numerical bounds must be chosen during environment development and included in the released benchmark configuration.

### 7.5 Evaluation Protocol

Training and evaluation procedural seeds must be separated.

The project should report:

- track completion rate;
- gate completion rate;
- collision rate;
- timeout rate;
- mean lap/completion time;
- median and p95 completion time;
- mean control effort;
- inference latency;
- numerical physics parity error.

A statement such as “95% completion” is valid only when accompanied by the test distribution, number of held-out tracks, and evaluation conditions.

---

## 8. Cross-Language Physics Validation

### 8.1 Goal

Python and TypeScript implementations should model the **same equations, parameters, rotor convention, integrator, and timestep**.

The goal is numerical agreement sufficient for policy deployment, not a claim of universal bit-identical determinism.

### 8.2 Shared Configuration

The following values should come from shared versioned configuration files where practical:

- mass;
- inertia;
- gravity;
- arm length;
- drag parameters;
- rotor spin directions;
- thrust limits;
- motor time constant;
- physics timestep;
- observation scales;
- gate dimensions;
- collision margins.

### 8.3 Golden Trajectory Tests

Python generates reference fixtures containing:

- initial state;
- sequence of physical rotor actions;
- timestep;
- expected state after each step.

The TypeScript implementation replays the same actions and compares trajectories.

Tests should include:

1. zero-force constant-velocity motion with gravity disabled;
2. analytical free fall;
3. symmetric hover thrust;
4. pure roll-torque sign test;
5. pure pitch-torque sign test;
6. yaw-torque sign test;
7. quaternion norm preservation;
8. random short-horizon trajectories;
9. long-horizon bounded-error trajectory checks;
10. convergence tests as timestep decreases.

### 8.4 Error Reporting

Do not promise a universal `<1e-6` trajectory error before measurement.

Instead report measured values such as:

```text
One-step max absolute state error: ...
1-second trajectory RMSE: ...
10-second trajectory RMSE: ...
Quaternion angular disagreement: ... degrees
```

The benchmark environment, browser, CPU, and software versions should be recorded.

---

## 9. Browser Inference Architecture

### 9.1 ONNX Export Requirements

The exported actor must have a stable input/output contract.

Recommended contract:

```text
Input:
  obs: float32[1, 26]

Output:
  action: float32[1, 4]
```

The exported model should contain only the deterministic actor path required for inference. Training-only critic components should not be required by the browser.

### 9.2 Inference Backend Selection

WebGPU should be used when supported, with WASM fallback.

Example initialization pattern:

```typescript
let ort: typeof import('onnxruntime-web');

if ('gpu' in navigator) {
  ort = await import('onnxruntime-web/webgpu');
} else {
  ort = await import('onnxruntime-web');
}
```

The exact bundling approach may be adjusted for Next.js, but backend support must be detected rather than assumed.

### 9.3 WASM Multithreading

If ONNX Runtime WASM multithreading is enabled, deployment must configure the required cross-origin isolation headers.

Example Next.js configuration:

```typescript
async headers() {
  return [
    {
      source: '/(.*)',
      headers: [
        {
          key: 'Cross-Origin-Opener-Policy',
          value: 'same-origin',
        },
        {
          key: 'Cross-Origin-Embedder-Policy',
          value: 'require-corp',
        },
      ],
    },
  ];
}
```

This must be tested against all externally loaded assets because cross-origin isolation can affect resource loading.

### 9.4 Action Postprocessing

Browser inference must explicitly clamp and map the network action before physics:

```typescript
function normalizedActionToThrust(
  action: Float32Array,
  fMax: number,
): Float32Array {
  const thrust = new Float32Array(4);

  for (let i = 0; i < 4; i++) {
    const u = Math.max(-1, Math.min(1, action[i]));
    thrust[i] = 0.5 * (u + 1) * fMax;
  }

  return thrust;
}
```

### 9.5 Runtime Controller Sketch

```typescript
class DroneRuntime {
  private latestAction = new Float32Array(4);
  private physicsAccumulator = 0;
  private readonly physicsDt = 1 / 120;

  async updatePolicy(observation: Float32Array) {
    const tensor = new ort.Tensor('float32', observation, [1, 26]);
    const output = await this.session.run({ obs: tensor });
    const normalizedAction = output.action.data as Float32Array;
    this.latestAction = normalizedActionToThrust(
      normalizedAction,
      this.vehicleParams.fMax,
    );
  }

  advancePhysics(frameDt: number) {
    this.physicsAccumulator += Math.min(frameDt, 0.05);

    while (this.physicsAccumulator >= this.physicsDt) {
      this.state = this.physics.step(
        this.state,
        this.latestAction,
        this.physicsDt,
      );
      this.physicsAccumulator -= this.physicsDt;
    }
  }
}
```

The actual production version should guard against stale async policy responses and should avoid allocating unnecessary tensors or arrays in hot loops.

---

## 10. Interactive Track Editor

### 10.1 Gate Manipulation

Each gate stores:

```typescript
interface RaceGate {
  id: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  width: number;
  height: number;
}
```

Users may:

- translate gates;
- rotate gates;
- add/remove gates;
- reorder gates; and
- reset to benchmark tracks.

### 10.2 Feasibility Validation

The editor should display warnings for configurations outside the benchmark envelope, including cases such as:

- gates below the floor;
- overlapping gate frames;
- gate spacing below a minimum;
- extreme vertical jumps;
- gates outside the configured workspace.

The application may still allow experimental layouts, but it should label them as **out of distribution** rather than promising successful navigation.

### 10.3 Editing During Flight

The initial production mode should pause/reset the drone when course geometry is changed.

A future experiment may support moving gates during flight, but that requires training with moving targets or otherwise validating robustness to that distribution shift.

---

## 11. Project Directory Layout

```text
aerorl-3d/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── train_release.yml
│       └── deploy_web.yml
│
├── shared/
│   ├── vehicle_params.json
│   ├── observation_schema.json
│   └── benchmark_tracks.json
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MATHEMATICS.md
│   ├── BENCHMARKS.md
│   └── MODEL_CARD.md
│
├── training/
│   ├── configs/
│   │   └── ppo_drone.yaml
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── quadrotor_env.py
│   │   ├── quadrotor_physics.py
│   │   ├── observations.py
│   │   ├── rewards.py
│   │   └── track_generator.py
│   ├── scripts/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── export_onnx.py
│   │   └── generate_golden_trajectories.py
│   ├── tests/
│   │   ├── test_dynamics.py
│   │   ├── test_observations.py
│   │   ├── test_gate_crossing.py
│   │   └── test_onnx_export.py
│   └── artifacts/
│       └── .gitkeep
│
├── web/
│   ├── public/
│   │   ├── models/
│   │   │   └── drone_policy.onnx
│   │   └── assets/
│   │       ├── drone.glb
│   │       └── gate.glb
│   ├── src/
│   │   ├── components/
│   │   │   ├── DroneCanvas.tsx
│   │   │   ├── DroneMesh.tsx
│   │   │   ├── GateManipulator.tsx
│   │   │   ├── RaceTrack.tsx
│   │   │   └── TelemetryUI.tsx
│   │   ├── inference/
│   │   │   ├── createSession.ts
│   │   │   ├── observation.ts
│   │   │   └── actionMapping.ts
│   │   ├── physics/
│   │   │   ├── QuadrotorPhysics.ts
│   │   │   └── quaternion.ts
│   │   ├── runtime/
│   │   │   └── DroneRuntime.ts
│   │   ├── workers/
│   │   │   └── simulation.worker.ts
│   │   └── hooks/
│   │       └── useRaceTrack.ts
│   ├── tests/
│   │   └── physicsParity.test.ts
│   ├── next.config.ts
│   └── package.json
│
└── README.md
```

---

## 12. Test Strategy

### 12.1 Physics Tests

Required tests:

- free fall against analytical solution;
- hover equilibrium tolerance;
- rotor mixer sign tests;
- quaternion normalization;
- zero-input sanity checks;
- timestep convergence;
- collision geometry;
- Python/TypeScript fixture replay.

Momentum and mechanical-energy conservation should **not** be used as general pass/fail criteria because the model includes external gravity, thrust, drag, and actuator work.

### 12.2 Environment Tests

- Gymnasium environment checker;
- observation shape and finite-value tests;
- action clipping tests;
- gate crossing direction tests;
- reward component unit tests;
- deterministic seeded reset tests;
- time-limit termination tests;
- collision termination tests.

### 12.3 ONNX Tests

For a set of fixed observations:

1. run the PyTorch actor;
2. run the ONNX actor;
3. compare normalized actions;
4. verify correct action clipping and thrust mapping.

Store model metadata next to the ONNX artifact:

```json
{
  "model_version": "0.1.0",
  "observation_schema_version": 1,
  "action_space": "normalized_rotor_thrust",
  "policy_hz": 60,
  "training_commit": "<git-sha>"
}
```

### 12.4 Browser Tests

Validate at minimum:

- Chromium with WebGPU;
- Chromium WASM fallback;
- Firefox WASM fallback;
- Safari behavior where supported;
- cross-origin isolation configuration;
- model loading failures;
- tab visibility / pause behavior;
- sustained simulation performance.

---

## 13. CI/CD and Training Workflow

### 13.1 Pull Request CI

Every pull request should run:

```text
Python lint/type checks
Python unit tests
Gymnasium environment smoke test
short PPO smoke-training run
ONNX export test
PyTorch ↔ ONNX output comparison
TypeScript lint/type checks
TypeScript tests
Python ↔ TypeScript physics fixture comparison
Next.js production build
```

Full multi-million-step training should **not** run on every pull request.

### 13.2 Full Training Workflow

A manually triggered or release workflow may:

1. train the selected PPO configuration;
2. evaluate on held-out tracks;
3. export the actor to ONNX;
4. benchmark PyTorch/ONNX agreement;
5. generate evaluation plots;
6. publish model artifacts for a tagged release.

The exact runner strategy depends on available compute resources.

---

## 14. Implementation Roadmap

### Phase 1 — Correct Physics Core (Weeks 1–2)

- implement quaternion utility library;
- implement 6-DOF Python dynamics;
- define vehicle parameters and rotor convention;
- implement fixed-step numerical integration;
- create unit tests and analytical sanity checks;
- generate golden trajectory fixtures;
- implement matching TypeScript dynamics;
- measure short-horizon parity.

**Exit criterion:** hover, free fall, and torque tests pass; Python and TypeScript trajectories agree within documented tolerances.

### Phase 2 — Gymnasium Environment and Baseline Control (Week 2–3)

- implement observation builder;
- implement action scaling;
- implement gate plane/aperture checks;
- implement collisions and terminations;
- implement procedural track generator;
- validate environment with a simple scripted controller where possible.

**Exit criterion:** environment can run thousands of seeded episodes without NaNs or invalid states.

### Phase 3 — PPO Training and Curriculum (Weeks 3–5)

- train hover/single-gate policy;
- add multi-gate curriculum;
- tune reward scales;
- compare discount factors and policy frequencies;
- test direct rotor thrust vs hierarchical rate-control architecture if necessary;
- evaluate on held-out procedural seeds.

**Target:** establish a repeatable completion baseline on constrained randomized courses. A stretch target is `>95%` completion on the documented held-out distribution, but this is not considered achieved until measured.

### Phase 4 — ONNX Export and Browser Inference (Week 5–6)

- export deterministic actor only;
- validate PyTorch/ONNX numerical agreement;
- implement shared observation preprocessing;
- implement browser action postprocessing;
- support WebGPU with WASM fallback;
- benchmark p50/p95 inference latency.

**Target:** model small enough for immediate web delivery and inference comfortably below the selected policy interval.

### Phase 5 — Three.js Interactive Application (Weeks 6–7)

- implement drone and gate rendering;
- implement track editing controls;
- implement fixed-step browser runtime;
- integrate ONNX policy loop;
- add chase/orbit cameras;
- add telemetry and restart controls;
- add out-of-distribution track warnings.

**Target:** stable interactive demo on desktop browsers at a visually smooth frame rate.

### Phase 6 — Validation, CI/CD, and Portfolio Release (Week 8)

- build benchmark scripts;
- run held-out evaluation suite;
- collect browser latency and FPS measurements;
- create CI workflows;
- deploy to Vercel;
- write model card and benchmark methodology;
- record demo video;
- finalize README and resume bullets using measured values only.

---

## 15. Benchmark Plan

### 15.1 Metrics to Measure

Do not pre-populate portfolio pages with invented benchmark numbers. Measure and report:

| Metric | Reporting Format |
|---|---|
| Held-out track completion | `% over N tracks` |
| Gate pass rate | `%` |
| Collision rate | `%` |
| Median completion time | seconds |
| Policy latency | p50 / p95 / p99 ms |
| Physics step time | p50 / p95 µs or ms |
| Render performance | median / p5 FPS |
| ONNX file size | MB |
| PyTorch ↔ ONNX action error | max abs / RMSE |
| Python ↔ TypeScript physics error | one-step and trajectory metrics |

### 15.2 Example Benchmark Language

Use wording such as:

> Achieved **97.2% completion across 5,000 held-out procedurally generated tracks** satisfying the published AeroRL benchmark constraints.

or:

> Achieved **0.31 ms p50 / 0.52 ms p95 WASM policy inference** on Chrome 152 using a specified desktop CPU.

Only use these structures after replacing the example values with actual measurements.

### 15.3 Browser Benchmark Methodology

Record:

- browser and version;
- OS;
- CPU;
- GPU for WebGPU tests;
- ONNX Runtime version;
- warm-up iterations;
- measured iterations;
- whether cross-origin isolation and WASM threading are enabled.

A small MLP may be faster on WASM than WebGPU because GPU dispatch overhead can dominate computation. The implementation should benchmark both instead of assuming WebGPU is faster.

---

## 16. Portfolio and Resume Representation

Resume bullets should describe completed work and measured results, not planned architecture.

### 16.1 Robotics / Control Version

After implementation, a defensible structure is:

> Built a browser-deployable **6-DOF quadrotor simulator** with quaternion attitude dynamics and matched Python/TypeScript fixed-step physics, validating cross-language trajectories against shared regression fixtures.

> Trained a **PPO continuous-control policy** on procedurally generated drone-racing tracks using ego-centric gate observations, curriculum learning, and held-out evaluation across randomized courses.

### 16.2 ML / Deployment Version

> Built an end-to-end **PyTorch-to-ONNX deployment pipeline** for a reinforcement-learning control policy, reproducing training-time preprocessing and action scaling for client-side WebGPU/WASM inference.

> Developed an interactive **React + Three.js autonomous drone simulator** with editable 3D race tracks, fixed-rate physics, live telemetry, and fully client-side neural policy execution.

### 16.3 Adding Metrics

Once benchmarks exist, replace generic phrases with measured results, for example:

> ... achieving **X% completion over N held-out procedural tracks**.

> ... with **Y ms p95 browser inference latency** and a **Z MB ONNX model**.

Do not claim:

- “12-DOF” for this rigid body;
- C++ unless C++ is actually implemented;
- RK4 unless RK4 is actually used;
- zero cross-platform simulation error;
- 100% generalization over arbitrary user tracks; or
- measured latency values that have not been benchmarked.

---

## 17. Success Criteria

The project is considered technically successful when all of the following are true:

1. the Python simulator passes analytical and unit tests;
2. the TypeScript simulator reproduces reference trajectories within documented tolerance;
3. the Gymnasium environment trains without systematic numerical instability;
4. the PPO policy demonstrates reliable completion on a documented held-out course distribution;
5. PyTorch and ONNX actors produce sufficiently close actions for the same normalized observations;
6. the browser correctly performs observation preprocessing and action-to-thrust conversion;
7. the browser maintains fixed-step simulation independently of render timing;
8. the application supports WebGPU where available and functional WASM fallback elsewhere;
9. gate editing is constrained or clearly labeled relative to the policy's trained distribution; and
10. every public benchmark and resume metric is generated by a reproducible measurement script.

---

## 18. Stretch Goals

These features are intentionally excluded from the minimum viable version but can extend the project later:

- body-rate/PID hierarchical controller;
- dynamic motor and battery model;
- sensor latency and noise randomization;
- moving gates;
- obstacle avoidance beyond race gates;
- recurrent policy or Transformer policy;
- imitation-learning warm start;
- policy distillation into a smaller student model;
- domain randomization for sim-to-real research;
- Rust/WASM shared physics core to reduce duplicated Python/TypeScript logic;
- real flight-controller integration in a controlled research environment.

---

## 19. External Technical References

The implementation should be checked against current official documentation during development:

- Stable-Baselines3 PPO: https://stable-baselines3.readthedocs.io/
- Stable-Baselines3 ONNX export guidance: https://stable-baselines3.readthedocs.io/en/master/guide/export.html
- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/
- ONNX Runtime WebGPU: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- ONNX Runtime web performance guidance: https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html
- Gymnasium custom environments: https://gymnasium.farama.org/
- Three.js: https://threejs.org/
- React Three Fiber: https://r3f.docs.pmnd.rs/
- MDN WebGPU: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API

---

## 20. Final Architecture Summary

AeroRL 3D is feasible as an eight-week portfolio-scale project if the implementation prioritizes correctness and measurable scope over unsupported performance claims.

The recommended production path is:

```text
Procedural Track Generator
          │
          ▼
Quaternion 6-DOF Gymnasium Simulation
          │
          ▼
Normalized Ego-Centric Observation
          │
          ▼
Stable-Baselines3 PPO
          │
          ▼
Deterministic Actor Export
          │
          ▼
ONNX
          │
          ▼
Browser Observation Builder
          │
          ▼
ONNX Runtime Web
(WebGPU / WASM)
          │
          ▼
Action Clamp + Physical Thrust Mapping
          │
          ▼
120 Hz Fixed-Step TypeScript Physics
          │
          ▼
Three.js Visualization + Track Editor
```

The primary portfolio value is not a claim that the browser simulator perfectly reproduces a physical racing drone. The value is demonstrating an end-to-end engineering pipeline in which a mathematically valid control environment, reinforcement-learning policy, model-export process, browser inference engine, interactive 3D application, and reproducible evaluation framework operate together as one system.
