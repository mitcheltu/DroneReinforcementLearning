# 09 — Decisions, research evidence and traceability

## 1. Evidence scope

Official documentation and primary research were reviewed on 2026-09-05. They support component feasibility, not the chosen simulator's measured success. Source pages may show development-version documentation; dependency locks must use tested stable releases as document 04 requires. No Kaggle account, GPU quota, deployment host or running application was inspected during this documentation task.

The architecture is feasible for simulation: numerical rigid-body dynamics and low-dimensional control can be trained headlessly; the actor can be exported; browser rendering consumes state. Training convergence, exact compute cost, final completion percentage and browser latency remain empirical acceptance requirements.

## 2. Primary references

| Source | What it supports | What it does not establish |
|---|---|---|
| [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) | Continuous Box policies, vectorized training, PPO parameters, CPU suitability for MLPs | This custom task's learning success or ideal hyperparameters |
| [SB3 export](https://stable-baselines3.readthedocs.io/en/master/guide/export.html) | ONNX deployment path and explicit action postprocessing requirement | Automatic actor-only export with this exact schema |
| [SB3 vector environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) | Vector API/autoreset differences relevant to terminal recording | Correctness of a custom recorder |
| [SB3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html) | Checkpoint/evaluation hooks and vector step counting considerations | Complete curriculum/RNG/session restoration |
| [Gymnasium Env](https://gymnasium.farama.org/api/env/) | reset/step API and termination/truncation semantics | A requirement to treat a finite task deadline as an external truncation |
| [Kaggle notebooks](https://www.kaggle.com/docs/notebooks) | Hosted Python notebook resources, output/version workflow and finite session duration | An individual account's available quota or durable unsaved working files |
| [Kaggle GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage) | Resource-conscious notebook operation | That GPU outperforms CPU for this simulator |
| [ORT WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) | Browser WebGPU inference and explicit provider setup | Universal browser support or superiority over WASM |
| [ORT flags/session options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html) | WASM threading/path and session configuration | That navigator.gpu alone guarantees a working session |
| [Three.js Object3D](https://threejs.org/docs/pages/Object3D.html) | Pose-based scene transforms and configurable up direction | Physics correctness |
| [Three.js Quaternion](https://threejs.org/docs/pages/Quaternion.html) | xyzw API and quaternion interpolation | Automatic conversion from the simulator's wxyz storage |
| [Autonomous Drone Racing with Deep RL](https://arxiv.org/abs/2103.08624) | Research precedent for relative-gate policy observations | Guaranteed arbitrary-track generalization |
| [Champion-level drone racing](https://www.nature.com/articles/s41586-023-06419-4) | PPO-based racing and collective/body-rate action precedent | That this smaller project reproduces Swift's system, compute or real-world performance |

## 3. Key architecture decisions

ADR-001: Use a proportional body-rate inner loop. This reduces the policy's motor-coordination burden and provides a testable classical baseline. It replaces direct rotor PPO control and is included in the initial scope, not a stretch goal.

ADR-002: Fix gate size and keep gates upright. Center/normal observations then encode the allowed orientation. Arbitrary rectangular gate roll or changing openings would require additional observation fields and retraining; those editor controls are absent.

ADR-003: Expand observation to 40 features. Motor response, task time, target-presence flags and workspace constraints are exposed. This improves state information but does not create a global map: the policy still sees only two upcoming gates and may be limited on unusual long-range sequences.

ADR-004: Use rejection-sampled incremental tracks, fixed for an episode. This makes the procedural distribution reproducible and gives explicit constraints. It is not uniform over all possible courses and geometric validity is not dynamic-feasibility proof.

ADR-005: Use strict policy-boundary scheduling. Waiting for inference may reduce real-time speed but avoids untrained variable action holds. Worker scheduling and display FPS cannot change simulation-time action duration.

ADR-006: Record numerical states and actual actions. This supports camera-independent replay and authentic failed attempts without reliance on deterministic re-simulation. Checkpoints alone cannot recreate the history.

ADR-007: Use explicit fixed feature scales. No hidden VecNormalize dependency exists in the first actor release. Future scaling changes require schema/version updates.

ADR-008: Use conservative swept collision geometry and a fixed sphere. This limits implementation complexity and prevents thin-frame tunneling for the defined piecewise-linear sweep. It is not mesh-accurate contact or physically modeled crash dynamics.

ADR-009: Use task deadlines with remaining time observable. Timeout is a terminal task failure. Collector restarts are incomplete records and not training task outcomes. This avoids ambiguity about bootstrapping an artificial external time limit.

ADR-010: Use CPU-first free Kaggle training with locks/checkpoints. Resource and training-time claims are measured. Three-seed standard-release evidence is a quality target; inability to afford it yields an experimental classification, not fabricated reproducibility.

ADR-011: Use single-thread WASM fallback. Version 1 does not require cross-origin isolation. WebGPU is attempted with real initialization checks; actual performance is benchmarked and diagnostics permit WASM selection.

ADR-012: Build replay before long training. Inspection of actual failures is a core development requirement, not a final demo task. Large amounts of training without recordable outcomes are not an acceptable shortcut.

## 4. Historical specification changes

| Historical section | Superseding decision |
|---|---|
| 2.1 latest completed action / loose async loop | Exact two physics ticks per freshly computed action; slow inference slows simulation |
| 4.6 optional motor lag | Required four motor states with tau=0.03 s |
| 5.2 26 observations | 40-field obs-bodyrate-v1 |
| 6.1 direct rotor actions | Collective/body-rate policy plus rate-P mixer |
| 7.1 gamma 0.995 and 16 envs | gamma 0.999; measured 1/2/4/8 env configuration |
| 7.2 repeated alignment reward | Omitted; segmented distance progress and explicit event/time reward |
| 7.3 variable gate size/sharp 3D rotation curriculum | Fixed openings/upright headings, explicit seven-stage curriculum |
| 7.4 unspecified procedural bounds | Exact envelope and rejection algorithm |
| 9.1 [1,26] ONNX input | [1,40] actor-only input |
| 9.3 WASM multithreading headers | Single-thread fallback; isolation deferred |
| 10 gate add/remove/arbitrary rotation | Ten gates; translate, yaw and reorder only |
| 13 unspecified full-training runner | Kaggle notebook plus reusable CLI/checkpoint contract |
| 14 eight-week sequence/replay late | Evidence-gated milestones; early recorder/viewer |
| 18 hierarchical controller stretch goal | Mandatory v1 inner loop |

## 5. Risks and bounded responses

| Risk | Detection | Required response |
|---|---|---|
| PPO fails to stabilize | Stage-0 metrics/reference tests | Fix physics/control first; bounded logged experiments |
| Policy learns reward exploit | High return, low completion; exploit tests | Diagnose on development data; version reward and retrain |
| Supported editor course fails | Per-geometry evaluation breakdown | Show measured limits; do not promise arbitrary completion |
| Python/TS mismatch | Golden fixtures and closed-loop tests | Block release; correct shared contract/implementation |
| Notebook interruption loses work | Missing checkpoint/output version | Resume verified saved artifact; state lost work honestly |
| Replay lacks actual failure endpoint | Autoreset round-trip tests | Fix recorder placement before long training |
| Slow browser/GPU loss | Cycle latency/real-time factor/errors | Slow or pause; recover WASM without stale actions |
| Storage saturation | Archive and free-disk checks | Prune optional replays or stop gracefully with checkpoint |
| Dependency drift | Lock/config hashes and clean install | Controlled lock update with full affected checks |

## 6. Deferred scope and change procedure

Deferred: gate tilt/roll, variable openings, moving gates, more than ten gates, laps, obstacles beyond gate frames, manual flight, camera perception, recurrent policies, wind/noise, real-world flight, multi-GPU PPO, live streaming, automatic video export, detailed contact response, mobile support and WASM multithreading.

To add a deferred capability: write an ADR identifying changed observations/actions/physics/data/UI, bump affected schema/config versions, update training distribution, regenerate fixtures and test partitions where required, train/evaluate compatible models, and provide migration or explicit rejection of old artifacts. UI controls must not expose untrained capabilities as supported features.

## 7. Documentation consistency checklist

Check observation indices total 40; action meanings remain body-rate throughout; gate count and geometry agree in editor/generator/evaluator; hover exception is explicit; timeouts agree with Gymnasium flags; event precedence agrees with replay; checkpoint revision metadata supports episodes spanning updates; reference and measured figures are labeled; relative Markdown links resolve; original specification points to this set.

## 8. Open empirical measurements, not open design choices

The following results cannot be filled in honestly until implementation: selected locked dependency versions after installation tests, fastest Kaggle environment count/device, transitions/s, time to curriculum mastery, best learned checkpoint, held-out completion, actual browser backend speed and hardware matrix. The procedures for deciding them are fixed in documents 04/05/08. No implementer may invent those results to satisfy a documentation placeholder.

## 9. Definition of a supported model

A supported model is an actor bundled with the exact validated simulator/controller/observation/course configuration, satisfying the standard release evaluation and browser tests. A standalone ONNX file or successful smoke run is insufficient. Experimental models remain useful for development/replay, but their UI/model card must say experimental and show actual evaluation status.

## 10. Requirement traceability

| Accepted requirement | Implementation authority | Verification |
|---|---|---|
| Gates 1–10 ordered | 02 events; 03 constraints; 07 editor | Wrong-order/reorder/final-pass tests |
| Finish at 10, no laps | 02 event terminal rule; 01 masks | No wraparound; exact final state |
| Early 1–3 gate learning | 03 seven stages | Stage sampling/promotion cases |
| Upright, fixed aperture | 01 course schema; 02 boxes | Schema rejection and editor controls |
| Reset on editing | 07 state machine/generations | Edit-during-inference race tests |
| Supported and experimental modes | 03 validation; 07 UI | Violated-rule display and disabled Run |
| Thrust/body-rate controller | 01 actions; 02 mixer | Rotor signs, hover, saturation, export mapping |
| Recovery after miss | 02 progression | Miss/recovery/backward cases |
| Reliability before speed | 03 promotion; 04 selection | Saved pre-speed checkpoint and development metric |
| Kaggle notebook training | 05 full workflow | Fresh session, preserved output, new-session resume |
| 3D drone | 07 frames/rendering | Fixture rendering and camera tests |
| Actual failed attempts | 06 data and recorder | Terminal-before-autoreset replay test |
| Metrics for all, sampled trajectories | 04 logs; 06 reservoir | Row counts, deterministic retention and disk caps |
| Desktop chase/orbit | 07 UI/cameras | Desktop/browser matrix |
| Free, resumable resource plan | 04 checkpoint; 05 budgets | Verified continuation and measured throughput |
| Exact state, no perception | 01 observation; 09 scope | No camera input/vision dependencies |

