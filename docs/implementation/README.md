# AeroRL 3D implementation contract

Status: M0 contracts, experimental Python training, a browser replay viewer, and a TypeScript dynamics core are implemented. See the [training guide](../../notebooks/README.md) and [browser progress](10-browser-progress.md). Complete browser environment parity, verified Kaggle-hosted execution and trained-policy qualification remain pending.
Design version: 1.0.0. Reviewed: 2026-09-05.

## 1. Authority and interpretation

This document set is the implementation authority for AeroRL 3D version 1. It supersedes conflicting decisions in `../AeroRL_3D_Project_Specification_Revised.md`. The original remains background and historical architecture context. No implementation, successful training run, Kaggle execution, browser benchmark, or release is claimed by these documents.

MUST is a release requirement. SHOULD is not used for required behavior. Numerical values below are selected engineering baselines, not experimentally proven optimums. A change to a baseline must be committed as a new configuration version and evaluated; implementations must not silently choose different values. A failed acceptance test blocks the affected milestone; it does not authorize lowering the threshold without documenting the decision.

Paths in this document set are relative to the repository root unless prefixed with `./` or `../` in a Markdown link. All commands and future file paths describe the implementation to build, not files already present.

## 2. Reading order and ownership

| Document | Authoritative subject |
|---|---|
| [01 — System contracts](01-system-contracts.md) | Modules, units, shared schemas, observation and action contracts |
| [02 — Physics and environment](02-physics-environment.md) | Dynamics, rate controller, integration, events, rewards, environment lifecycle |
| [03 — Courses and curriculum](03-courses-curriculum.md) | Exact generation, validation, ordering, seeds, curriculum |
| [04 — Training and evaluation](04-training-evaluation.md) | PPO, logging, checkpoints, continuation, model selection, export |
| [05 — Kaggle notebook](05-kaggle-notebook.md) | Notebook cells, filesystem, resource benchmarking, output preservation |
| [06 — Recording and replay](06-recording-replay.md) | Recorded data, sampling, binary format, failed attempts, playback |
| [07 — Browser application](07-browser-application.md) | UI, editor, worker scheduling, rendering, inference backend, deployment |
| [08 — Delivery and verification](08-delivery-verification.md) | Repository, ordered implementation work, tests, acceptance and release |
| [09 — Decisions and evidence](09-decisions-evidence.md) | Research, limitations, requirement traceability and superseded choices |

When two documents conflict, the subject owner in this table wins. Fix the conflict in documentation before implementation continues. No hidden defaults in framework code may override these contracts.

## 3. Accepted product scope

1. Standard courses contain exactly ten gates, labeled 1 through 10. Passage must occur in ascending order. Internal array indices are zero-based; labels are one-based.
2. Passing gate 10 finishes the attempt immediately. There are no laps or return-to-start requirement.
3. Introductory training includes hover and 1–3 gate courses; later training includes ten gates.
4. Gates are upright. Users can change heading about world +Z, but cannot pitch or roll gates.
5. Every gate has a 2.5 m wide by 2.5 m high clear opening. Frame dimensions are fixed. Introductory training uses the same opening; reducing size is deferred.
6. Gate edits invalidate the active attempt, pause it and return to its initial state. Gates never move during a flight.
7. Supported mode enforces published course constraints. Experimental mode permits finite, schema-valid geometry inside the workspace after explicit user opt-in and labels it outside the supported distribution; basic physical/schema validity still applies.
8. The policy outputs collective thrust and body rates. A deterministic proportional body-rate controller and motor mixer run at 120 Hz. Direct rotor policy control is excluded.
9. Missing a gate permits recovery. Wrong-order or wrong-direction passage does not advance progress. Ground contact, frame contact, boundary exit, or time limit ends the attempt.
10. Reliability is optimized first. Speed receives more reward only after reliability criteria are reached.
11. Training is headless in Python on Kaggle. The browser provides interactive saved replays. Live streaming from Kaggle is excluded.
12. Every attempt has metrics. A bounded, reproducible sample of actual training successes and failures is retained. Fixed-course evaluation trajectories are also retained.
13. Desktop browser support is required. Chase and orbit cameras are required. Mobile, VR, multiplayer and real drone control are excluded.
14. Training uses free Kaggle resources and resumable sessions. There is no promise that a fixed step budget or one notebook session suffices.
15. Policy observations use exact simulated state and known gate geometry. Camera perception, sensor estimation, noise, wind, moving gates and real-world transfer are excluded.
16. First release is a client-side application. No accounts, backend database, external telemetry service, or upload of imported local replays is required.

## 4. User-visible outcomes

A user can load a benchmark track, see the numbered course, run the autonomous drone, pause/resume/restart, edit allowed gate positions and headings, and observe success or a specific failure reason. A user can import a replay bundle, select a training checkpoint/attempt, scrub or slow down the flight, inspect its target gate and telemetry, and compare progress at different training steps on the same evaluation course.

There is no manual flight control in version 1. The low-level reference controller is a test/development tool, not an alternative UI flight mode. Editing does not retrain the model. The released actor performs inference with fixed weights.

## 5. Design invariants

- One authority integrates the drone state: the custom dynamics implementation.
- World coordinates use meters, seconds and radians; world and body are right-handed, Z-up.
- State quaternions use `[w,x,y,z]`. Render-library conversion occurs only at the rendering adapter.
- Physics: 120 simulation steps/s. Policy: one decision per two physics steps (60/s).
- A slow inference pauses simulation-time advancement at a policy boundary. It never silently reuses an old action for an additional policy interval.
- Observation contract: `float32[40]`, schema `obs-bodyrate-v1`.
- Action contract: `float32[4]`, schema `collective-bodyrate-v1`.
- Full state: 17 float64 values: position 3, quaternion 4, velocity 3, angular velocity 3, motor thrust 4.
- Success/failure events are timestamped in simulation time, not wall-clock time.
- Replays display recorded states. They do not recreate flights by rerunning a stochastic policy.
- Course geometry, model metadata and simulator configuration are versioned and checksummed.
- A candidate model must pass the held-out evaluation and cross-runtime deployment checks before release.

## 6. Completion boundary

Documentation delivery is complete when this set is internally consistent and all accepted product decisions are mapped in document 09. Software delivery is complete only when document 08 acceptance gates are measured and pass. A demo with an untrained controller is a development artifact, not a completed autonomous racing release.
