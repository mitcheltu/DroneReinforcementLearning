# 06 — Actual training recordings and interactive replay

## 1. Purpose and fidelity

The recorder preserves what happened during sampled stochastic training attempts, including failures. A saved model plus seed is not a substitute: PPO actions are sampled, episodes can span policy updates, and cross-runtime physics is not bit-identical. The browser displays recorded state directly and never reruns the actor to recreate a training flight.

Retain numerical trajectories, not videos, as the primary artifact. Video export is deferred. Position/quaternion samples allow arbitrary camera angles and playback speeds after training. The initial implementation records selected attempts at the full 120 Hz physics rate; the browser interpolates for rendering. No 30 Hz downsampling is performed in v1 archives, so failure analysis retains both physics substeps.

## 2. Recording layers

1. Campaign/session/update metadata explains which code, configuration and policy revisions were active.
2. One metrics row exists for every completed or interrupted attempt.
3. Full trajectory buffers are held in memory for active attempts; only retained samples are written durably.
4. Evaluation recordings use the same schema but `origin=evaluation` and deterministic action selection.

The recorder is below the SB3 autoreset boundary. It finalizes an episode before a replacement reset state can overwrite its last sample. The outer collector adds actual raw sampled actions and policy revision information. Gymnasium's and SB3's different vector-environment APIs must be covered by integration tests; do not infer the terminal state from the next reset observation.

With subprocess environments, each worker owns its active numerical buffer and one immutable finalized buffer. A separate telemetry pipe announces completed metadata; the main-process recording coordinator decides retention and requests the completed payload before the next vector step. A reset may allocate/reuse a different active buffer but cannot overwrite the finalized one until acknowledgment. The coordinator alone writes bundles, updates reservoirs/indexes and prunes files; workers never race to edit a shared manifest. A bounded writer queue of eight attempts applies backpressure to collection when full. Checkpoints wait for that queue to flush before saving recorder state. This keeps bulk trajectory arrays out of ordinary Gymnasium info messages and guarantees that accepted recordings are not lost through queue overflow.

## 3. Replay bundle format: `aerorl-replay-v1`

Extension: `.aerorl.zip`. ZIP contains only relative UTF-8 filenames and these entries:

```text
manifest.json
course.json
vehicle.json
observations.json
states.f64
physics_commands.f64
policy_inputs_actions.f32
policy_times.f64
policy_revisions.u32
rewards.f32
terminal_observation.f32
events.json
metrics.json
```

All numeric binary files are contiguous row-major little-endian, without an NPY header. Manifest specifies exact row/column counts, dtype, byte length and SHA-256 per file. Python writes explicit `<f8`, `<f4` and `<u4` dtypes. Browser validates byte lengths before allocating typed views and handles endianness explicitly. Compression uses standard ZIP deflate; consumers must support stored entries too. No pickle, executable JavaScript or arbitrary object deserialization is permitted.

### Manifest fields

Required: schema version 1, replay UUID, origin (`training`, `evaluation`, `reference`, `browser`), run/session/episode/env IDs, task_kind, curriculum stage, reward_mode, campaign transition at start/end, policy revision start/end, checkpoint ID when applicable, all policy revision-to-actor-hash mappings referenced by this attempt, deterministic_actions boolean, complete boolean, interruption reason if incomplete, simulator/controller/observation/action versions, configuration hashes, physics_hz, policy_hz, created_utc, duration_s, file table and summary outcome.

Checkpoint ID is nullable for actual training attempts between checkpoint saves. It must not imply that all actions came from one checkpoint. Per-revision actor hashes are required, but retaining every intermediate actor's weights is not required for visual replay. The course file includes exact generated initial state and `start_reference_m`; the recorded initial state is authoritative if a perturbation is already applied.

### State records

`states.f64`: shape `[N,18]` with columns `[time_s,px,py,pz,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz,f0,f1,f2,f3]`. Include initial time 0, every completed physics tick, and the exact defined interpolated terminal event state when terminal occurs within a tick. No duplicate final timestamps. Times are strictly increasing after the initial row. A zero-duration failure has one state row and its event at zero.

Actual motor thrusts are state values. Do not replace them with requested thrusts. Float64 preserves simulator output precision in archives; browser display may downcast temporary GPU render buffers only.

### Physics-command records

`physics_commands.f64`: one row per executed or partially executed physics interval, columns `[start_time_s,cmd_f0,cmd_f1,cmd_f2,cmd_f3,mixer_lambda]`. Commands are held during the corresponding RK4 interval. These records explain saturation and distinguish requested motor thrust from actual response.

### Policy and reward records

`policy_inputs_actions.f32`: `[M,52]`: normalized observation 40, raw action 4, clamped action 4, physical collective 1, desired body rates 3, in that order. Store actual values used by the environment. Values/log-probabilities from PPO remain in training diagnostics outside this bundle; they are not extra permitted ZIP entries in v1.

`policy_times.f64`: `[M,2]` with start_time_s,end_time_s for each transition. Terminal end may be fractional. `policy_revisions.u32`: `[M]` gives the producing revision. `rewards.f32`: `[M,7]` with progress, gate_bonus, finish_bonus, time_cost, smoothness_cost, failure_cost,total. Costs are positive magnitudes; total equals positives minus costs. Hover uses progress column for its duration-shaped term, finish_bonus for hover success, failure_cost for hover failure, other columns zero except smoothness; metadata identifies task_kind.

`terminal_observation.f32`: 40 values, actual final or last observed next observation. For an incomplete recording this is the last available observation and complete=false. This prevents losing terminal features when vector wrappers auto-reset.

### Events

events.json is an ordered array of objects with event_id, type, time_s, physics_tick_start (integer), tick_fraction [0,1], policy_step (zero-based), gate_id/label or null, target_index_before/after, center_world_m[3], and reason/object_id when relevant. Contact events additionally include contacted frame bar name or workspace face and conservative contact normal if defined. When slab entry ties on axes, choose X before Y before Z for the diagnostic normal. This does not alter collision timing. Same-time ordering follows document 02.

metrics.json contains the attempt row from document 04 plus retention bucket/priority and archive size. `is_success` must agree with the final event and complete flag. Incomplete/aborted records cannot count as failures or successes in evaluation aggregates.

## 4. Retention policy

Store every completed attempt's metrics in append-only JSONL. Each environment holds an entire active attempt's numerical buffer, bounded by task timeout. At 45 seconds, a standard attempt has at most 5,401 state rows and 2,700 policy rows, approximately 1.7 MiB of uncompressed required numeric data plus metadata. Size is an estimate; enforce actual byte counts. With eight environments the active buffers remain modest; allocate arrays once per environment and reuse after finalization.

Training retention buckets are `(million_transition_window,stage,outcome_family)`. Window is floor(global_transition_at_episode_start/1,000,000). Outcome families: success, frame_collision, ground_collision, workspace_exit, timeout, other_failure. Hover departure and state_limit belong to other_failure. Incomplete/user-aborted attempts have metrics but are not reservoir candidates.

Assign deterministic priority as the unsigned integer value of SHA-256 over `run_id + ':' + episode_id + ':retention-v1'`. Lower is better. Keep at most two complete trajectories with the lowest priorities in each bucket. At finalization, replace the bucket's worst retained priority only if the new episode ranks lower; otherwise discard its in-memory trajectory. This samples both successes and failures reproducibly without affecting environment RNG.

Global training replay cap is 1 GiB compressed. If adding a candidate exceeds the cap, evict the highest-priority-value retained trajectory globally until within cap. Log evicted replay IDs; metrics remain. Therefore the cap may leave some buckets without a recording. A summary must display retention counts so users do not mistake sampled failures for the failure frequency.

Evaluation retention: preserve the same first 16 development case IDs at each recorded evaluation to support comparisons. Under the separate 1 GiB evaluation cap, prune oldest intermediate checkpoint groups first; always preserve the first and latest complete 16-case groups. Final release evaluation additionally retains up to 32 successes and 32 failures chosen by lowest hash priority over case ID. If a required preserved group would exceed the cap, stop packaging with an explicit storage error; never silently omit a required release replay. Full per-case metrics remain regardless of trajectory retention.

Crash diagnostics retain the last valid in-memory trajectory separately with complete=false and an error JSON, subject to a 32 MiB diagnostics cap. An infrastructure failure is never labeled as an ordinary PPO failure example.

## 5. Export grouping and index

`replay-index.json` contains summaries and relative bundle paths for retained files, grouped by campaign and evaluation snapshot. It must be rebuilt atomically after replacements/pruning. All entries include file checksum, compressed size, origin, stage, outcome, training step, course ID, duration and policy revision range. User-visible training step refers to the start transition and labels attempts spanning updates accurately.

The web app can import a single replay or an archive of bundles plus replay-index.json. A collection archive must not nest archives recursively: only one outer collection and one level of replay zip files is supported. Validate import budgets cumulatively before extracting bodies.

## 6. Browser replay behavior

Replay has its own mode and immutable course. No Run, gate editing or model action affects a replay. Controls: play/pause, restart, seek slider, step to previous/next recorded physics state, speed 0.25x/0.5x/1x/2x, camera orbit/chase, trail visibility, telemetry visibility, next/previous event and exit replay.

At playback time t, binary-search adjacent state timestamps; linearly interpolate position and scalar/vector telemetry, shortest-arc SLERP attitude. Action, target gate and event indicators are held piecewise constant, not interpolated across decisions. At an event timestamp show the event and updated gate index according to event ordering. Stop at the final recorded state. Never animate a crash bounce or success continuation that is absent from the data.

Telemetry shows time, speed, altitude, current gate, clamped action, actual motors, mixer saturation, accumulated reward and outcome. Optional raw action display is under a diagnostics toggle. Reward curves are sampled at transition ends; distinguish positive rewards and costs. Trail is colored by speed; a separate event marker identifies collisions. Color is never the only indication of gate order/outcome.

Checkpoint comparison selects two recordings with the same course-and-initial-state hash. Show synchronized split views on a common simulation-time axis starting at zero; each stops at its own terminal state. If course hashes differ, disable synchronized comparison with a clear reason. Training attempts are not suitable for claims of checkpoint improvement unless initial conditions match; fixed evaluation pairs supply that evidence.

## 7. Import validation and privacy

Maximum single bundle: 32 MiB compressed, 64 MiB uncompressed, 32 entries. Maximum collection: 1 GiB compressed, 2 GiB declared uncompressed, 2,000 replay bundles; load bundle bodies on demand, not all into RAM. Reject encrypted archives, absolute paths, `..`, duplicate filenames, symlinks, unexpected filenames, oversized shapes, NaN/Infinity, malformed quaternions, timestamps outside [0,45] for standard courses, and invalid event/state consistency. Curriculum time limits follow their task. Abort decompression once actual bytes exceed declared or permitted limits.

Verify hashes before playback. Errors identify the file and rule without executing embedded content. Imported names render as text. Files remain local in browser memory; no upload endpoint exists. Revoke object URLs and release buffers on closing a replay. No persistent browser storage is required in v1; re-import after page reload. Downloads are user-triggered only.

## 8. Required recorder tests

Test a terminal event in each substep, SB3 autoreset, multi-update episode, stochastic versus deterministic action logging, seed-independent retention, priority replacement, archive pruning, little-endian read/write, invalid ZIP paths, size limits, checksum failure, seek at an event, quaternion sign flip, zero-duration error and fixed-course comparison. Validate that the terminal position is the collision location rather than the next reset's position. Numerical arrays round-trip exactly through the binary encoding; JSON metadata round-trips structurally.
