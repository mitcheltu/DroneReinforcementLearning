# Run 008: recovery-focused reinforcement learning

## Objective and limits

Learn closed-loop recovery from drift, overshoot, wrong height and heading,
including wide gate spacing with large altitude changes, while retaining the
existing hover, navigation and maneuver skills. This is simulated state-based
control with ordered upright yaw-rotated non-solid gates. It does not add camera
perception, physical obstacle avoidance, moving gates, wind or real-flight claims.

Run 007 remains an independent experiment. Its current accepted model is copied
once to run 008's immutable `source.zip` before any seed trains. The source hash
is recorded. Later run 007 updates cannot change run 008's initialization.

## Observation contract: recovery-workspace-v8

All observations are float32, length 74, bounded to [-1,1]. Physics remains
120 Hz and control remains 60 Hz. Ground clearance and workspace boundaries
use the existing 0.3 m collision radius.

| Indices | Meaning |
|---|---|
| 0–45 | Exact legacy previous-gate observation, including its existing scaling |
| 46–52 | Current gate: body displacement /200 (3), unit direction (3), distance /200 (1) |
| 53–59 | Next gate, same seven fields |
| 60–66 | Previous gate, same seven fields |
| 67 | Ground clearance /20 |
| 68–69 | Lower/upper X boundary distances /120 |
| 70–71 | Lower/upper Y boundary distances /120 |
| 72–73 | Lower/upper Z boundary distances /20 |

Absent gate slots are zero. Zero-distance direction uses a denominator of 1e-9.
The workspace diagonal is below 171 m, so displacement and distance /200 do not
saturate anywhere inside the valid workspace. Legacy clipped values are retained
only for checkpoint compatibility; new fields preserve the missing information.
The feature extractor concatenates the old 39 features and all 74 observations.
Both actor and critic therefore see ground clearance, margins, motor state,
gate presence and timing instead of silently dropping them.

The networks remain two 256-unit tanh layers. The original first-layer weights
are copied into the first 39 columns, with new columns initialized to zero.
Other compatible weights are copied exactly. Deterministic initial actions match
the old policy. Its previous demonstration/imitation learning initializes this
run; run 008 performs no action-label fitting. A separate critic-only warmup uses
512 vector steps of on-policy deterministic rollouts and leaves actor weights
unchanged. With four environments that is 2,048 warmup transitions per seed,
reported separately from PPO transitions.

## Episode distribution

Each reset samples independently: 40% earlier lessons (hover and 0–12), 20%
normal broad lessons (13–20), and 40% broad lessons with a recovery reset.
Broad lessons include explicit lateral spans 8–70 m and combined 2.5–16 m gate
heights. All five recovery kinds are sampled uniformly:

- Lateral drift: lateral offset up to 20 m and sideways speed 1–4 m/s.
- Overshoot: start 2–12 m beyond the required gate, moving away at 1–4 m/s.
- Wrong height: signed altitude offset 4–10 m and vertical speed within ±2 m/s.
- Wrong heading: heading error from 90 to 180 degrees.
- Combined: all the above together.

Recovery episodes start near a randomly chosen gate, with earlier gates marked
as already passed. The original gate ordering is preserved. Position is bounded
to X/Y ±52 m and height 2.5–17 m; downward velocity near the floor and upward
velocity near the ceiling are prevented. Initial roll/pitch vary within ±0.2
radians and body rates within ±0.3 rad/s. Perturbations grow through multipliers
0.35, 0.65 and 1.0 over thirds of the per-seed PPO budget. Evaluation always uses
full strength. Gate courses allow 60/180/600 seconds for 1/3/10 gates; hover is
5 seconds. Tests establish feasibility for sampled resets, not every random draw.

## Reward and PPO

Hover retains its existing reward. Navigation uses gate-pass +10, completion
+50, failure -50, elapsed-time cost 0.2/s and squared action-change cost 0.02.
The shaping potential is `-tanh((distance_to_gate + 2*max(exit_side_distance,0))/20)`.
The reward includes `10*(gamma*next_potential-current_potential)`, with terminal
next potential zero and gamma 0.995. This gives exit-side recovery an entrance
direction signal while keeping the potential bounded. It replaces the old
distance-only progress term. No teacher determines actions during PPO.

Each of seeds 801, 802 and 803 receives 1,048,576 PPO transitions, 3,145,728 total.
Four spawned environments collect experience; rollout length is 1,024 per worker,
batch size 256, five optimization epochs, learning rate 0.00003, clip range 0.1,
target KL 0.01, gamma 0.995 and GAE lambda 0.95. Exploration log standard deviation
starts at -3. Training continues from the latest weights even when an evaluation
fails. Failed candidates do not replace the selected checkpoint. This avoids
repeatedly resetting learning to the same initializer after each failed fit.

Checkpoints and selection evaluations run every 262,144 transitions. The best
mean selection score that preserves every individually successful baseline case
is selected per seed. Selection uses four cases per group; this is a low-cost
training screen, not release evidence. If nothing qualifies, the initial model
remains selected and must still pass the final release screen.

## Held-out release gate

There are 32 groups: hover, all 21 normal gate lessons, and the five recovery
types each on steep-zigzag and broad-ten routes. Selection seeds start at
510,000,000. All checkpoint selection across all three training seeds completes
before final evaluation begins. Final seeds start at 610,000,000 and never
contribute to training or checkpoint choice. Training course seeds are sampled
from 400,000,000–499,999,999 with distinct per-worker generators.

Final evaluation uses 32 cases per group: 1,024 courses per training seed,
3,072 total. Every group must succeed at least 31/32 times, for every one of the
three training seeds. Logged recovery metrics include first required gate
reacquisition time, altitude loss, remaining gates passed and terminal outcome.
The deployed seed is chosen from selection scores before seeing final results.
There is no automatic waiver, failed-checkpoint fallback deployment or claim of
arbitrary-layout reliability.

ONNX export uses actual held-out observations and requires action error <=1e-5.
Python/browser observation parity and Python preflight checks must also pass.
Only then is a versioned ONNX file written and `active-policy.json` atomically
switched. The old manifest is saved in the run directory. The website's dynamic
policy endpoint supports the new observation contract and serves the exact
checksum-verified model; new flights pick up the qualified model without another
build. Flights already running retain their loaded session. If tests fail, the
current deployed model stays active and the report records the failure.

## Execution and evidence

```powershell
.venv\Scripts\python.exe -u -m notebooks.train_recovery_v8 --output runs/agile-curriculum-008 --deploy
```

The preflight file `runs/recovery-v8-verification.json` must exist and be copied
to the run. Source files, configuration, initializer hash, per-seed checkpoints,
selection histories and final results are retained. `status.json` identifies the
active phase and seed; `training-summary.json` separates PPO and critic counts.
Each worker writes episode summaries and a full NPZ recording every 32 episodes;
selection/final evaluation saves the first case and up to two failures per group.
Recorded recovery traces identify the initial target index. Other episodes retain
summary metrics but cannot be reconstructed as full flight paths.

`latest-resumable.zip` contains PPO parameters and optimizer state on orderly
exit; intermediate block checkpoints also survive interruption. This first runner
does not automatically resume the full multi-seed campaign or restore an exact
environment/RNG state. Use a new campaign directory when restarting; never
overwrite existing evidence. Training duration depends on simulation throughput
and is not inferred from a fixed step count.
