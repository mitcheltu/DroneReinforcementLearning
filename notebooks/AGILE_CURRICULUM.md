# Broad-layout retraining

The original deployed actor was trained by imitation on hover and single-gate flights. Its broader tests used the original generator: 6–8 m spacing, at most 30-degree incremental turns and 10-degree gate-heading jitter. Rotating/translating an entire course is not the same as independently rotating a gate or reversing the route.

## New experiment

`train_agile.py` uses a new neural actor, demonstration initialization, DAgger corrections when needed, critic preparation and genuine PPO updates. Inference takes the existing 40-element observation and produces four normalized actions. The training expert is not called during neural evaluation or ONNX inference.

The 25-feature extractor retains body velocity, angular velocity, gravity, current and next gate displacement/heading, and previous actions. It no longer merges the current gate center and normal into a fixed point beyond the gate. This gives the actor information needed to learn entrance-side repositioning and turns. All gates remain upright; rotations here are yaw, not pitch/roll.

| Level | Lesson | Gate count | Horizontal spacing | Incremental route turns | Heading offset from route |
|---|---|---:|---|---|---|
| 0 | Gentle | 1 | 6–10 m | None | ±15° |
| 1 | Wide spacing | 1 | 12–35 m | None | ±20° |
| 2 | Turns | 3 | 7–12 m | Up to 60° | ±15° |
| 3 | Sharp turns | 3 | 9–16 m | Up to 120° | ±25° |
| 4 | Reversals | 3 | 10–22 m | 150–180°, either direction | ±20° |
| 5 | Independently rotated gates | 1 | 8–25 m | None | ±180° |
| 6 | Mixed | 3 | 10–30 m | Up to 180° | ±180° |
| 7 | Ten-gate mixed | 10 | 8–25 m | Up to 180° | ±180° |

Each course also randomizes global heading, starting drone heading, altitude and successive altitude changes. Centers stay within X/Y ±45 m and Z 3–14 m, and at least 5 m from earlier centers/start. Rejection sampling enforces those conditions, with a finite 1,000-proposal budget. These checks do not prove that every generated route is traversable or free of obstructing non-target frames.

The entrance-side training expert stages behind the gate, routes around a frame when approaching from its exit side, then crosses forward. Its yaw request avoids the reference attitude controller's singular zero correction at 180°. It is imperfect: the initial independent 16-flight diagnostic contained failures on reversal and ten-gate cases. Demonstration collection and final evaluation retain failures rather than hiding them.

## Progression and retention

The first local run uses 16 flights per lesson/round, 150 supervised epochs, up to three collection/fitting rounds per lesson, and 8,192 PPO transitions after a lesson passes selection. Supervised fitting retains the original 7,740 labeled hover/single-gate observations and aggregates new examples. Every third transition supplies a label plus one nearby perturbed-state label. Later rounds alternate expert and learner rollouts; the expert labels learner-visited states (DAgger).

Promotion requires all four fixed selection cases to succeed in each checked group: hover, gentle flight, the preceding lesson and the current lesson. These are development checks, not statistical release evidence. At later levels they do not exhaustively retest every earlier lesson; final evaluation covers every level.

PPO uses a 60% current-level mixture and 40% uniform mixture over hover and levels through the current level. It uses the original reward/collision model, learning rate 2e-5, clip range .05, target KL .005, gamma .995, 512-step rollouts, batch size 256 and three epochs. Exploration starts at exp(-4). Before PPO, the value network alone is prepared on 2,048 deterministic rollout transitions. The actor is protected by post-PPO selection checks; loss of measured success restores pre-PPO policy weights and clears PPO optimizer state.

If a lesson fails all permitted fitting rounds, escalation stops. The last curriculum-promoted model is selected, not the last failed fit. A separate final seed namespace evaluates all requested levels and compares the old deployed model on exactly the same layouts and time budgets. A run must not be called successful merely because it performed PPO updates or reduced action loss.

Training seeds begin at 120,000,000, selection at 130,000,000, PPO at 140,000,000 and final comparison at 160,000,000, with distinct level/round offsets. Seed 2031 controls initialization and fitting. New output directories preserve old models and checkpoints.

## Time budgets and browser compatibility

The experiment explicitly allows 60 seconds for one gate, 180 seconds for three, and 600 seconds for ten. The original simulator/browser limits remain 12/20/45 seconds. Both models receive the new budgets in the comparison, so the result is not an old-timeout versus new-timeout comparison. A browser release of a long-course policy must also support its advertised timeout; copying an ONNX file alone does not establish parity.

Custom short-course artifacts use experimental mode with null generator provenance. They are accepted by the browser editor's additive schema, not the unchanged original Python curriculum schema. The original training/config fingerprint stays unchanged. The run's executed source copies define the new curriculum.

## Run and inspect

```powershell
.venv\Scripts\python.exe -m pytest notebooks/test_agile.py -q
.venv\Scripts\python.exe -m notebooks.train_agile --output runs/agile-curriculum-001 --episodes 16 --epochs 150 --rounds 3 --ppo-steps 8192 --max-level 7 --final-cases 8
```

Use a new output directory for another run. This runner is not the original PPO campaign resume interface. Final comparison with eight cases per group is a development screen; broader independent testing remains necessary before making a high-reliability claim.

Artifacts:

- `configuration.json`: ranges, budgets and original fingerprint.
- `train_agile.py`, `agile_curriculum.py`: executed source snapshots.
- `history.json`: collections, fit loss, selection outcomes and accepted/rejected PPO updates.
- `level-N-round-R/model.zip`: fitted policy before PPO; `ppo-candidate.zip`: policy after PPO, including rejected candidates.
- `accepted-model.zip`: latest promoted policy; `last-model.zip`: final fit, possibly unaccepted; `interruption-model.zip`: final cleanup snapshot. Only the accepted policy is eligible for release consideration.
- `labels.npz`: aggregated supervised dataset, saved on normal completion or interruption.
- `trajectories.jsonl`: every completed collected episode, downsampled to at most 601 display poses with final pose preserved, course, lesson, seed, source, outcome and reward. Synthetic perturbed labels are not fabricated flight trajectories. In run 001, source `ppo` includes deterministic critic-preparation rollouts as well as stochastic PPO rollouts; it must not be interpreted as a count of optimizer transitions.
- Selection/final folders: full first-case NPZ recordings for each level.
- `report.json`: final model selection, per-case old/new comparison, PPO transition count and release-screen result.

The original ONNX asset is preserved until the replacement passes evaluation and export/runtime checks. Arbitrary user layouts can still be infeasible, outside training coverage, obstructed by other frames, or longer than a selected timeout. Exact state observations are assumed; this experiment is not camera navigation or real-drone validation.

## September 8 experiments and current implementation

The preceding sections describe experiment 001. They do not describe every later run. Each run retains its executed Python sources; use those snapshots to reproduce its behavior.

| Run | Observation / extracted features | Change | Final screen |
| --- | --- | --- | --- |
| 001 | 40 / 25 | Expanded layouts, imitation and guarded PPO | Did not qualify; 0/8 mixed ten-gate routes |
| 002 | 46 / 31 | Previous-gate observation and a waypoint-based teacher | Did not qualify; teacher's hidden waypoint phase was not observable by the actor |
| 003 | 46 / 31 | Memoryless teacher; independent headings precede reversals | Finished; promoted through level 5, did not qualify |
| 004 | 46 / 39 | Gate-entry braking, segment-based previous-gate avoidance, geometric features and balanced labels | 8/8 in each shorter-course group, 5/8 mixed ten-gate routes; did not qualify |
| 005 | 109 / 138 | Full other-gate observations and collision-aware roadmap demonstrations | Training; see [OBSTACLE_TRAINING.md](OBSTACLE_TRAINING.md) |

Runs 003 and 004 use levels 0 gentle, 1 wide spacing, 2 turns up to 60 degrees, 3 sharp turns up to 120 degrees, 4 independent headings, 5 reversals, 6 mixed three-gate routes, and 7 mixed ten-gate routes. Gates are still traversed in array order, numbered 1 through the course's gate count. Course position and heading are randomized at reset; they do not move during an episode. Heading rotation here means yaw about the vertical axis. These experiments do not train tilted gates with arbitrary pitch or roll.

The 46-value observation preserves the original first 40 values and appends the previous gate's displacement divided by 40 and unit normal, both in body coordinates. Before passing the first gate, the six appended values are zero. Run 004 additionally derives eight scalar features from this observation: normal, sideways and vertical position and normal velocity relative to both current and previous gates. This adds no privileged teacher state. The exported actor performs the same extraction internally.

Run 004 transfers the fitted checkpoint `runs/agile-curriculum-003/level-3-round-1/model.zip`. Its new feature columns start with zero weights. It starts learning at level 4, retains all original imitation labels and collects eight additional expert episodes from each of levels 1, 2 and 3. Each new lesson collects 32 episodes per round and fits for 100 epochs, with at most four rounds. Each collected episode contributes at most 512 evenly selected labels, including perturbed labels. Collection stops at 300 seconds; final evaluation retains the 60/180/600-second course budgets. A completed episode's displayed trajectory remains available even when its labels are subsampled.

The run 004 teacher only commands gate crossing when on the entrance side, within 0.65 metres of the normal axis and below 0.5 metres/second of lateral speed. Previous-gate avoidance checks whether the desired path segment intersects the previous gate plane near its frame. This avoids unnecessarily returning toward the previous gate on distant maneuvers. Initial teacher diagnostics passed 4/4 sharp turns, 4/4 independent headings, 4/4 reversals, 4/4 mixed three-gate courses and 3/4 mixed ten-gate courses. The remaining ten-gate course timed out after eight gates. These are demonstration results, not learned-policy results or held-out release evidence.

Final evaluation seed bases are 180,000,000 for run 003 and 200,000,000 for run 004. The earlier 160,000,000 suite is now development evidence. Four-case promotion checks and eight-case final screens are too small to establish general reliability. The runner only attempts PPO after its imitation candidate passes promotion selection; PPO is not currently used to discover a maneuver when imitation remains blocked. A failed PPO update is rolled back. The selected checkpoint may therefore be an imitation policy with earlier accepted PPO updates.

The generator rejects courses with another gate center within 3.5 metres of a straight course leg. This does not prove that detour paths around rotated gates are unobstructed. The actor sees current, next and previous gate information, not a map of all other frames. Success on the generated suite must not be described as support for every possible editor layout.

### Inspect training overlays

Run `.venv\Scripts\python.exe -m notebooks.build_curriculum_montage` to publish a snapshot, then open `http://127.0.0.1:3000/curriculum`. Select a run, lesson and rollout phase. All matching attempts are overlaid, with opacity, shared playback time and optional start alignment. This snapshot does not refresh itself when training writes more episodes. Run 004 distinguishes `critic_preparation` from stochastic `ppo` in raw records; the viewer groups both under PPO and critic preparation. Demonstrations, learner corrections, policy evaluations and the original-policy comparison remain separately selectable.

When new per-lesson files are generated after `next start`, restart the local production server so Next.js recognizes them. Existing file contents can change without introducing a new path. The September 8 snapshot contains 1,305 attempts across four runs and was checked in the browser after restarting.

Replaying the seven initial failed level-5 demonstrations from run 004 reproduced all seven frame collisions. Every case had passed gate 1 and was targeting gate 2; the nearest frame at collision was gate 3. A straight-leg center-clearance test therefore missed an obstacle on the maneuver detour. `notebooks.diagnose_agile_failures` saves terminal state, target label and distances in each gate's local coordinates. Its `--version v5` option tests an experimental next-gate avoidance teacher against the same failed courses. This teacher is separate from the running v4 experiment and is not a released policy. Later source revisions to the v4 runner also include terminal state, target index and gates passed directly in future JSONL records; these fields are absent from the already-running run 004's source snapshot.

### ONNX release status

`notebooks.export_agile` exports a completed run's selected policy to that run's `candidate.onnx`, checks ONNX structure and compares outputs on recorded evaluation observations. It does not publish the candidate. Browser observation parity and long experimental replay support have been added, but the active flight worker still loads the original 40-input actor and uses original flight budgets. Switching to a 46-input model requires explicit worker integration, runtime validation and a passing candidate; copying the file into the existing asset path is insufficient.

The agile runners currently depend on local demonstration files and specific transfer checkpoints under `runs`. They are not standalone Kaggle notebooks. Preserve these input artifacts with the executed source snapshots when moving training to another machine.
