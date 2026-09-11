# Obstacle-aware retraining, run 005

## Purpose and starting point

Run 004 passed all eight final cases in each of hover, gentle gates, wide spacing, 60-degree turns, 120-degree turns, independent headings, reversals and mixed three-gate routes. It completed five of eight mixed ten-gate routes. The other three collided with frames. It did not qualify for release.

Seven earlier failed reversal demonstrations were replayed. In every case the drone had passed gate 1, was targeting gate 2 and collided beside gate 3. Avoiding only the previous gate was insufficient. A local next-gate patch completed five cases and stalled on two. The replacement roadmap teacher, including its slow escape from extra clearance margins, completed all seven. Those seven cases are regression evidence, not independent generalization evidence.

Run 005 transfers `runs/agile-curriculum-004/accepted-model.zip`. The checkpoint hash is recorded in `configuration.json`. Training starts again at level 5; inherited level 4 means that advanced lessons must pass the new checks even though the source checkpoint had passed the earlier mixed-three-gate screen.

## Detection and observation contract

This is exact geometric detection from the simulator's known course, not image recognition, lidar simulation or noisy real-world sensing. Every gate is known regardless of occlusion or distance. Gates have the standard fixed dimensions and yaw rotation. Arbitrary pitch/roll, moving obstacles and adjustable frame dimensions are not represented in this policy contract.

The neural actor receives 109 float32 values:

1. Values 0–45 retain the v2 observation, including the previous gate.
2. Values 46–108 contain nine seven-value slots for every gate except the current target, ordered by distance from the drone with original gate index as a tie breaker.
3. Each slot contains body-frame displacement divided by 200 (three values), body-frame unit normal (three values), and a presence mask (one value).
4. Unused slots are zero. The course has at most ten gates, so nine slots cover every non-target frame. Position normalization uses 200 metres to avoid clipping valid workspace displacements.

The extractor retains the previous 39 features and adds 11 features for each obstacle: scaled displacement, normal, presence and four geometric scalars for normal/side/vertical position and normal velocity. Absent slots are masked. Total extracted width is 138. Existing actor and critic weights transfer into the first 39 columns; new columns start at zero.

The ONNX graph includes feature extraction. The roadmap is a demonstration teacher only and is not called by neural inference. The deployed browser worker currently supports the original actor; a 109-input candidate requires a matching browser observation implementation and runtime validation before activation.

## Teacher and maneuvering

The teacher builds a visibility graph around all gate-frame bars. It expands each bar by 0.8 metres for planning, compared with the simulator's unchanged 0.35-metre collision radius. Apertures remain open. Graph nodes include points on both sides of each gate and points around its corners. Paths may go around or over a frame, or through an aperture when that is physically clear. Passing a non-target aperture does not advance the required gate sequence.

The teacher recomputes the next visible waypoint from the drone's current position and current goal. The static graph and goal distances are cached only to reduce computation; there is no hidden progress through a waypoint queue. Its goal is three metres behind the target gate until the drone is on the entrance side, within 0.4 metres of the normal axis, and below 0.4 metres/second of lateral speed. The crossing goal is then three metres beyond the gate.

If the drone has entered the extra planning margin and no normal graph connection exists, the teacher checks connections with a 0.4-metre expansion and commands at most a 0.5-metre position offset. This provides a slow escape while keeping a margin above the physical radius. If no such connection exists, it commands braking at the current position. Finite visibility graphs and imperfect tracking can still fail; teacher success is measured rather than assumed.

## Curriculum and optimization

The configured run is:

```powershell
.venv\Scripts\python.exe -u -m notebooks.train_agile_obstacles --output runs/agile-curriculum-005 --episodes 24 --epochs 100 --rounds 4 --ppo-steps 16384 --max-level 7 --final-cases 16
```

Use a new empty directory for another run. This runner does not implement campaign resume. It depends on the transferred checkpoint and original imitation demonstrations stored under `runs`.

Training retains the original hover and single-gate demonstrations and collects eight new teacher episodes for each of levels 1–4. It then learns level 5 reversals, level 6 mixed three-gate routes, and level 7 mixed ten-gate routes. Each advanced round collects 48 episodes, with at most 512 labels per episode, then performs 100 imitation epochs. Subsequent rounds alternate teacher and learner-controlled episodes, labeling learner states with the teacher. A lesson allows at most four rounds.

Levels 6 and 7 deliberately remove the earlier rejection of other gate centers close to direct course legs. Gates are still separated by at least five metres, have randomized positions and headings within the existing workspace, and are traversed in numerical order. Geometry remains fixed during an episode. Earlier lessons retain their prior generator restrictions.

Promotion requires eight successes out of eight selection cases in hover and **every** lesson through the current level, using seed base 230,000,000 plus the runner's level offsets. After that, critic preparation is followed by 16,384 PPO transitions. The same selection suite checks the update; a regressing PPO update is rolled back. Maximum configured PPO work is 49,152 transitions across the three lessons, excluding critic preparation. Most maneuver acquisition in this runner is imitation/DAgger; PPO remains a guarded refinement step.

Final comparison uses 16 cases per group at fresh seed base 240,000,000, for both the selected candidate and the original model on identical layouts and time budgets. The budgets are five seconds for hover, 60 for one gate, 180 for three gates and 600 for ten gates. Collection is capped at 300 seconds. The development release screen requires every lesson promoted and at least 95% success per final group; with 16 cases this requires 16/16. This screen is not a statistical guarantee for arbitrary layouts.

## Evidence and visualization

`runs/agile-curriculum-005.log` contains console progress for the launched run. Its directory contains configuration, executed source snapshots, checkpoints, selection history, JSONL trajectories and, when complete, the paired final report. Records include the full course, lesson, source, seed, outcome, gates passed, target index, terminal state and at most 601 display poses. First-case evaluation NPZ files retain full recordings. Label subsampling does not remove displayed attempts.

Run `python -m notebooks.build_curriculum_montage` in the project environment to update the snapshot at `/curriculum`. The builder publishes all lesson paths immediately, including lessons with zero recordings, so future snapshots can fill them. Restart a production Next.js server once after introducing a new run's paths. The viewer snapshot is not a live training monitor.

The obstacle observation, frame/opening collision checks, known reversal regression and ONNX feature parity tests are in `notebooks/test_agile_obstacles.py`. All five passed before launching run 005. Final policy evaluation and complete-actor ONNX parity remain separate requirements.

## Experimental browser trial

Run 005 completed all four fitting rounds without promotion: its last candidate passed 8/8 reversals, 7/8 moderate turns and 6/8 sharp turns. No new PPO updates ran. The final run report evaluates the protected starting checkpoint, not the failed final fit.

At the user's request, the site now offers the actual latest fit from `level-5-round-3/model.zip` as **Run 005 · latest experimental**, selected by default. The original actor remains selectable. This is a trial deployment despite failed acceptance checks, not a qualified release. The model receives all 109 observations from the browser simulator and uses the 60/180/600-second gate-course budgets. No teacher or roadmap runs in the browser.

The exported asset is `web/public/models/obstacles-005-round-3.onnx`, SHA-256 `34b1551edf6aa1e754e707209a9f56a9cac5ae2d5e9385749c61c4451a9f9ada`. Complete-actor ONNX parity passed on 1,328 recorded observations with maximum absolute action error 0.000001073. All 90 browser unit tests passed, including direct 109-input comparison with Python fixtures. Production build and targeted lint passed. These compatibility checks do not imply flight reliability on arbitrary layouts.

Live browser verification with the experimental model on **Ten gates · varied, variation 1** completed all ten gates in **32.45 seconds**. This verifies one built-in course and the browser execution path; it is not evidence that arbitrary edited courses will succeed.
