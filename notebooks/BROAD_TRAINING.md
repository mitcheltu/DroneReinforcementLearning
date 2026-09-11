# Run 007: broad non-solid navigation

This experiment responds to failures on large left/right gate edits. It starts
from the actual deployed run 006 candidate, `level-9-round-3/model.zip`, rather
than run 006's protected fallback. The deployed ONNX file is not changed by this
runner. No automatic deployment occurs.

## Coverage

The new lessons are learned in the order below. All coordinates are metres.
Gate labels remain ordered, and crossing must be forward through the aperture.
Frames are non-solid; ground and workspace limits still apply.

| Lesson | Gates | Distribution |
| --- | --- | --- |
| 13, lateral small | 3 | Alternate left/right by 8–16 m, forward jitter ±3 m |
| 14, lateral large | 3 | Alternate left/right by 20–35 m, forward jitter ±3 m |
| 16, backtracking | 3 | Alternate forward/backward positions ±span/2, span 15–45 m, lateral jitter ±6 m |
| 17, independent headings | 3 | Horizontal leg lengths 10–40 m, independent full-circle gate headings, heights 2.5–16 m |
| 18, steep zigzag | 3 | Horizontal legs 12–40 m, alternating heights 2.5–4.5 and 12–16 m, independent headings |
| 15, lateral extreme | 3 | Alternate left/right by 40–70 m, forward jitter ±3 m |
| 19, near coincident | 3 | Horizontal separation 0.5–3 m, height jitter ±0.25 m, independent headings; frame overlap allowed |
| 20, broad ten | 10 | Horizontal legs 8–70 m, heights 2.5–16 m, independent full-circle headings |

Each course receives a random global yaw. Lateral courses randomize the first
left/right sign; the drone starts facing the local forward axis, so these are
actual sideways approaches rather than a forward route merely rotated in world
coordinates. Lessons 13–16 align gate headings approximately with travel
(±0.25 radians). Later lessons decouple heading from travel. Random point
proposals are bounded to local X/Y ±35 m before global rotation, keeping gate
centres within approximately ±49.5 m in world coordinates, with room for entrance
staging inside the ±60 m workspace. Timeouts remain 180 s for three gates and
600 s for ten. The initial policy and observation contract are unchanged.

The 46-value observation still clips displacement components beyond 40 m. Long
legs explicitly exercise this regime, but saturation loses distance information
and can distort bearing when multiple components clip. This run measures whether
the existing contract can learn adequate long-range behavior; it does not claim
to fix that representation limit. A failing long-range held-out group is a reason
to revise the observation contract and Python/browser parity together before
claiming broad support.

This is finite coverage, not “everything imaginable.” Exact coincident planes,
gate pitch/roll, moving gates, boundaries outside the above envelope, wind,
sensor noise, real camera detection, physical obstacles, and real-drone transfer
are not trained here. Gate rotations are yaw only. No obstacle avoidance is
introduced in this run.

## Learning and retention

The runner protects individually successful baseline cases across hover and all
13 prior lessons, including climbs, descents, mixed turns and ten-gate courses.
It uses 12 selection cases per group. Four successful-policy anchor attempts
and four teacher demonstrations per old group initialize the data; only
successful anchor flights contribute frozen policy labels.

Each new lesson gets four rounds of 24 flights. Round zero uses demonstrations;
later rounds alternate demonstrations and flights by the previous rejected
candidate, with teacher labels on the states that candidate actually visited.
Every round starts fitting from the accepted checkpoint and keeps accumulated
correction data. This fixes the previous mismatch where “DAgger” repeatedly
visited the protected starting policy's states instead of the failed fit's.

Every round also collects one new teacher flight and one policy correction
flight for each old and promoted lesson, on separate training seeds. Batches
contain 256 frozen anchor examples and 256 examples sampled uniformly across
available lesson pools. Fit learning rate is 0.00003, with 40 epochs per round.
Each flight supplies at most 512 evenly spaced observation/action pairs.
Imitation steps are reported separately from PPO transitions.

Promotion requires every individually successful old baseline case to remain
successful and 12/12 successes for the current and previously promoted new
lessons. A blocked lesson no longer terminates the whole campaign: its data is
retained and the next lesson is attempted from the protected checkpoint. A
blocked lesson is not silently marked learned. Final release requires all eight
new lessons promoted, so an unresolved early block still prevents release.

After promotion, 2,048 critic preparation steps precede 8,192 PPO steps, with
rollback if retention checks fail. Maximum configured new PPO work is 65,536
transitions, plus up to 16,384 critic preparation transitions. Many simulated
transitions also occur during demonstrations and evaluation; these must not be
reported as PPO training. The job may perform no PPO if no fit is promoted.

## Evaluation and artifacts

Old training seeds start at 360,000,000; new training seeds at 362,000,000;
retention corrections at 390,000,000, with documented level/round/case offsets
in the source. Selection seeds start at 361,000,000 and final evaluation at
370,000,000. These final seeds are different from run 006's evaluation seeds and
are never used for labels. PPO uses the shared runner's 265,000,000–266,000,000
training range. Course generators and runner sources are snapshotted per run.

Final evaluation covers all 22 groups with 32 cases each (704 courses). Both the
protected accepted model and the last fitted candidate are evaluated separately;
their results are explicitly separate in `report.json`. Release requires at
least 95% per group, which means at least 31/32 here, plus all lesson promotions.
This is a development screen, not a statistical reliability certification.

`status.json` shows the current phase; the console log records episode outcomes
and fit losses. `history.json` records promotions and rollback decisions.
`trajectories.jsonl` retains every completed attempt for montage reconstruction.
First-case NPZ files per evaluation group retain full trajectories. Anchor and
per-lesson NPZ files retain training labels when the runner exits normally or
handles an interruption. A forced process kill cannot guarantee final dataset
saving. Source snapshots, configuration and checkpoint hashes identify the run.

Launch from the repository root using a fresh output directory:

```powershell
.venv\Scripts\python.exe -u -m notebooks.train_broad --output runs/agile-curriculum-007
```

Before launch, `notebooks/test_broad_maneuver.py` and `notebooks/test_maneuver.py`
passed 31 tests. These include 192 seeded generator checks, 16 teacher flights
across the eight new lessons, five existing vertical teacher flights, directional
crossing/non-solid collision checks and individual-case retention logic. Teacher
success establishes feasible demonstrations; it is not evidence that the neural
policy has learned them.
