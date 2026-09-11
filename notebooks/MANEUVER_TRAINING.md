# Non-solid gate maneuver training

Run 006 starts from the preserved run 004 accepted checkpoint. Run 005 and both existing browser ONNX assets remain available. This is a new experiment, not a continuation of the failed obstacle-learning fit.

## Collision and traversal contract

Gate frames are visible but do not collide in the new Python environment or live browser flights. Ground, workspace boundaries and vehicle state limits still terminate flights. A gate pass still requires crossing its aperture forward with the existing 0.35-metre opening margin. Crossing a bar or missing the opening does not pass the gate. Crossing backward does not pass it. Only the current ordered gate advances, so overlapping/copanar gates may require leaving and re-entering the plane; one crossing does not automatically award multiple gates.

Training rejects gate centers closer than five metres. Overlapping gates are not part of this curriculum. The browser permits edited layouts, but non-solid frames do not guarantee that the selected model can navigate them.

Original `training/` contracts and the default low-level browser environment remain unchanged for historical tests. The new Python subclass uses an empty gate collision world; the live worker explicitly selects non-solid mode. Saved historical recordings still depict their original physics. Both currently selectable browser models were trained with solid frames, so their new non-solid flights are comparisons under changed conditions, not newly retrained maneuver policies.

## Lessons

Earlier lessons 0–7 retain their random positions/headings and include hover, gentle and wide gates, 60/120-degree turns, independent headings, reversals and mixed three-/ten-gate courses. Five additional lessons are learned in this order:

1. Descents: three gates, start altitude 10–13 metres, each gate 1.5–2.5 metres lower.
2. Climbs: three gates, start altitude 3–5 metres, each gate 1.5–2.5 metres higher.
3. Alternating heights: three gates, alternating decreases and increases of 1.5–2.5 metres from a 5–10-metre start.
4. Descending turns: descents combined with the 120-degree-turn course generator.
5. Ten-gate 3D maneuvers: randomized ten-gate routes and headings with alternating altitude changes.

Heights are bounded to 2.5–14 metres. Gate headings remain yaw only, and layouts stay stationary throughout each episode. The teacher stages three metres on the entrance side at the gate's altitude, then crosses when lateral position error and lateral speed are both below 0.5 in their respective units. There is no obstacle planner or hidden waypoint queue. The policy keeps the 46-value observation and existing 39-feature extractor, allowing exact checkpoint transfer.

## Retention and optimization

The starting policy is evaluated on eight fixed cases per old lesson, including hover. Every individually successful old case becomes a mandatory retention check, rather than merely protecting an aggregate success count. Old baseline failures do not make promotion impossible, but all old lessons receive new teacher examples too.

For each old lesson the runner collects four anchor flights from the frozen starting model, retaining labels only from successful flights, plus four teacher demonstrations. Each flight contributes at most 512 evenly spaced labels. Every fitting batch contains 256 anchor examples and 256 examples sampled across the lesson pools with equal expected representation per lesson. This keeps half the batch dedicated to preserving earlier behavior even as new data accumulates. Saved `anchors.npz` and per-lesson `labels-N.npz` support inspection.

Each new lesson permits four rounds of 16 collected flights and 60 fitting epochs, at learning rate 0.0001. The initial round uses the teacher; later rounds alternate teacher and neural-policy flights, labeling both with the teacher. Each round starts again from the accepted checkpoint and retains the growing correction dataset. A candidate must preserve all required old cases and pass 8/8 cases for the new lesson and every newly learned lesson.

After successful fitting, the critic receives 2,048 preparation steps, then PPO receives 8,192 transitions sampled uniformly across all old and newly learned lessons. Any PPO update failing the same checks is rolled back. Maximum configured new PPO work is 40,960 transitions across five lessons; imitation optimizer steps and actual PPO transitions are recorded separately. PPO still only refines candidates that first pass the fitting checks.

Training courses use seed ranges beginning at 260,000,000 (old training examples) and 262,000,000 (new examples), selection uses 261,000,000 and final evaluation uses 270,000,000, with level/round offsets. Final evaluation covers hover and all thirteen gate lessons with 16 cases each. The release screen requires all five new lessons promoted and at least 95% success in every final group; with 16 cases, this requires 16/16. It is a development screen, not an arbitrary-layout guarantee.

## Run and inspect

```powershell
.venv\Scripts\python.exe -u -m notebooks.train_maneuver --output runs/agile-curriculum-006 --episodes 16 --epochs 60 --rounds 4 --ppo-steps 8192 --cases 8 --final-cases 16
```

The launched run also writes `runs/agile-curriculum-006.log`. Use a fresh output directory for another launch. The runner has no automatic resume interface. Source snapshots, starting checkpoint hash and original simulation fingerprint are retained in the run directory.

`retention-baseline.json` defines the old cases that must be preserved. `history.json` records fit counts, acceptance results and PPO rollback decisions. `report.json` appears after final evaluation and identifies both the accepted policy and last fitted candidate. These may differ. Training and evaluation trajectories are recorded in JSONL for the montage, with full first-case NPZ recordings per evaluation group. NPZ metrics explicitly identify non-solid mode. Terminal state records include velocity; full NPZ arrays include actions and the whole state trajectory.

The viewer snapshot can be refreshed with `python -m notebooks.build_curriculum_montage`. Retention-anchor demonstrations and the retention baseline are labeled separately in raw records. No new model replaces a browser asset automatically.

The edited descent layout was subsequently reconstructed from visible editor values by `notebooks/user_descent_check.py`, with displayed-coordinate rounding. Its course and diagnostic recordings are under `runs/user-descent-regression`.

## Requested experimental browser deployment

At the user's request, **Run 006 · descent experimental** is now the default browser option. It loads the actual final fitted checkpoint `level-9-round-3/model.zip`, not the protected `accepted-model.zip` evaluated in the final report. That fit passed 8/8 descent selection cases but only 6/8 mixed-three-gate retention cases, so it failed acceptance and is offered for testing only.

The asset `web/public/models/maneuver-006-round-3.onnx` has SHA-256 `99d0772831eed7534e38a7ec2c88b0836e96b6165f60604f5c76e61b2832ec78`. ONNX parity passed on 2,039 recorded observations with maximum absolute action difference 0.000000954. It receives 46 values through the browser's verified previous-gate observation implementation, uses non-solid gates, and retains the 60/180/600-second course budgets. Run 005 and the original model remain selectable for comparison.
