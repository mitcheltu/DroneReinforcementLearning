# Start training

**September 7 update:** a replacement neural policy has been trained and run
locally, passing 64/64 hover and 64/64 single-gate validation flights. See
[POLICY_REPAIR.md](POLICY_REPAIR.md) for the model, replay and exact commands.

**Original campaign status:** the supplied campaign lost hover after entering
stage 1. Follow [RECOVERY.md](RECOVERY.md) for the measured findings and separate
bounded recovery experiment. The original campaign instructions below describe
the existing trainer; do not use them to blindly extend this failing campaign.
That PPO continuation recipe did not demonstrate recovery.

Use **AeroRL_Kaggle_Training.ipynb**. This experimental pipeline collects real training experience while browser implementation continues.

## Kaggle steps

1. Build the upload bundle from the repository root: `.venv/Scripts/python.exe notebooks/build_bundle.py`. Its ready-made output is `dist/aerorl-kaggle-source.zip`.
2. Create a private Kaggle dataset containing that zip. Keep the filename unchanged and attach it using Add Input. Automatically extracted source files are also supported. If several source inputs are attached, set `SOURCE_INPUT` in the first code cell to the intended zip or extracted folder. Do not upload `.venv`, `node_modules`, or the entire workspace.
3. Import `notebooks/AeroRL_Kaggle_Training.ipynb`. Set Internet on and accelerator None. Setup creates an isolated Python 3.11.14 CPU runtime.
4. Run setup, simulator checks and the training/resume smoke cells. These must pass before real training.
5. Leave `SESSION_STEPS=1_000_000`, `TOTAL_BUDGET=10_000_000`, `SEED=101`, `N_ENVS=1`, and `RESUME=None` for the first campaign. A complete final rollout can overshoot the requested count by 2,047 transitions with one environment.
6. Run training, curves, replay and export cells. Download `/kaggle/working/aerorl-run-001.zip` and save a notebook version with outputs before ending the session. The runtime installation is excluded from the archive.

The Linux lock uses a pinned [official CPU Torch wheel](https://download.pytorch.org/whl/cpu/torch/). Internet is required for setup, not simulation. A Kaggle-hosted run has not been performed on your account; the notebook smoke test is the final environment check.

## Resume

Attach the previous notebook output or extract your run archive into a private input dataset. Set `RESUME` to a directory containing **model.zip, rng.pt, campaign.json**. Do not point it at model.zip itself. Keep the source dataset, total budget, seed and environment count unchanged. Choose a new output directory for the session. Source/config fingerprints must match; the original source zip is preserved in each run.

Resume restores the policy, optimizer, global transition count, learning-rate progress, curriculum and RNG streams. Active episodes restart and unfinished rollouts are discarded; continuation is not bit-identical to uninterrupted execution. Old metrics/traces remain in the old archive; the new directory contains new session records. Retention counters carry forward. Load only your own trusted checkpoints because checkpoint loading uses Python serialization.

Atomic checkpoints are saved every 32,768 transitions and at exit, keeping the newest two per output directory. The nine-hour allowance is checked between rollouts; evaluation can extend it. Abrupt shutdown can lose work since the previous checkpoint. KeyboardInterrupt is handled when delivered to the training process, but a notebook process kill is not guaranteed to deliver it. Preserve outputs after each session.

## Local commands

```powershell
uv sync --frozen --extra learning
.venv/Scripts/python.exe -m pytest training/tests/test_simulator.py training/tests/test_training_pipeline.py -q
.venv/Scripts/python.exe -m notebooks.verify_reference
.venv/Scripts/python.exe -m training.learning.train --output runs/first --steps 1000000
```

For continuation, add `--resume` followed by the actual checkpoint directory reported by the previous run and use a new output directory. The path is also recorded in `latest.json`. For a fast wiring check use `--smoke --steps 256` in a separate directory; smoke settings cannot resume a real campaign or qualify a pilot.

## Curriculum and randomization

At each reset, the runner samples a stage, then separate course and initial-state seeds. Frontier 0 means all hover. Otherwise 80% select the frontier and 20% select uniformly among earlier stages. PCG64 randomness and full course/reset seeds are retained in diagnostics. Gates remain fixed during an attempt. Labels follow the route; only the current gate scores.

| Stage | Gates | Maximum turn | Height change | Gate yaw jitter |
|---|---:|---:|---:|---:|
| 0 | 0: hover | — | — | — |
| 1 | 1 | 0° | 0 m | 0° |
| 2 | 1 | 0° | 0.25 m | 5° |
| 3 | 3 | 0° | 0 m | 0° |
| 4 | 3 | 15° | 0.5 m | 5° |
| 5 | 10 | 20° | 0.75 m | 5° |
| 6 | 10 | 30° | 1.5 m | 10° |

The generator samples 6–8 m horizontal segment lengths, permitted height/heading changes, global yaw and feasible workspace translation. Gates have fixed 2.5 m openings. It rejects invalid frame bounds, overlaps, excessive heading drift, invalid approaches and route crossings. Initial position, velocity, attitude and angular rate are perturbed by stage. The first height change is capped at 0.25 m. Ten-gate tasks finish after gates 1–10 in order.

Every 131,072 transitions, 64 fixed deterministic cases test the frontier. At least 61 successes twice consecutively and 131,072 transitions actually sampled at the frontier are required for promotion. Failure resets the streak. Stage 6 retains the reliability reward and does not claim release qualification. Relative gate observations support adaptation within this distribution; generalization still requires holdout measurements.

## Recorded data

The diagnostic format is `aerorl-notebook-trace-v1`, compressed NPZ, loaded with `allow_pickle=False`. It is separate from the future browser `.aerorl.zip` contract.

| Array | Shape | Contents |
|---|---|---|
| states | N × 18 | seconds, position XYZ, quaternion WXYZ, world velocity XYZ, body rates XYZ, four actual rotor thrusts |
| commands | K × 7 | seconds, four commanded thrusts, desaturation scale, applied duration |
| observations | M × 40 | normalized policy inputs |
| actions | M × 10 | start/end seconds, four clipped actions, collective newtons, target rates XYZ |
| raw_actions | M × 4 | actual sampled actions before clipping, training only |
| policy_update | M | PPO optimizer update counter, training only |
| rewards | M | scalar reward totals |
| terminal_observation | 40 | final normalized observation |
| metadata_json | string | full course/seeds, events, stage/outcome/episode metrics |

N=K+1 and normally K=2M, with an earlier terminal event permitted in the final action. Physics uses float64. The diagnostic writer currently stores observations as float64 after list conversion; values originate from float32 policy inputs. Playback includes gate outlines, physical drone arms/orientation, path and time controls in offline HTML. It subsamples to approximately 600 frames while NPZ retains every physics tick.

Retention saves the first two completed attempts per million-transition/stage/outcome bucket while the run trace directory is below 1,000,000,000 bytes before writing. The cap can be exceeded by one trace. This selection favors early examples; use every-episode `episodes.jsonl` and fixed-case `evaluations.jsonl` to measure performance. Evaluation previews retain the latest four cases per stage and can overwrite older previews.

## Implementation boundary

Implemented: coupled float64 RK4, motor lag, rate control, 120 Hz physics / 60 Hz policy, swept frame/workspace collision, ordered passes, finite-horizon termination, seeded course rejection, 40 observations, PPO, automatic promotion, actual training failures, checkpoints/resume, TensorBoard, 3D diagnostic playback, and ONNX export with parity checks.

Remaining full-release requirements: TypeScript dynamics parity and browser flight UI; full browser artifact manifests; recorded reward components; reservoir and queued retention; exact policy-weight revision archives; multiprocess benchmarking; best-model/pre-speed preservation; speed phase; final holdout and multi-seed qualification. Hover departure and speed/rate bounds are checked at physics endpoints. Course yaw jitter is currently sampled interleaved with segment proposals. These are explicit experimental-delivery differences, not a declaration that the complete v1 contract has passed.

See `docs/benchmarks/notebook-training-validation.md` for local validation. Kaggle throughput and learning convergence must be measured in the actual campaign.
