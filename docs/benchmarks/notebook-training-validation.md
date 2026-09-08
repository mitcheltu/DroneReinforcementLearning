# Notebook training delivery validation

Date: 2026-09-06. Platform: local Windows, CPython 3.11.14, CPU Torch 2.14.0, NumPy 2.4.6, Gymnasium 1.3.0, Stable-Baselines3 2.9.0. This report describes local execution, not a Kaggle-hosted run or a trained racing release.

## Automated checks

`python -m pytest -q`: **75 passed** in 66.04 seconds. This includes the 67 existing shared-contract tests, six simulator tests and two notebook/pipeline tests. Two expected deprecation warnings came from Torch's legacy ONNX exporter; export itself passed.

- Gymnasium's environment checker passed.
- Exact hover equilibrium and mixer torque signs passed.
- 30 seeds at each of seven stages produced 210 deterministic, geometrically valid courses with correct ordered labels and gate counts.
- Gate direction and aperture-margin scoring checks passed.
- Reference-controller hover succeeded on 20/20 starts.
- 2,000 random transitions retained finite bounded observations and normalized quaternions; recorded state/command lengths matched.
- A 256-transition smoke run resumed to 384 with its campaign count preserved.
- Actual raw sampled actions, clipped actions and applied commands matched in retained failure traces.
- Diagnostic HTML playback and ONNX export passed.
- Notebook nbformat validation and every code cell's Python syntax passed. After the input-layout update, notebook structure/syntax was checked again.

`ruff check training notebooks/build_bundle.py`: passed. Configured strict mypy checks: passed for the four contract/CLI files. The newly added numerical modules are not claimed to be covered by strict mypy.

The notebook's actual input-setup cell was executed against both a zipped mock dataset and an already-extracted mock dataset. Both produced the expected source tree and preserved source archive. Linux dependency resolution with hashes succeeded against the pinned CPU wheel; Linux/Kaggle installation has not been executed locally.

## Sustained PPO execution and continuation

Command: `python -m training.learning.train --output runs/notebook-validation --steps 32768`.

Used the real settings: 2,048 steps per rollout, batch size 512, up to ten PPO epochs, two 256-unit Tanh layers in actor and critic, one environment, CPU, seed 101, total schedule budget 10,000,000 transitions. Completed 32,768 transitions and saved the periodic checkpoint. Optimization statistics stayed finite and TensorBoard recorded rollout and optimization metrics.

Continuation into `runs/notebook-validation-resumed` completed another 2,048 transitions: global count **34,816**, PPO update counter **170**, frontier **0**, and frontier count **34,816**. The learning-rate schedule continued from the global count. TensorBoard output went to the new session directory.

Observed rollout collection rates were approximately 173–305 transitions/second. Other validation processes ran concurrently during part of this measurement, so it is an indicative local range, not an isolated benchmark or a Kaggle estimate. Use the notebook's actual runtime logs to plan session length. Ten-million-transition completion may require multiple sessions.

The early policy had not achieved hover success in these training statistics and had not reached the first promotion evaluation at 131,072 transitions. No learning convergence, gate-racing success, stage promotion or final holdout qualification is claimed from this short run.

## Independent course-control baseline

`python -m notebooks.verify_reference` used the geometric reference controller, not PPO:

- Stage 1, one straight gate: **20/20 successes**.
- Stage 3, three ordered straight gates: **20/20 successes**.

This verifies that the simulator's initial gate tasks can be completed by a controller. It does not establish learned-policy performance on randomized ten-gate courses.

## Export and diagnostics

The 34,816-transition actor passed ONNX structural checking and comparison against Torch on 512 reachable observations. Maximum absolute command error: **4.172325134277344e-07**, below the 1e-5 threshold. The export is explicitly marked experimental and not release qualified.

An actual retained training failure was rendered to `runs/notebook-validation/failed-attempt.html`. The offline player includes time controls, position, orientation, drone arms and trajectory. Automated checks verified that animation frames are present; no manual browser visual acceptance or browser physics parity is claimed.

Current trace format, retention behavior, resume limitations and remaining full-release work are documented in `notebooks/README.md`. The complete approved implementation documents remain the release acceptance authority.
