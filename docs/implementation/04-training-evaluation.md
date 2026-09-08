# 04 — Training, evaluation, continuation and export

## 1. Dependency and execution contract

Use CPython 3.11 for the reference training environment. Required libraries: NumPy, PyTorch, Gymnasium, Stable-Baselines3, TensorBoard, ONNX, ONNX Runtime CPU, pytest, jsonschema and PyYAML only if YAML is used for human-facing launch overrides. JSON remains the shared configuration authority. Optuna is excluded from the first implementation; conduct bounded, recorded experiments instead of uncontrolled searches.

During milestone M0, resolve the latest non-prerelease mutually compatible releases available for Python 3.11 and commit a fully pinned lock with hashes. Install only from that lock thereafter. If Kaggle's preinstalled Python differs, record it and create/test a separate Kaggle lock for that supported interpreter; never overwrite Kaggle's system Python. Both environments must pass the same fixtures and export checks. Version resolution is an explicit implementation setup task, not permission to use floating dependencies in a delivered notebook. Do not install development `master` branches because documentation happens to use them.

Training entrypoint is an importable package plus `python -m training.scripts.train`. It must work without a GUI, browser, display server or rendering context. Multiprocessing entrypoints use an `if __name__ == '__main__'` guard. Imports must not launch workers or training.

## 2. PPO baseline

| Parameter | Selected value |
|---|---|
| Algorithm | Stable-Baselines3 PPO |
| Policy | MlpPolicy; separate actor and critic MLPs |
| Hidden layers | actor [256,256], critic [256,256] |
| Activation | Tanh |
| Orthogonal initialization | enabled |
| Initial log standard deviation | -1.0 for all four actions |
| Device | CPU baseline; CUDA only after a measured benefit |
| n_steps | 2048 per environment per rollout |
| n_envs | selected by document 05 throughput procedure from 1,2,4,8 |
| batch_size | 512 |
| n_epochs | 10 |
| gamma | 0.999 |
| gae_lambda | 0.95 |
| clip_range | 0.20 |
| clip_range_vf | None |
| normalize_advantage | True |
| ent_coef | 0.0 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| target_kl | 0.03 |
| use_sde | False |
| stats_window_size | 100 |
| Initial campaign budget | 10,000,000 total transitions per root seed |
| Learning rate | explicit global schedule below |
| VecNormalize | disabled |

All supported n_envs yield rollout sizes divisible by 512. `num_timesteps` counts total environment transitions across environments, not calls to vector step and not physics ticks. One ordinary transition contains two physics ticks; early terminal transitions may contain fewer. Track both counters separately.

Gamma 0.999 at 60 Hz has a nominal exponential discount timescale of about 16.7 seconds, not a guarantee of long-range planning. GAE remains 0.95; these are separate mechanisms. Record all effective settings, including PyTorch thread counts and any library defaults affecting reproducibility.

Learning rate at optimizer update start is `3e-5 + (3e-4-3e-5)*max(0,1-S/B)` where S is global campaign transitions already collected and B is the campaign budget. Implement this from global counters, not a per-notebook-chunk progress fraction. A continuation must not restart the schedule. If the budget is extended, record a new campaign segment and its explicit schedule; do not retroactively change B for the old segment.

## 3. Training phases and observability

Before training, run simulator tests and Gymnasium's checker plus SB3's environment checker. Run 10,000 random-action transitions to exercise failures/resets; this is not expected to complete tracks. Then execute a 32,768-transition PPO smoke run and verify finite losses, checkpoints, replay files and resume behavior.

Initialize a campaign at curriculum stage 0. Task promotion follows document 03. Training samples stochastic actions; evaluation and browser deployment use deterministic actor means followed by the same clamp/map. No exploration noise is added in browser flight.

Every completed attempt logs a row including run/episode/env IDs, seed, task stage, start/end transition counts, policy revision range, simulated duration, actual physics ticks, outcome, gates passed, reward sum and component sums, peak/mean speed, minimum clearance, mean mixer lambda, saturation fraction, observation clipping counts, and recorder status.

Every optimizer update logs global transitions, rollout size, collection wall time, optimization wall time, total wall time, transitions/s, policy loss, value loss, entropy, approximate KL, clip fraction, explained variance, learning rate, action standard deviations and curriculum counters. Log metrics to JSONL as the durable source and TensorBoard for convenient viewing. Flush at every update; console prints one concise update summary at least every 60 wall seconds when the process is active.

Use a thin, tested PPO collector extension to associate pre-clamp samples, values and log-probabilities with the actions sent to each environment. Do not independently resample actions for logging. Hash actor weights at each optimizer update and increment `policy_revision`. Episodes can span several updates; record the revision per action. A sampled training attempt must not be mislabeled as produced entirely by the last checkpoint.

## 4. Checkpoints and interruption

Save an atomic resume checkpoint at each completed optimizer update after crossing a multiple of 32,768 total transitions. Also checkpoint at a graceful wall-time stop and before/after a curriculum reward-mode switch. A checkpoint occurs after optimization so no partial on-policy rollout must be resumed. Write temporary files in the same filesystem, fsync/close, verify, then rename. Keep two latest resume checkpoints plus the best development checkpoint and the pre-speed checkpoint. Never overwrite the sole valid resume state before replacement is verified.

Checkpoint directory contains:

- `model.zip`: SB3 policy/critic/optimizer state using its save format.
- `campaign.json`: campaign/root seed, global transitions, optimizer revision, budget and LR schedule, checkpoint ID, counters, stage frontier, success streaks, stage transition counts, reward mode, best-model comparison values.
- `rng.json` / binary payload: Python random, NumPy stream, PyTorch CPU and CUDA RNG states as applicable; serialization format is versioned.
- `environment_streams.json`: each environment's course/reset/task RNG state and episode counter. Active episodes are deliberately not resumed.
- `recorder.json`: reservoir priorities/indexes, retained file hashes and archive counters.
- Exact shared configurations, dependency lock, code commit, hardware metadata, and file hash manifest.

On continuation, recreate environments, restore stream states/counters, discard previously active episodes with reason `collector_restart` in the attempt index, and begin new episodes. Load optimizer/policy and restore global schedule/curriculum. Log a new session ID under the same campaign. Use SB3 continuation semantics (`reset_num_timesteps=False`) and verify counters in a dedicated test. Restoring stream state does not imply bit-identical training: reset active episodes, hardware/library variation and worker scheduling prevent that claim.

No PPO replay buffer is required: PPO is on-policy. Trajectory archives are visualization/diagnostic data, not an off-policy training dataset.

The training loop checks its wall-time budget at each completed update. Default Kaggle process budget is 10 hours from process start; leave the rest of the session for setup, checkpointing, packaging and output saving. The smoke/throughput stage must establish that one update fits the remaining safety margin. A graceful stop finishes its current update, checkpoints and exits with status 0 and `stop_reason=wall_budget`. Hard interruption may lose work since the last checkpoint.

## 5. Validation and model selection

Every 131,072 transitions, after the nearest completed optimizer update, pause collection and evaluate an immutable snapshot. Do not mutate training RNG, normalization, curriculum episodes or weights during evaluation. Evaluation case sets are fixed artifacts, with fresh environment instances and no stochastic actions.

Run current-stage 64-case promotion suite. At stage 6 also run the 512-case development suite. Earlier stages run a fixed 16-case subset of that full suite for visualization only; those 16 cases do not determine promotion. Full-suite evaluation cadence and snapshot step must be explicit in logs.

Best development checkpoint is selected lexicographically by: (1) number of complete courses, larger wins; (2) total ordered gates passed over all cases, larger wins; (3) median finish time among completed courses, lower wins; (4) earlier training step wins. With zero completions, finish time is +infinity. Do not select by mean training return.

Final release candidate must finish >=95% of the 1,000 untouched standard cases. Report numerator/denominator and Wilson 95% interval. This is a point-estimate release threshold, not a claim that the confidence interval's lower bound exceeds 95%. Each of the three training seeds must separately achieve >=90% on the development suite for the reproducibility milestone. If free compute cannot support the three runs, deliver a single-run experimental model and explicitly leave that milestone incomplete; do not claim a reproducible release.

Final-test cases are evaluated only after selecting a candidate from development results. Failures block standard release. Diagnose on development data; if final cases informed subsequent design, rotate the final set as document 03 requires. Experimental/stress cases are reported separately with no success threshold.

## 6. Evaluation report

Persist per-case data and an aggregate JSON/Markdown report. Required metrics:

- Course completion count/rate and Wilson interval.
- Ordered gate passage count divided by total scheduled gates (include failed episodes).
- Outcome counts: success, gate frame, ground, workspace, timeout, state_limit; infrastructure errors separately invalidate the run.
- Completion-time median and p90 over successful cases only, with successful sample size.
- Time-to-failure median for failures; do not mix it with completion times.
- Mean and minimum gate crossing clearance; maximum speed and control saturation.
- Breakdown by maximum course turn, height change and approach-angle bins: [0,10], (10,20], (20,30] degrees for turns; [0,0.5], (0.5,1], (1,1.5] m for height; [0,5], (5,10] degrees for approach. Include sample sizes and mark empty bins.
- Training transitions and wall hours to selected checkpoint, hardware and dependency versions.

No crashed flight counts as successful because it passed several gates. A valid gate 10 completion is success according to the temporal event rule in document 02.

## 7. ONNX export

Load the selected checkpoint on CPU, set eval mode and export only deterministic actor computation: observation -> actor feature extractor -> actor MLP -> action mean linear layer. Exclude critic, Gaussian sampling, value and log-probability outputs. Do not append Tanh unless the trained actor actually uses it at its output. Clamp remains an explicit postprocessing step in both runtimes.

Use ONNX opset 17, fixed batch one, input `obs` float32[1,40], output `action` float32[1,4]. The implementation must explicitly select and test the PyTorch ONNX export API supported by the pinned lock; no implicit exporter change during dependency updates. Export must succeed in the locked Kaggle environment. Validate the graph with onnx.checker and ONNX Runtime CPU before browser tests.

Test at least 2,048 observations: 1,024 recorded real observations including failure/boundary states, and 1,024 deterministic schema-bound synthetic observations. Compare raw actor means and clamped commands to PyTorch. CPU ONNX maximum absolute error <=1e-5, RMS <=1e-6. For browser backends use document 08 limits. Postprocessing must map all-negative/zero/all-positive and out-of-range commands identically.

Export `actor.onnx`, shared metadata, hashes and model manifest from document 01. Never export normalization statistics that were not used: v1 uses explicit scales, no VecNormalize. If normalization changes later, it is a new schema/model release.

## 8. Controlled experiments

If learning stalls for three consecutive stage evaluations, save diagnostics and compare one change at a time on the same development cases. Allowed experiment dimensions are gamma (0.997,0.999), initial action log std (-1.5,-1.0,-0.5), progress coefficient (0.25,0.5,1.0), and controller gains scaled by (0.5,1.0). These are experiments with unique config hashes, not runtime options in one release. A controller/physics change requires regeneration of physics fixtures and retraining; it cannot be applied only in the browser.

Keep a campaign table of hypothesis, changed value, root seed, transitions, wall time, stage reached and evaluation result. Do not respond to reward improvement alone by promoting difficulty. Exhausting the budget without release performance is a measured training limitation, not technical proof of impossibility or success.
