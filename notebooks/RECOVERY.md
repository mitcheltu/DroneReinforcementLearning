# Recovery of the supplied 3-million-step campaign

**Update:** a replacement imitation-trained actor now passes independent hover
and single-gate validation. See [POLICY_REPAIR.md](POLICY_REPAIR.md). The results
below describe the earlier unsuccessful attempt to continue the damaged PPO
policy, which remains unchanged.

## Status and evidence

Do not continue the original campaign for millions of additional steps yet. The
recovery runner is a bounded experiment; it does not claim to have repaired the
policy or validated the seven-stage curriculum.

**Completed local recovery experiment:** 16,384 new transitions from the
3,000,320-step policy, split equally across hover and stage 1, still produced
0/64 successes on each skill. Gate outcomes were 59 timeouts, 4 frame
collisions and 1 ground collision. Mean hover return changed from -3.43925 to
-3.46459 and mean gate return from -50.45951 to -50.50300. The protected best
remained the starting checkpoint. This does not establish that a longer run
could never work; it provides no demonstrated recovery to justify a large
additional training budget. Hold off on another long Kaggle run with this
recipe. Local results are in `runs/recovery-controlled/summary.json` and
`evaluations.jsonl`. A prior 4096-transition plumbing pilot also had no
successes on its smaller eight-case evaluation.

Next development should establish a reproducible successful hover policy,
protect that checkpoint, and validate the transition into one-gate training
while retaining hover. A reference-controller imitation warm start is one
candidate to investigate; it has not been implemented or validated here.

The supplied 3,000,320 and 3,151,872 checkpoints both match the current original
training/shared fingerprint:
`44af04cb5396c679bedb50d96a1ae1d7c8102d5c22bdea4b231de77730c54044`.

Local deterministic evaluations used 64 identical courses per stage for both
models, with reset seeds `10001 + stage * 100000 + case`, case 0 through 63.

| Checkpoint steps | Hover successes | Hover mean duration | First-gate successes | First-gate outcomes |
| --- | --- | --- | --- | --- |
| 3,000,320 | 0/64 | 1.928 s | 0/64 | 58 timeout, 5 frame collision, 1 ground collision |
| 3,151,872 | 0/64 | 1.431 s | 0/64 | 26 timeout, 36 ground collision, 1 frame collision, 1 workspace exit |

These are evaluations of the saved policies, not stochastic training episode
statistics. The final checkpoint is later than the last evaluation in each
supplied log, so those logged counts need not match this table. Hover was
already lost in the 3-million-step checkpoint. The extra 150,000 steps made
behavior worse, but did not initiate the regression.

The original reset lottery allocates 80% of episodes to the frontier and 20% to
earlier stages. Short failed hover episodes made this only approximately 3.5%
of transitions for hover in the continuation. Original promotion evaluations
check only the frontier; they do not catch forgetting of hover. The original
runner also retains only its newest two checkpoints, not a protected best
policy. These are confirmed process defects; they do not prove that sampling
alone explains every policy failure.

## What the recovery runner changes

`python -m notebooks.recover_training` is separate from the original trainer.
It leaves all original training modules, configuration bytes and supplied
checkpoints unchanged.

1. Two environments run together: one always samples stage 0 and one always
   samples stage 1. Each vector step adds one transition to each skill, exactly
   50% each regardless of episode length. Execution uses CPU DummyVecEnv;
   this batches policy inference but does not parallelize physics across cores.
2. Course and initial-state randomization still occur at every reset using the
   original generator and stage configuration. Gates stay fixed during an
   episode. Stage 1 has one gate, with randomized course placement and heading;
   this experiment does not train ten gates or promote to stage 2.
3. Initial migration copies the actor, critic and learned action standard
   deviations from the input. It intentionally creates a fresh Adam optimizer
   and resets the recovery transition counter to zero. Original training steps
   are retained separately in `original_steps`. This is a new recovery run,
   not a bitwise continuation of the original PPO optimizer.
4. Learning rate is fixed at 0.00003, PPO clipping at 0.1, target KL at 0.01,
   and optimization epochs at 3. Each rollout has 2048 transitions per
   environment, 4096 total; batch size is 512. Gamma, GAE lambda, value loss
   weight and network architecture come from the input model. Entropy weight
   is zero and gradient norm limit is 0.5. These are conservative experimental
   settings, not established optimal settings.
5. Evaluate both skills before training and after each 16,384 new transitions
   (or at the end of a smaller requested budget). Use 64 cases each by default.
   Any decline in either success count relative to the protected best stops
   the run. At a zero-success baseline this guard cannot distinguish worsening
   failures; inspect outcome types, rewards and traces as well.
6. Among non-regressing candidates, more combined successes wins; combined
   mean return breaks ties. `best.json` points to that checkpoint. `latest.json`
   points to the most recent candidate, including a rejected candidate. Always
   inspect `summary.json`; latest does not imply safe or best.
7. Keep every evaluation checkpoint in the new output directory. No checkpoint
   pruning occurs. Evaluation traces retain the first four cases of each stage
   at each evaluation. Existing training trace sampling and episode logging
   are retained. TensorBoard logs PPO metrics.
8. A score of at least 61/64 for both skills stops with status
   `candidate_ready_for_independent_validation`. It is not promotion or proof
   of recovery: test fresh seeds and repeat retention checks before expanding
   the curriculum. `validated_recovery` stays false.

## Kaggle cell

This cell makes the experiment reproducible; it is **not** a recommendation
to spend more training time on the unsuccessful recipe above.

Rebuild/upload the new `dist/aerorl-kaggle-source.zip`, then run the existing
notebook's source extraction and Python environment setup cells. Confirm the
new extracted source contains `notebooks/recover_training.py`. Replace the
original training cell with this cell; do not also run the old training cell.
This uses the existing notebook's `SOURCE`, `PYTHON`, `ENV`, and `run_module`.

```python
import json
from pathlib import Path

RECOVERY_SOURCE = Path(
    "/kaggle/input/datasets/mitchelkevintu/dronerl-results3/"
    "aerorl-run-003/checkpoints/step-000003000320"
)
RECOVERY_OUTPUT = Path("/kaggle/working/aerorl-recovery-001")
assert (SOURCE / "notebooks/recover_training.py").is_file(), "Upload/extract the new source bundle"
assert (RECOVERY_SOURCE / "campaign.json").is_file(), "Check the attached dataset path"

run_module(
    "notebooks.recover_training",
    "--source", RECOVERY_SOURCE,
    "--output", RECOVERY_OUTPUT,
    "--steps", "32768",
    "--cases", "64",
)
print((RECOVERY_OUTPUT / "summary.json").read_text())
for line in (RECOVERY_OUTPUT / "evaluations.jsonl").read_text().splitlines():
    entry = json.loads(line)
    print({key: entry[key] for key in
           ("steps", "successes", "mean_rewards", "outcomes", "rejected")})
```

The 32,768 transitions are an experimental upper bound, not a recommendation
to keep extending a failed run. A regression can stop it earlier. Evaluation
also consumes wall time, but does not count toward training transitions.
`--steps` must be a positive multiple of 4096. Case counts below 64 are only
for plumbing tests and cannot produce the validation-ready status.

The loader accepts ordinary `model.zip`, an extensionless ZIP named `model`,
or Kaggle's extracted `model/` directory. For the directory form it rebuilds
an archive temporarily and leaves the dataset unchanged. Only load your own
trusted model checkpoints, as the underlying model loader uses serialization.

The output must be empty or new. To resume an already assessed recovery
checkpoint, supply its checkpoint directory as `--source`, add `--resume`,
and choose another empty output directory. Use the same code, seed (default
2027) and case count. Resume restores optimizer and RNG; active episodes
restart. Input provenance and source fingerprints are checked. Absolute
pointers inside a copied dataset may be stale: pass the actual mounted
checkpoint directory rather than blindly trusting a copied `best.json` path.

Keep the entire output directory, especially `summary.json`, `best.json`,
`evaluations.jsonl`, `episodes.jsonl`, `checkpoints`, and trace directories.
The current runner saves after completed evaluation blocks; interruption
during a block leaves the previous complete checkpoint available. A partial
directory is not a resumable checkpoint. The original notebook's subsequent
campaign/export cells expect the original campaign format: do not run them
on recovery output without adapting them.

## Local diagnostics and verification

Run `python -m notebooks.diagnose_checkpoints` from the project root to repeat
the supplied-checkpoint comparison. Results and replayable NPZ traces go to
`runs/recovery-diagnostics`; `baseline-source.zip` records the original source.
Import the diagnostic NPZ files into the local 3D viewer to inspect failures.

Run `python -m pytest notebooks/test_recovery.py -q` to check fixed transition
allocation across auto-resets, regression detection and non-mutating model
archive reconstruction. These tests verify safeguards, not learning success.
