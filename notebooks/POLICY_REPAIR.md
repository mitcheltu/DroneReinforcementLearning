# Locally trained replacement flight policy — September 7, 2026

## Result and scope

The replacement neural policy passed 64/64 hover flights and 64/64 stage-1
single-gate flights on independent validation seeds. The damaged PPO policy
had passed 0/64 on both tasks in the earlier diagnosis. The replacement also
passed the single hover and gate courses used for the original case-0 replay.

Additional generalization tests are recorded in
`runs/imitation-broad/report.json`. These use 16 independent cases per harder
stage, including three- and ten-gate courses. They are exploratory checks,
not the original curriculum's repeated 64-case promotion or release tests.

| Stage | Task | Independent successes |
| --- | --- | --- |
| 0 | Hover | 64/64 |
| 1 | Single gate | 64/64 |
| 2 | Single gate with more variation | 16/16 |
| 3 | Three straight gates | 16/16 |
| 4 | Three gates with turns and height changes | 16/16 |
| 5 | Ten gates, intermediate variation | 16/16 |
| 6 | Ten gates, maximum configured variation | 16/16 |

Total: 208/208 independent test flights passed. This excludes the 32 selection
flights and the two successful same-course diagnostic comparisons.

All reported flights use deterministic neural actions, exact simulated-state
observations, and the unchanged original physics, rewards, collisions and
course generator. No real-drone or camera-input claim is made.

## Files to use

- `runs/imitation-repair/iteration-0/model.zip`: replacement SB3 neural policy.
- `runs/imitation-repair/iteration-0/actor.onnx`: deterministic actor with feature
  preprocessing included; input float32 `[batch, 40]`, output `[batch, 4]`.
- `runs/imitation-repair/report.json`: independent hover/gate validation and
  training history.
- `runs/imitation-broad/report.json`: harder-course results and terminal reasons.
- `runs/imitation-repair/demonstrations.npz`: 7,740 labeled observations/actions.
- `runs/imitation-repair/train_imitation.executed.py`: exact source used for this
  training run, matching the hash in its report.
- `runs/imitation-repair/single-gate-replay.html` and `ten-gate-replay.html`:
  standalone offline 3D replay players. Open in a browser and press Play.
- NPZ traces in the validation, comparison and broad-test directories can also
  be imported into the existing local flight viewer.

The ONNX export agreed with PyTorch on 512 sampled observations, with maximum
absolute action error `8.344650268554688e-7`, below the `1e-5` export threshold.

## What changed

This is a **new imitation-trained policy**, not a continuation of the damaged
PPO weights. The reference controller supplied labels during training only;
it is not called during neural flight evaluation or ONNX inference.

The actor has two 256-unit Tanh hidden layers. Its feature extractor derives
12 values solely from the existing public 40-value observation: body velocity,
body gravity direction, relative control target and gate/hover heading. For a
gate, the control target is two metres beyond the gate centre along its normal,
so the drone crosses the plane rather than stopping at the centre. Hover uses
its start reference without the gate offset. The features scale velocity by
5 m/s and target distance by 10 m. No global position or hidden simulator
variable is supplied to the actor.

Tests verify that translating and rotating a course preserves these features
and the corresponding reference actions. Course placement and heading are
still randomized by the original generator; gates remain fixed during an
episode and must be passed in order.

Training used 16 hover and 16 single-gate reference flights. Every third
transition contributed its observation and reference action, plus two nearby
state examples. The nearby examples add Gaussian position noise (standard
deviation 0.25 m), velocity noise (0.35 m/s) and body orientation noise (0.10 rad
per Euler component), then recompute both the observation and reference action.
These synthetic states are training examples, not claimed successful flights.

The actor trained for 150 epochs with Adam, learning rate 0.001, batch size 512,
mean squared action error, and gradient norm clipping at 1.0. Seed was 2029.
Final training action MSE was approximately 0.00006440. This loss was not used
as a substitute for flight evaluation.

The script supports optional DAgger rounds: collect labels at states visited
by the learner and retrain on the aggregated dataset. **No DAgger round was
needed for this artifact**: the first trained actor passed both 16-case
selection sets, then both independent 64-case validation sets.

Training reset seeds start at 20,000,000, with separate iteration/stage offsets.
Iteration selection uses seed base 40,000,000. Final validation uses seed base
60,000,000. Broader stage tests use 80,000,000. Within evaluation, the actual
reset seed is `base + stage * 100000 + case`. Validation and broader-test
rollouts are never used to generate training labels.

## Run locally

From the project root, with the existing Python environment:

```powershell
.venv\Scripts\python.exe -m notebooks.evaluate_policy --checkpoint runs/imitation-repair/iteration-0 --output runs/my-policy-evaluation --stages 0 1 2 3 4 5 6 --cases 16 --seed 90000000
```

Choose a new empty output directory each time. Results include per-case
outcomes, passed-gate counts, episode duration and reward. The evaluator saves
the first two traces of each outcome in each stage, including failures.

To train another replacement from scratch:

```powershell
.venv\Scripts\python.exe -m notebooks.train_imitation --output runs/imitation-new --iterations 5 --epochs 150 --episodes 16
```

Five is the maximum number of training passes, not a requirement to use all
five. Early stopping selects the first actor that passes both selection sets.
The final independent validation is run after selection; failure there is
reported rather than silently used for further fitting. The original executed
source is retained separately; later source changes add explicit parameter
validation and clearer provenance fields without changing this training recipe.

## Continuing reinforcement learning

The policy is ready for deterministic local simulation. It is **not an exact
PPO resume checkpoint**: the critic was not trained, the PPO transition counter
is zero, and there are no PPO optimizer updates to resume. The saved action
standard deviation is `exp(-3)`; stochastic flight with that exploration noise
has not been validated. Do not feed this artifact into the original campaign's
`--resume` path or assume deterministic success means safe exploration.

Further PPO work needs an explicit actor transfer, critic preparation, balanced
retention training and protected evaluations. The original unsuccessful PPO
recovery experiment remains documented in `RECOVERY.md`; its conclusion does
not describe this successful replacement actor.

The existing source bundle includes the training and evaluation scripts. The
same commands can run through the Kaggle notebook's `run_module` helper after
runtime setup, but local deterministic inference needs no Kaggle connection.
