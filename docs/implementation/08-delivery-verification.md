# 08 — Delivery plan, verification and release

## 1. Repository to implement

```text
README.md
pyproject.toml
requirements.lock
requirements-kaggle.lock
notebooks/train_kaggle.ipynb
shared/
  vehicle.v1.json
  observation.v1.json
  course-rules.v1.json
  training.v1.json
  schemas/{course,state,model,replay,config,worker-message}.schema.json
  fixtures/{physics,observations,actions,events}/
  courses/{promotion,development,final-test,stress,public}/
training/
  __init__.py
  physics/{quaternion,dynamics,controller,collision}.py
  envs/{quadrotor_env,observations,rewards,course_generator,course_validation}.py
  recording/{recorder,binary,bundles,reservoir}.py
  learning/{ppo_collector,curriculum,checkpoint,schedules}.py
  scripts/{train,evaluate,export_onnx,benchmark,generate_cases,generate_fixtures,package_run}.py
  tests/
web/
  package.json
  package-lock.json
  next.config.ts
  public/{models,courses,ort}/
  src/app/{layout,page}.tsx
  src/components/{DroneCanvas,DroneMesh,GateMesh,GateEditor,Transport,Telemetry,ReplayPanel}.tsx
  src/physics/{quaternion,dynamics,controller,collision}.ts
  src/simulation/{environment,observations,events,courseValidation}.ts
  src/inference/{session,manifest,actionMapping}.ts
  src/workers/simulation.worker.ts
  src/replay/{importer,binary,timeline,comparison}.ts
  src/schemas/
  tests/{unit,integration,browser}/
docs/
  implementation/
  benchmarks/
  model-cards/
.github/workflows/{ci,release-checks,deploy-preview}.yml
```

Python directories require `__init__.py` as needed for normal packages. Filenames in braces represent separate files. Generated runs, checkpoints, local .env files, caches and node_modules are gitignored. Fixtures and small hashed course sets are committed. Final-test data must be inaccessible to routine training code paths even if present in a release-evaluation checkout; training launch validates its allowed inputs and never opens that directory.

No live application implementation exists at documentation delivery. Build artifacts, model cards and benchmark numbers are created only by the subsequent implementation/testing work.

## 2. CLI contract

All CLIs support `--help`, exit 0 on success/graceful wall-budget completion, 2 on invalid configuration/input and 1 on runtime/test failure. Paths are explicit; no script depends on the shell's current directory beyond resolving a supplied relative path. Structured result summaries always include status and output manifest path. Never print a passed status when quality failed.

| Command | Required interface and result |
|---|---|
| `python -m training.scripts.generate_cases` | `--config PATH --partition promotion|development|final-test|stress|public --output DIR`; deterministic artifacts/manifest; refuses overwrite unless `--replace-generated` explicitly supplied |
| `python -m training.scripts.generate_fixtures` | `--config PATH --output DIR`; creates numerical fixture manifest; CI compares existing fixtures and does not regenerate expected answers automatically |
| `python -m training.scripts.benchmark` | `--config PATH --output DIR`; document 05 candidates/results; separate benchmark seed |
| `python -m training.scripts.train` | `--config PATH --output DIR --root-seed INT --wall-budget-seconds INT`, optional `--resume CHECKPOINT_DIR`, `--mode smoke|train`; new run refuses a nonempty output without compatible resume |
| `python -m training.scripts.evaluate` | `--checkpoint DIR --cases DIR --output DIR --purpose development|release|stress`; checks allowed partition, deterministic actor and report |
| `python -m training.scripts.export_onnx` | `--checkpoint DIR --evaluation DIR --output DIR`; produces experimental export unless a passing release-purpose evaluation and required metadata exist |
| `python -m training.scripts.package_run` | `--run DIR --output FILE`; verifies and bundles selected artifacts without wheelhouse/duplicate caches |

Config PATH names an explicit resolved JSON launch manifest referring to shared files, dependency lock, selected n_envs/backend, campaign budget and hashes. Notebook parameter overrides write a new resolved manifest before invoking the CLI; no ad hoc CLI values silently replace physics constants. Public-course generation produces 12 stage-6 cases from namespace 50001. Replace the first case with a stage-3 straight generator configured for ten gates so the default course is straight; validate it against standard rules and label it standard.

Stress generator `stress-generator-v1` uses the document 03 algorithm with N=10, T=45 degrees, H=2 m, J=15 degrees and cumulative heading bound 90 degrees. Retain height [3,10], spacing [6,8], global transform, proposal limit and reset perturbations from stage 6. At validation, apply only experimental basic-validity rules and require at least one failed supported constraint; reject otherwise. Produce 100 cases with namespace 40001. Label all experimental and never mix them into training/development/final standard cases. Stress performance has no release threshold.

NPM scripts: `dev`, `build`, `start`, `lint`, `typecheck`, `test:unit`, `test:integration`, `test:browser`, `benchmark`. CI uses npm ci followed by these named checks. Browser test runner must support Chromium and Firefox locally/CI; macOS Safari evidence may require a separate host and must be recorded.

## 3. Ordered implementation milestones

### M0 — Contracts and reproducible setup

Implement shared JSON values/schemas from these documents, dependency locks, configuration validators and directory scaffolding. Add `start_reference_m` to CourseV1. Record exact toolchain versions. Check current security advisories for the selected Next.js/React and Python dependency releases. Create a requirements traceability checklist linked to section 10 of document 09.

Exit: both languages load the same configuration; schema rejects unknown/missing fields; clean installs and imports succeed; configuration hashes are reproducible. No training is needed.

### M1 — Physics and low-level controller

Implement quaternion/matrix operations, mixer, motor lag, RK4 and pure Python dynamics. Add reference-controller hover/straight flight. Generate independent analytical/convergence fixtures; implement TypeScript equivalents and cross-language comparison. Build the simplest 3D scene capable of displaying fixture state without ONNX.

Exit: numerical/unit tolerances below pass; hover reference completes 20 perturbed 5-second cases; straight three-gate reference completes at least 19/20 collision-free cases at <=3 m/s reference speed. Retain failed reference cases. Failure blocks PPO work until sign/geometry/control issues are understood.

### M2 — Environment, courses and replay first

Implement ordered progression, swept events, task timeout, reward segmentation, observation contract, deterministic generator, validation and partitioned cases. Implement actual episode recorder/binary bundle and browser replay importer/player before multi-million-step training. Add course editing without policy flight.

Exit: 10,000 seeded random-action transitions without nonfinite states; generator acceptance tests; all event/reward tests; Python-recorded terminal failure replays correctly in browser; exactly ten gate UI ordering and all constraints pass tests.

### M3 — Training and Kaggle lifecycle

Implement PPO configuration, actual-sample collector logging, curriculum, global LR schedule, checkpoints/resume, metrics and notebook cells. Run throughput benchmark and smoke training in a fresh Kaggle session. Resume in a separate session from preserved outputs. No model quality claim is needed for this milestone.

Exit: smoke/continuation counters and schedule verified, one stochastic failure retained with correct action provenance, durable Kaggle outputs verified, one ONNX smoke export with parity tests. A random or smoke actor is clearly labeled untrained/experimental.

### M4 — Reliable learned policy

Run the campaign; diagnose stalled curriculum using development data. Reach stage 6, preserve reliability checkpoint, optionally refine speed. Select candidate through development suite. Complete three-seed reproducibility development results or explicitly classify delivery as single-run experimental.

Exit for standard model: document 04 release thresholds pass and artifacts/model card are complete. Budget exhaustion without thresholds is an incomplete quality milestone, not a reason to publish invented metrics.

### M5 — Browser autonomous flight

Integrate actor loading and explicit backends, strict worker timing, events, UI state machine and failure handling. Run fixed observations and closed-loop cases through WASM and WebGPU. Connect edited supported courses to the same observation/physics pipeline. Add telemetry, replay comparison and production deployment configuration.

Exit: browser numerical parity, course outcome comparison, supported edit/reset behavior and reference performance targets pass. Slow devices show reduced real-time factor without changing the simulation-time control contract.

### M6 — Release validation and handover

Run untouched final test on selected candidate, assemble model card, validate production preview, test import limits and browser matrix, verify output checksums and roll back once in preview. Publish only measured claims.

Exit: release checklist below is complete with artifact links. There is no fixed calendar guarantee; track milestone completion and actual effort rather than treating the historical eight-week estimate as acceptance evidence.

## 4. Numerical unit tests and tolerances

All fixed numerical tolerances are absolute unless relative is explicitly stated. Test data includes zero, nominal and near-boundary values. Tolerances are not to be loosened merely to accommodate a sign or convention mismatch.

| Test | Required result |
|---|---|
| Quaternion identity/axis rotations/inverse | vector error <=1e-12 in float64 |
| R orthogonality and determinant | max abs(R^T R-I)<=1e-12, abs(det(R)-1)<=1e-12 |
| q and -q | same transformed vectors within 1e-12 |
| Hover equilibrium | ideal level hover, exact mg/4 motors, zero commands error: position drift <=1e-8 m over 10 s |
| Free fall analytical | test-only drag=0, motors=0, no collision termination: position error <=1e-9 m at 1 s; test config explicitly differs from release |
| Motor step response | compare f(t)=fc+(f0-fc)exp(-t/tau); max error <=1e-4 N over 0.3 s at dt=1/120 |
| Mixer inverse | reconstruct feasible collective/torque <=1e-12 |
| Rotor signs | individual thrust causes tx/ty/tz signs from table |
| Mixer saturation | commands in [0,4], collective conserved <=1e-10 N, differential vector scales by recorded lambda |
| Quaternion norm | error <=1e-12 after every tick in 10,000-step bounded command test |
| Timestep convergence | smooth unsaturated trajectory vs dt=1/1920 reference: dt=1/240 total state error norm < dt=1/120 error; additionally estimate fourth-order behavior on pure smooth dynamics without controller switching |
| Python/TS one tick | each state component max absolute error <=1e-10 on golden cases |
| Python/TS 1,200 ticks fixed commands | position <=1e-6 m, velocity <=1e-6 m/s, rotation <=1e-6 rad, omega <=1e-6 rad/s, motors <=1e-8 N |
| Observation parity | max absolute feature error <=1e-6 after float32 casting |
| Action clamp/map parity | output error <=1e-7 for float32 inputs |

Generate physics parity fixtures from a deterministic bounded sequence of body-rate/collective commands and no event termination, plus separate fixtures with motor saturation and event boundaries. Do not compare two implementations of the same incorrect formula as the only verification: analytical, sign, and convergence tests provide independent checks.

## 5. Geometry, episode and reward cases

Required cases: centered forward pass; backward crossing; start on plane; wrong-order crossing; missing aperture without frame contact; recovery from positive side by flying around and recrossing; grazing margin; touching frame; high-speed traversal through a thin frame; colliding with a previously passed gate; floor and each workspace face; final pass before collision; collision before final pass; same-fraction tie; nonfinal pass then collision in one tick; timeout coincident with final passage; reward target switch; no following gate; mask clearing after gate 10; partial-tick termination; step-after-done; invalid initial overlap; and numerical-error propagation.

Expect exact labels/target indices/event order. Event times between runtimes differ <=1e-8 s. Same-time final valid passage beats timeout, but collision beats passage. State_limit is evaluated at the integrated endpoint after earlier geometric events; no retrospective failure invalidates an earlier final finish. Reward component sums agree within 1e-6 per transition and 1e-4 per complete episode.

Reward exploit tests: stationary hovering outside a gate has negative racing return over an interval; moving away produces negative progress; moving away and returning to the same position for the same gate has zero undiscounted distance progress; switching gates produces no arbitrary distance jump; out-of-order passages yield no gate bonus; collision is not marked success; timeout cannot earn finish bonus. These tests do not prove all RL exploits absent; development completion remains primary.

Generator: produce 10,000 stage-6 courses across fixed test seeds, zero invalid accepted courses, no exhaustion, and report acceptance/proposal histogram. If this chosen rejection generator fails the exhaustion requirement, revise/version its algorithm and course sets before training; never silently fallback. Check independent streams, deterministic reset, no partition duplicates ignoring IDs, and all editor validator reason codes.

## 6. ML/export/browser parity

Run 2,048 observation corpus checks from document 04. Browser raw-action maximum absolute difference <=1e-4 and RMS <=1e-5 against PyTorch for WASM and WebGPU. Shape and output names must match exactly. Clamp/action mapping is checked separately. Failure blocks use of that backend/model combination.

Closed-loop deployment comparison uses 200 fixed development cases in Python and each supported browser backend with no wall-time truncation. Browser completion must not be more than 2 percentage points below Python on the same cases; record paired outcomes and gate counts, not just aggregate rate. Long closed-loop positions are not required to remain identical, because small floating-point action differences can accumulate. Investigate every difference crossing the release threshold.

Test worker generation races: pause during inference, reset during inference, load course during inference, model failure, hidden tab, rapid repeated Run, device loss and resumed WASM. No stale action may mutate a new generation. Test a deliberately delayed inference: simulation-time transitions remain exactly two physics ticks and real-time factor drops visibly.

## 7. CI levels

Every PR: locked install, Python lint/type checks, unit tests, environment checker, small deterministic PPO smoke (4,096 transitions), export/parity smoke, TypeScript lint/type/unit/integration, replay round-trip, production Next build and Chromium WASM smoke. Keep CI fixtures/model small and explicitly experimental. Do not run long training or final-test model selection on ordinary PRs.

Release checks: full numerical corpus, development closed-loop parity, browser matrix/performance, actual saved Kaggle execution evidence, three-seed results, untouched final test, replay import/retention tests and dependency/license scan. Export/checkpoint/config changes trigger complete model compatibility checks. Pure documentation changes run links/contracts checks only.

CI writes test reports and artifacts. It must not automatically regenerate expected fixtures, edit thresholds, publish a model or deploy production on a failed check. Preview deployment can be automatic after build/tests; production promotion requires a passing release bundle and the project's authorized release action.

## 8. Model card and operational handover

Model card states task, exact observation/action meaning, known simulator state (no camera perception), upright fixed gate constraints, stage curriculum, training seeds/steps/hardware, selected checkpoint, reward mode, development and untouched test methodology, measured results/intervals, browser results, numerical tolerances, failure examples and excluded capabilities. Include source/config/lock/artifact hashes and licenses.

README provides local setup, test commands, Kaggle new-run/continuation steps, output preservation, browser launch, course editing, replay import and known limitations. Troubleshooting maps symptoms to specific checks: spinning/wrong tilt -> quaternion/mixer tests; falling -> collective/motor mapping; missing gates -> event normals/order; browser-only failures -> observation/config/backend parity; low throughput -> CPU/env benchmark; reset jumps in replay -> recorder below autoreset; resumed LR jump -> global schedule restore.

Backup/recovery instructions identify verified checkpoints and saved Kaggle versions. Release rollback switches application and model/config bundle together, never just actor.onnx. No real-world flight use is supported.

## 9. Final release checklist

- [ ] All shared schemas, constants and compatibility IDs match this contract.
- [ ] Python and TypeScript numerical/event/observation tests pass.
- [ ] Actual Kaggle smoke, training, cross-session resume and export evidence exists.
- [ ] Actual stochastic failed attempt plays at its recorded terminal state in browser.
- [ ] Ten ordered gates, recovery, finish and editor behavior pass UI tests.
- [ ] Development/model-selection data is separate from untouched final evaluation.
- [ ] Candidate >=95% on 1,000 final cases; intervals and per-case results reported.
- [ ] Three seeds each >=90% development completion, or release explicitly remains experimental.
- [ ] ONNX/backend/closed-loop parity and desktop performance requirements pass.
- [ ] Import limits, local-file privacy, error recovery and browser matrix are verified.
- [ ] Checkpoint, actor, course, replay, lock and config hashes verify.
- [ ] Production preview and rollback are tested.
- [ ] README/model card contain measured facts with no placeholder success numbers.
