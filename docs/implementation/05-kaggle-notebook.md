# 05 — Kaggle notebook implementation

## 1. Deliverable and constraints

Deliver `notebooks/train_kaggle.ipynb` as a thin, reproducible launcher for repository modules. Physics, rewards and course generation must not be duplicated in notebook cells. The notebook must run from top to bottom in a fresh Kaggle session with the documented inputs. No Kaggle execution is claimed until a saved run and its output manifest exist.

Kaggle documentation currently describes finite CPU/GPU notebook sessions; plan for the documented 12-hour limit, but read actual session/account limits in the UI at run time. Quotas, accelerator availability and installed Python are platform facts, not constants in simulator code. No paid resource or external account is required. [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), [GPU usage guidance](https://www.kaggle.com/docs/efficient-gpu-usage).

## 2. Input and output layout

Attach one versioned Kaggle input dataset containing a source archive, exact dependency lock/wheelhouse, shared fixtures, validation cases and optional prior checkpoint. Inputs under `/kaggle/input` are read-only. Never install into or modify that directory.

Copy/extract code into `/kaggle/working/aerorl/source`; outputs go to `/kaggle/working/aerorl/runs/<run_id>/`. Reject archive path traversal and verify input hashes before extraction. Source archive must contain no credentials or large previous runs. Prefer an attached wheelhouse to network installation; when using Internet, the notebook installs the same hash-locked requirements.

Run directory:

```text
manifest.json
configs/
environment.json
logs/{updates.jsonl,episodes.jsonl,console.log}
tensorboard/
checkpoints/{latest,previous,best,pre_speed}/
evaluations/
replays/
exports/
session_summary.json
```

`run_id` is a UUID created once for a new campaign and preserved on continuation. `session_id` is a new UUID on every notebook/process invocation. CLI --resume must name a verified checkpoint explicitly; do not select an arbitrary zip from a directory.

## 3. Required notebook cells, in order

### Cell 1 — Instructions and run parameters (Markdown + configuration)

Explain new run versus continuation, attached input location, output saving, headless training and no live browser streaming. Expose only `INPUT_ROOT`, `RESUME_CHECKPOINT` (null/new or exact path), `ROOT_SEED` (101 default), `MAX_WALL_HOURS` (10 default), `CAMPAIGN_TRANSITIONS` (10,000,000 default), and `RUN_MODE` (`smoke`, `train`, `evaluate`, `export`). Other changes require a versioned config override file and hash.

### Cell 2 — Runtime preflight

Print Python, OS, CPU count/affinity, RAM, free disk, available CUDA device names and package-lock compatibility. Confirm working directory is writable with a create/read/delete of a task-owned probe file. Require at least 8 GiB free disk before full training. If insufficient, stop with instructions to remove prior task outputs or reduce archived replay retention through an explicit config version. Do not delete unrelated notebook files.

### Cell 3 — Verified source and dependencies

Verify source/checkpoint manifests, safely extract, install locked dependencies, and import every required library. If install requires a runtime restart, stop before training, tell the user exactly which cell to resume at, and verify versions after restart. Do not repeatedly upgrade PyTorch on each run. Store `pip freeze`, lock hash and interpreter version in environment.json.

### Cell 4 — Tests and input validation

Validate all JSON schemas/hashes; run fast physics, observation, gate-event and action tests, both environment checkers, and course-set disjointness checks. Any failure prevents training. New runs execute the 32,768-transition smoke path once per source/config/lock combination. Continuations may reuse a verified smoke artifact only when all three hashes match.

### Cell 5 — Throughput selection

Use a disposable benchmark model and separate benchmark seeds; never alter the campaign policy or its RNG. Measure end-to-end collection plus PPO optimization, with rendering disabled and normal logging/recording enabled. Run at least one warm-up rollout followed by three timed full updates for each candidate.

Candidates: DummyVecEnv with 1 and 2 environments; SubprocVecEnv with 2,4,8 but only when count <= available CPU affinity count. Use `spawn` as multiprocessing method. Set PyTorch intra-op threads=1 and inter-op threads=1 before work; set BLAS thread environment variables to 1 in worker launch to avoid oversubscription. Notebook launches the CLI subprocess instead of creating multiprocessing workers from a nested notebook function.

Choose greatest total transitions/total measured wall seconds, including optimization. If within 5%, choose fewer environments; if same count, choose DummyVecEnv. Benchmark CPU first. CUDA comparison is optional only when an accelerator is already enabled; select it only if >=15% faster at the same selected environment count and all outputs/tests are finite. Do not assume two available GPUs imply SB3 multi-GPU training; distributed PPO is excluded.

Persist every measurement and selected configuration. Stop if a single update exceeds 20 minutes; profile before attempting a long run. Estimate wall time as remaining transitions/measured throughput and display that it excludes variable evaluation/packaging overhead. This estimate is not a completion guarantee.

### Cell 6 — Initialize or restore

For new campaign, generate manifest and load stage 0. For continuation, verify lock/config compatibility, restore document 04 state and report checkpoint step, frontier, reward mode and remaining transitions. Reject an exhausted budget unless a new explicit campaign extension is supplied. Never silently start from random weights when restore fails.

### Cell 7 — Launch training subprocess

Run `python -m training.scripts.train` with explicit config/output/root-seed/resume/wall-budget arguments. Stream stdout to the cell and console.log. The subprocess is the only owner of workers and campaign writes. On interruption request graceful shutdown; close all worker processes and wait for file handles. Do not rely on a Python atexit handler to survive a hard Kaggle termination.

Training emits periodic summaries including transition count, stage, recent completion, throughput, elapsed wall time, latest checkpoint and archive size. Notebook plotting reads flushed files and must not lock writer files. Console heartbeats may show progress, but the notebook does not host a public web server or tunnel.

### Cell 8 — Inspect learning

Plot development completion versus total transitions, training completion split by stage, gate count, outcome breakdown, reward components, control saturation and throughput. Shade/mark curriculum and reward-mode changes. Do not join heterogeneous stage returns into an unlabeled learning curve. Provide a table of retained failed attempts with replay file paths. Static plots are sufficient inside Kaggle; interactive 3D playback happens in the web application.

### Cell 9 — Evaluate and export

Evaluate the selected development checkpoint, not automatically the most recent one. Run final-test evaluation only for an explicit release-candidate run; repeated ordinary notebook resumes must not repeatedly inspect the untouched set. Export ONNX and verify it according to document 04. CPU export requires no GPU. If quality fails, preserve outputs and label the export experimental; do not generate a passed release manifest.

### Cell 10 — Package and preserve outputs

Close log writers; verify latest checkpoint and replay manifests; generate a SHA-256 output index; write session_summary.json with completed steps, stop reason, quality status and errors. Bundle selected artifacts into `aerorl-session-<session_id>.zip`. Avoid including the source wheelhouse and duplicate checkpoint copies in the download bundle.

The notebook must visibly instruct the user to save a Kaggle notebook version with outputs retained and verify files appear in that saved version. Interactive `/kaggle/working` files alone are not a durable backup guarantee. A subsequent session attaches the saved output/version as input and selects the exact checkpoint. Session persistence may help but is not the recovery contract.

## 4. Storage and failure behavior

Limit training replay archive to 1 GiB and evaluation replay archive to 1 GiB. Check disk availability after each update; if free space <2 GiB, gracefully stop after checkpoint rather than allowing a partially written model. Keep metrics and best/latest checkpoints ahead of optional replays. Record every pruning action.

If source download/install fails, no training starts. If a worker crashes, abort the run with diagnostics and preserve the last verified checkpoint; do not silently reduce environment count mid-run. If the notebook is disconnected but the kernel continues, the CLI must keep training and checkpointing without UI interaction. If the kernel dies, resume from saved artifacts; no promise of recovering unsaved working files is made.

## 5. Kaggle acceptance evidence

Required before calling the notebook supported: one fresh saved smoke run, one full training session or graceful time-budget session, one continuation in a new session, one ONNX export/parity result, and one replay from actual stochastic training imported into the browser. Save notebook version references, input/output manifest hashes and actual hardware information in `docs/benchmarks/kaggle-validation.md`.
