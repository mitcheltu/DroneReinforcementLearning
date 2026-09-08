# AeroRL 3D

A simulation-only autonomous drone racing project with a headless Python simulator, randomized ordered courses, PPO training, checkpoints, and a Kaggle notebook.

Start with [the Kaggle training guide](notebooks/README.md) for upload, training, resume, recorded failures and actor export. The [browser replay viewer](docs/implementation/10-browser-progress.md) imports those recordings directly and shows the drone in 3D. No qualified trained model is included. The [implementation contract](docs/implementation/README.md) defines the full release.

## Setup

Use Python 3.11, uv, and Node.js 22.14 or newer. Run commands from the repository root unless stated otherwise.

```powershell
uv python install 3.11
uv sync --frozen --extra learning
cd web
npm ci
cd ..
```

`uv.lock` is the Python resolution authority. `requirements.lock` is its hash-pinned pip-compatible export. The `learning` extra installs NumPy, PyTorch, Gymnasium, Stable-Baselines3, TensorBoard, ONNX and ONNX Runtime; omit that extra for contract-only work. No shell activation is required for the commands below. On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

If a Windows corporate certificate is required for HTTPS, use `uv --native-tls ...` and a current Node runtime with `--use-system-ca`. Do not disable TLS certificate verification. The runtime version must still satisfy the package engines.

## Verify the foundation

```powershell
.venv/Scripts/python.exe -m training.scripts.check_contracts
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check training
.venv/Scripts/mypy.exe
cd web
npm run typecheck
npm run lint
npm run test:unit
npm run test:integration
npm run build
```

Python tests include required cross-language validation and therefore require `npm ci` first. The web integration test requires the Python environment first. For a nonstandard interpreter location, set `AERORL_PYTHON` for web tests or `AERORL_NODE` for Python tests.

Both contract CLIs report parsed values, original byte lengths and SHA-256 hashes. `npm run contracts` runs the TypeScript CLI. For the shared valid/invalid corpus, pass `--fixtures ../shared/fixtures/contracts/cases.json` to the web CLI or `--fixtures shared/fixtures/contracts/cases.json` to the Python CLI.

## Run the replay viewer

```powershell
cd web
npm run dev
```

Open the local URL reported by Next.js. Import a `.npz` from the notebook's `training-traces` folder, or select a provided example. The viewer supports playback, time scrubbing, orbit/chase cameras, ordered gate progress and telemetry. Imported files stay in the browser. Live model-controlled flight is not enabled yet; the TypeScript numeric dynamics core has Python trajectory-parity tests but the full browser environment is still in progress.

## Current contract boundary

- The four JSON configurations are shared by both runtimes.
- Schemas reject missing/unknown properties and incompatible versions.
- Loaders reject nonfinite/unsafe numbers, malformed UTF-8, duplicate JSON keys and invalid state quaternions.
- Structural course checks enforce ten ordered gates, fixed openings, unique IDs and curriculum counts.
- Config checks cover timing, rotor geometry/signs, collective capacity, bounds and rollout consistency.
- Model/replay/message schemas establish artifact boundaries; synthetic fixtures contain no model or trajectory evidence.
- Python implements course geometry validation, swept frame collision checks, dynamics and training. The TypeScript numeric core matches Python test trajectories; browser event/observation/complete-flight parity remains pending. Schema acceptance alone is not a geometric feasibility result.
- A validated model manifest does not prove trained-model quality or that referenced files exist; artifact verification and release evaluation are later milestones.

## Dependency maintenance

Edit `pyproject.toml`, then `uv lock`, `uv sync --extra learning`, and regenerate the export with:

```powershell
uv export --frozen --extra learning --no-emit-project --format requirements-txt --output-file requirements.lock
```

Use exact direct versions in `web/package.json` and commit the updated package lock after npm changes. Run all affected checks before accepting either update. `notebooks/requirements-kaggle.lock` targets Linux x86_64 Python 3.11 and CPU Torch. Local validation and lock resolution are not evidence of Kaggle-hosted execution.

Next milestone: implement quaternion math, controller/mixer, motor response and coupled RK4 dynamics, then verify analytical physics and cross-language trajectories. See [M1 delivery criteria](docs/implementation/08-delivery-verification.md).
