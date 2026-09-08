# Live ONNX flight in the web app

Open the local app on port 3000, select a course stage and variation, and press
**Fly with ONNX**. The model makes new decisions in a browser Web Worker; this
is not replaying Python actions. Stop cancels that worker. Completed or stopped
flights remain available for replay, orbit/chase cameras and timeline inspection.
Flight data remains in memory until another flight/replay is loaded or the page
is closed. Existing NPZ import and reference/failure replay options remain.

The initial course library has 16 deterministic generated layouts per stage,
112 definitions total. These are selectable variations, not a browser port of
the Python random course generator. Use **Edit gates** to customize a layout; see `GATE_EDITOR_AND_MONTAGE.md`. The bank uses
fresh seed base 90000000, plus stage*100000 and variant-1. No prerecorded actions
or states are shipped in the course definitions.

## Runtime

- `web/src/simulation/environment.ts`: the 40-value observation, normalized
  actions, body-rate controller, 120 Hz physics, segment collisions, ordered
  gate crossing, reward, hover departure and terminal conditions.
- `web/src/simulation/flight.worker.ts`: ONNX Runtime Web WASM on one thread,
  60 Hz policy actions, two fixed physics ticks per action. No Python service,
  GPU, CDN, cross-origin-isolation headers or network inference is required.
- UI snapshots arrive every four policy steps. Rendering and camera work stay
  on the main thread. The simulation targets real time but slows on hardware
  that cannot keep up; it never increases the integration timestep to catch up.
- Stop/unmount terminates the worker. Worker load/inference errors are reported
  visibly. Another flight creates a fresh worker and inference session.
- The ONNX actor includes its feature extraction; JavaScript supplies the public
  40-field observation, not the actor's internal 12-field feature vector.

Static assets are served from the same local app: `public/models/actor.onnx`,
`public/onnx/ort-wasm-simd-threaded.{mjs,wasm}`, and `public/courses.json`.
The runtime assets must match the pinned `onnxruntime-web` dependency (1.29.0).
The WASM file is approximately 14 MB and is loaded on the first neural flight.

## Rebuilding and validation

From the repository root, `python -m notebooks.build_live_assets` copies the
validated actor and installed WASM assets, regenerates course definitions, and
creates Python ONNX reference fixtures. It requires the local repaired model
under `runs/imitation-repair/iteration-0`. Normal web build/run needs only the
already prepared public assets, not that Python environment or checkpoint.

The new tests replay Python actions through JavaScript and compare every
policy-step state, observation, accumulated reward, target gate and terminal
time. Fixtures cover successful hover, one gate, ten gates, and ground failure.
State/reward tolerances are 1e-7; observations 2e-6; time 1e-8. Additional geometry
tests cover forward/backward crossings and frame/ground collisions. This checks
environment parity separately from neural backend behavior.

Manual production-build browser tests confirmed live WASM-controlled hover
success (5.00 s) and ten-gate stage-6 variation-1 success (25.45 s, reward 180.13),
matching the displayed precision of the Python reference. Browser coverage is
not yet an exhaustive rerun of all 208 Python evaluation cases.

Additional UI checks passed: stage-6 variation 2 finished all ten gates in
24.36 s; Stop froze a flight at 6.40 s and enabled replay; a subsequent stage-1
flight succeeded in 3.10 s (reward 62.96). Live status labels, disabled replay
controls during flight, the 3D layout and course progression were inspected in
the actual browser. All 82 web unit/integration tests, TypeScript, ESLint and
the final production build passed.

Run the web unit/integration tests, TypeScript check, ESLint and
`next build --webpack` before shipping changes. The production server can be
started from `web` with `next start --hostname 127.0.0.1 --port 3000`.
The original curriculum resume limitation still applies: this is the
imitation-trained deterministic actor, not a trained PPO critic/optimizer.

