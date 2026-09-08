# 07 — Browser application and deployment

## 1. Technology and package policy

Use TypeScript strict mode, React 19, Next.js App Router, Three.js, React Three Fiber with a release compatible with React 19, Drei, ONNX Runtime Web and a maintained ZIP reader/writer supporting streaming size checks. Resolve compatible stable package versions at M0, commit package-lock.json, and use npm ci in CI/deployment. Run dependency/security checks before locking and updates. Do not copy old example version numbers from the historical specification as current release requirements.

The 3D canvas and worker launcher are client components. No server rendering of WebGL, navigator, Worker or browser-only ONNX objects. Serve static model/config assets from the same origin. Next.js supplies the application shell; no API routes or inference server are required. Deploy the Next.js application to Vercel with reproducible build settings.

## 2. Page structure

Desktop layout: top bar with course name, mode/status and model readiness; main 3D viewport; right panel with gate list/editor or replay metadata; bottom transport controls and collapsible telemetry. Minimum supported viewport is 1024x768. Smaller viewports show a desktop-layout notice and remain readable; mobile interaction/performance is not a release requirement.

Required controls in flight mode: course selector, Run, Pause/Resume, Restart, Edit, Supported/Experimental toggle, Orbit/Chase camera, telemetry toggle and import replay. Gate list always displays labels 1–10. Current target is identified by text and color; completed gates have checkmarks.

Status states: loading_assets, ready, running, paused, editing, finished_success, finished_failure, replay_paused, replay_playing, error. Buttons unavailable in a state must be visibly disabled, not silently ignored. Missing model allows course viewing/editing and replay, but Run remains disabled with a reason.

## 3. Editor behavior

Default course is the first shipped supported benchmark. Ship at least 12 generated supported courses with fixed hashes, including straight, turning and varying-height examples; names describe geometry without claiming measured difficulty.

Edit while running sends pause, invalidates pending inference, marks the attempt user_abort, resets to initial state, and enables controls. Gate translation uses a horizontal-plane drag handle plus a distinct vertical handle; yaw uses a rotation handle. Right-panel numeric X/Y/Z and heading inputs provide precise keyboard access. Display meters to two decimals and degrees to one decimal, but retain full numeric precision internally.

On each drag/input update, perform full validation and display all relevant errors. Invalid tentative geometry may remain visible while editing, but Run/Apply is disabled in Supported mode. Escape cancels the current drag/input transaction. Undo/redo stores complete course snapshots, capped at 100 edits. Apply validates, resets drone state and returns to ready. Cancel restores the course snapshot from entry to Edit. Gate labels follow order; reordering via Move Up/Down rewrites labels and resets state. Exactly ten gates remain; add/remove buttons are absent.

Experimental mode requires a one-time per-page acknowledgment: layouts outside the supported training range may fail. Basic validity still blocks Run. This is a product explanation, not an external permission workflow. Show the specific violated support rules persistently and mark attempt metrics mode=experimental.

Restart uses the exact current initial state and course; it does not randomize. Selecting a benchmark aborts the active attempt, loads its snapshot and resets. Gate edits never change trained model weights. There are no moving targets during an active attempt.

## 4. Worker protocol

Main-to-worker message types: INIT (manifest and asset URLs), LOAD_COURSE (complete course), RUN, PAUSE, RESET, DISPOSE. Every state-changing request increments generation_id in the main thread. INIT/LOAD_COURSE/RESET responses acknowledge the generation and validated status. A request from an older generation is discarded.

Worker-to-main: READY, SNAPSHOT, EVENT, PAUSED, FINISHED, ERROR, PERFORMANCE. A snapshot includes current and previous state/time, target index, previous action, status, wall-clock send time and generation. Emit one snapshot per policy transition, plus immediate terminal/error/pause snapshots. Event messages contain all events since previous snapshot. Main ignores mismatched generations. Use transferable numeric buffers from a small pool; returned buffers can be reused. Never transfer the worker's only authoritative state buffer.

Add RETURN_BUFFERS as a non-state-changing main-to-worker message; it does not increment generation. A snapshot buffer returned from an old generation is safe to recycle only as storage, never as simulation state. On DISPOSE, cancel scheduling, invalidate pending results, release ONNX session and terminate the worker after acknowledgment or a 2-second timeout. INIT creates a fresh generation and numerical state; React remount cleanup must call DISPOSE to avoid duplicate controllers.

## 5. Deterministic simulation-time action scheduling

There is at most one inference in flight. One worker cycle performs:

1. At policy boundary t=k/60, build obs from current state.
2. Run ONNX with that observation; await completion without advancing simulation time.
3. Verify generation and running state still match. Discard output if reset/pause/load occurred while awaiting.
4. Validate/clamp/map the action.
5. Execute exactly two 120 Hz physics steps or stop at a terminal event, as in Python.
6. Publish snapshot/events; schedule the next cycle against the wall-clock pacing target.

Use performance.now() for wall pacing only. Let nominal cycle deadline advance by 1000/60 ms. If the cycle finishes early, wait until that deadline before starting the next cycle. If it finishes late, reset the next scheduling baseline to the current wall time; do not execute a burst of old decisions to catch up. Simulation may run slower than real time. Report real-time factor = simulated elapsed / wall elapsed over a rolling 5-second window. Display "Simulation running below real time" below factor 0.90 for two consecutive windows.

This deliberately replaces the historical latest-action hold during arbitrary async delay. The only action hold is the specified two physics ticks. Browser training parity is more important than maintaining wall speed on an overloaded machine. No requestAnimationFrame callback directly advances physics.

PAUSE increments generation and prevents the next physics cycle; if an inference is pending, discard it. A resumed cycle reobserves the unchanged state. Already completed physics steps are not undone. Document the pause acknowledgment as the exact paused state. When the tab becomes hidden, automatically pause; returning to visibility requires explicit Resume. No hidden-tab catch-up occurs.

## 6. ONNX backend setup and recovery

Default preferred backend is WebGPU when a usable adapter/session can be created; otherwise use WASM. Presence of navigator.gpu alone is insufficient. Attempt adapter acquisition, session creation with explicit `executionProviders:['webgpu']`, and a finite warm-up inference with correct output shape. Catch failures and try explicit WASM. Log backend selection and reason locally.

WASM uses one thread in v1 (`ort.env.wasm.numThreads=1`) to avoid cross-origin isolation as a mandatory hosting dependency. This supersedes the historical conditional multithreading setup. Serve matching ORT wasm files from the same locked package version and explicit same-origin path. COOP/COEP are not required in v1; enabling WASM multithreading is a later measured configuration change with deployment tests.

WebGPU fallback on device/session failure during flight pauses, preserves state, disposes the failing session, initializes WASM and reports readiness. User resumes explicitly. Never silently apply a zero action or advance while recovering. Model/config/schema mismatch is not a backend fallback condition: it is a hard loading error.

Both backends must be benchmarked. A diagnostics setting permits selecting WASM explicitly because a small MLP can be faster on CPU. A release may change the default after measurements, but must record the choice. Timeout for model/session initial loading is 30 seconds; show Retry after failure. Cache model fetches with content-hashed URLs; do not mix model versions after deployment.

## 7. Rendering and cameras

Keep the Three.js scene Z-up: set Object3D.DEFAULT_UP and camera.up to [0,0,1] before creating scene objects. Convert simulator quaternion `[w,x,y,z]` to Three.js constructor/set arguments `[x,y,z,w]` at the adapter only. Author/import the drone mesh with nose +X and top +Z; apply any asset correction transform once to a child mesh, not to physics or gate coordinates. Gate meshes use +X normal, +Y width, +Z height.

Render at display refresh via React Three Fiber. Mutate refs for per-frame transforms; do not set React state at 120 Hz. Interpolate buffered worker snapshots with a one-policy-interval presentation delay using linear position and SLERP attitude. If no later snapshot exists, hold the latest pose; never extrapolate beyond recorded/authoritative state. On reset or terminal, snap to the acknowledged exact state and clear interpolation history.

Orbit camera starts aimed at the course bounding-box center with a distance sufficient to fit it. Chase camera offset is [-5,0,2] m in the drone's yaw-only frame, looking 2 m ahead of drone center; smooth camera motion with time-based exponential response constant 0.15 s. Camera smoothing never alters drone state. Reset camera is available in both modes. Floor, grid, lighting and simple gate/drone geometry suffice; photorealistic assets are not required.

Drone animation may show cosmetic rotor spin while motors produce thrust, but labels/telemetry must not present that spin as modeled RPM. Optional collision sphere display belongs under diagnostics. Render gate numbers facing the camera with readable text, and preserve numeric labels even when color highlighting is disabled.

## 8. Performance requirements

Reference desktop must be recorded by actual CPU/GPU, RAM, OS, browser/version and display refresh. Required Chromium WASM reference results: p95 inference <=8 ms after warm-up, p95 two-step physics+observation+event work <=2 ms, real-time factor >=0.98 over five minutes, median rendered FPS >=55 and p5 FPS >=45 on a 60 Hz reference display. Report p50/p95/p99 inference, worker cycle timings, render frame intervals and memory. These are release targets, not current measurements.

If reference hardware differs from 60 Hz, measure render intervals and cap the acceptance harness to a 60 Hz target; do not claim 60 FPS on an uncapped 144 Hz metric without context. Browser support tests include current stable Chromium with both backends, Firefox WASM, and Safari WASM on available supported macOS hardware. Safari testing unavailable means explicitly unverified Safari support, not a pass.

Run repeated start/pause/reset/load/replay operations for 100 cycles and verify no retained worker/session/buffer growth beyond 10% after garbage collection/steady-state allowance. A browser may not expose deterministic GC; record test methodology and compare settled measurements, not immediate allocations.

## 9. Accessibility, errors and deployment

All controls have accessible names, keyboard focus and visible focus styling. Numeric gate editing is an alternative to dragging. Pause/Run and camera controls are standard buttons. Respect reduced motion for UI transitions; simulation playback remains user-controlled. Never rely on red/green alone. Error banners explain corrective action: retry model load, repair invalid course, select fallback backend or import a valid replay.

Deploy static model/wasm/config assets with long immutable caching only when filenames are content-hashed; HTML and the current release pointer must revalidate. Set correct MIME types for wasm, JSON and model downloads. Content Security Policy must allow the actual self-hosted worker/wasm requirements established by the selected ORT build and prohibit unneeded third-party scripts; verify the production build rather than guessing headers. No external CDN is required for models, fonts or scripts.

Release artifacts must be built from one tagged commit with lock files. A preview deployment passes the browser checklist before production promotion. Keep the prior known-good deployment/model manifest for rollback. Automatic promotion of every newly trained model is prohibited; only an evaluated release bundle can replace the current manifest.
