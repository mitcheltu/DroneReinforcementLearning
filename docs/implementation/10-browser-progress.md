# Browser implementation progress

This progress report records implemented behavior; it does not replace the release requirements in documents 01–09.

## Replay viewer

The root web route is now a working recorded-flight viewer. Run the existing web development command and open its reported localhost URL. Select **Import training attempt** and choose a `.npz` from an existing notebook run's `training-traces` directory. No retraining, notebook update or file conversion is required. A checkpoint ZIP, ONNX actor or complete run ZIP is not a replay and is rejected.

Supported controls: play/pause, restart, time slider, 0.25×/0.5×/1×/2× speeds, orbit/chase cameras, reset view and clickable event times. Playback pauses when the tab is hidden and stops advancing at the final state. Selecting another recording replaces the course and resets playback. Failed imports preserve the previously loaded recording. In-flight import results use a generation counter so obsolete results cannot replace a newer selection.

The scene uses Z-up coordinates and converts WXYZ simulator quaternions to Three.js XYZW at the display boundary. Recorded position is linearly interpolated and orientation uses shortest-arc quaternion SLERP. Playback clamps to recorded time bounds and never extrapolates. Drone body/rotor geometry and gate labels are rendered locally without an external font service. The path displays the complete recorded trajectory; it is a replay aid, not a prediction. The playback camera never modifies recorded state.

Telemetry shows current altitude, current speed, gates passed and total episode reward. Episode reward is the final saved total, not a running reward curve. Gate progress is derived from recorded, ordered gate-pass events. Hover recordings show the hover task instead of ten fabricated gates. A WebGL error leaves playback telemetry accessible and explains that 3D is unavailable.

Two provided examples are explicitly distinguished: an actual failed PPO hover attempt and a successful three-gate geometric reference-controller flight. The reference flight is not evidence that PPO has learned the course. Provenance is in `web/public/examples/README.md`.

## Import boundary

The importer supports `aerorl-notebook-trace-v1` NPZ diagnostics. The future `.aerorl.zip` browser release format is not enabled. ZIP parsing supports stored and raw-deflate entries; decompression uses the browser's native DecompressionStream. No upload endpoint is involved. Imported files are not persisted or transmitted.

Limits: 32 MiB compressed input and 32 MiB total declared decompressed data, at most 24 entries, numeric trajectory at most 5,402 states, at most 10,000 events and duration at most 45.001 seconds. ZIP paths must be simple NPY filenames; duplicate entries, encryption, multipart archives, malformed directories, oversized content and CRC failures are rejected. Decompressed bytes are capped while streaming, before constructing arrays. ZIP64-sized central entries are outside the supported size bounds; ordinary NumPy local ZIP64 headers work.

NPY parsing accepts little-endian float32/float64 numeric arrays and scalar UTF-32 metadata; it never evaluates Python headers or loads pickled objects. The reader checks shape/byte agreement, finite numeric values, monotonic state times, quaternion norms, supported metadata format, shared course-schema validity, ordered event times and sequential gate-pass labels. The current viewer consumes states/course/events/episode metrics; it does not claim complete verification or display of every optional diagnostic array.

## Browser dynamics core

`web/src/physics/dynamics.ts` implements Hamilton quaternion math, body/world rotations, float64-number coupled RK4 for the 17-state drone, gravity/drag, motor lag, torque allocation, collective/body-rate action mapping, the inner proportional rate controller, gyroscopic compensation and uniform motor desaturation.

Comparison fixtures are generated with `python -m notebooks.build_physics_fixtures` using the existing Python training implementation. They contain 240 ticks each for exact hover, a yaw step, and mixed saturated commands. Tests compare all four motor commands, desaturation scale and all 17 state fields on every tick with absolute tolerance 1e-9. The fixtures bind to the exact vehicle JSON SHA-256.

All **720 ticks passed**. This validates the numeric core for these cases, not the complete browser flight environment. Remaining work includes collision/events, observations and rewards in TypeScript, the inference worker, model/config artifact verification, course editing, full-flight cross-runtime tests, performance measurement and release qualification. The replay viewer does not advance this new physics core; it plays saved authoritative Python states.

## Training continuity

This delivery changes web files, browser examples, documentation and standalone helper scripts. It does not change `training/**/*.py` or the shared configuration files. A byte comparison confirmed that those files match the currently preserved source upload ZIP. The earlier local 34,816-step validation checkpoint has an older source fingerprint and is not evidence of compatibility with the current bundle. The running Kaggle session has not been inspected. Keep its attached source archive unchanged while that campaign continues.

Validation includes 76 web unit tests and the existing cross-language integration test, including actual NPZ examples, browser-native decompression, malformed archives, interpolation and numeric physics parity. Lint passes; the production build includes TypeScript checking. Browser screenshots, interactive visual QA and browser-specific performance acceptance have not been performed; a successful bundle build is not those measurements.
