# Gate editor and training montage

## Configure a flight

1. Open http://127.0.0.1:3000/ and choose a course stage and variation.
2. Select **Edit gates**. Select a gate by its dropdown label or click its frame in 3D.
3. Enter X, Y, Z in meters and heading in degrees; press Enter or leave the field to commit. Alternatively drag the translation handles, or choose **Rotate heading** and drag the Z rotation handle. Orbit controls pause during handle dragging.
4. Arrows point in the required crossing direction: heading 0 degrees is +X; 90 degrees is +Y. Gates remain upright. Gate labels and crossing order remain 1 through the number of gates in the selected course. The editor does not add/remove gates or change the drone's initial state.
5. Select **Fly with ONNX**. A copy of the edited course is sent to the inference worker, and the neural policy responds to that layout's observations. Editing is disabled during flight. Stop ends inference; the resulting attempt can be replayed.

**Reset layout** restores the selected preset. **Save course JSON** downloads the current committed layout; **Load course JSON** restores one. Drafts are held in page memory separately for each stage/variation. Save before reloading, navigating to the montage, or closing the page. JSON imports are limited to 1 MiB; invalid layouts leave the current course intact. Short imported layouts are mapped to stages 0, 1, or 4; ten-gate layouts to stage 6. Gate count determines the flight timeout.

Validation requires finite values, unique IDs, consecutive labels, upright fixed-size gates, valid initial state and headings, and entire gate frames within X/Y ±60 m and Z 0–20 m. Supported counts are 0, 1, 3, 10. Heading input wraps to [-180, 180). Empty/invalid number fields block flight until corrected or the draft inputs are discarded. Validation does not prove reachability or enforce learned spacing/turn limits: overlapping gates, reversed approach directions and extreme turns can fail.

The browser editor has an additive schema in `web/src/editor/course.schema.json`, registered as `editable-course`. It permits experimental courses with 0, 1, 3 or 10 gates and no generator provenance. The original shared v1 contract remains unchanged: its experimental courses require ten gates. Short custom JSON files are therefore browser-editor artifacts, not inputs for the original Python curriculum loader. Ten-gate custom files retain the original contract shape. Original training source/config fingerprints and model assets are unchanged.

## View all retained attempts together

Open http://127.0.0.1:3000/montage. Choose **Source**, **Training type**, and **Outcomes**. Every matching retained attempt is displayed at once, without sampling a subset of attempts. Green means success; coral means failure.

- **PPO training**: 32 saved trajectories from runs 003 and 004. Their episode logs contain 16,653 hover episodes and 413 one-gate episodes. Most of those trajectories were never saved and cannot be reconstructed from reward/duration/outcome summaries.
- **Imitation demonstrations**: 32 reconstructed reference-controller flights, 16 hover and 16 one-gate, using the original seeds. These represent the original demonstration flights; augmented supervised label states and optimizer iterations are not flight trajectories.
- **Repaired policy validation**: 18 recorded evaluation flights, separated from training.
- **Imported traces**: select multiple original training NPZ files. Every successfully imported matching attempt appears together. A batch is accepted atomically. Limits: 32 MiB per file, 256 MiB per batch, 1,000 imported attempts per page session. Clear imported traces to free them. Files remain in browser memory.

Paths remain fully visible while **Play all attempts** animates each drone body. Playback can pause, restart, scrub, and replay after reaching the end. An attempt freezes at its final pose. The default clock is elapsed seconds since episode start; **Synchronize by progress** gives each attempt the same percentage of its own duration (a full pass takes ten seconds). Opacity ranges from 3% to 100%; 100% is opaque. **Show every gate layout** draws each attempt's gates. **Align starts & headings** translates each start reference to the origin and rotates its initial heading to +X, transforming gates and drone orientations consistently. Disable alignment to see original world coordinates.

Each trajectory is reduced to at most 601 display poses, preserving the final pose. This is a visual montage, not the authoritative physics record. Full NPZ files retain their original samples. Transparent paths and instanced bodies can overlap without correct per-instance depth sorting; opacity controls readability, not quantitative density. Large imports can consume substantial memory and render more slowly.

`python -m notebooks.build_montage` rebuilds the bundled data from local saved runs and reconstructs demonstrations only when the original training fingerprint matches. The published data records source and provenance per attempt. No missing PPO flight is synthesized.

## Record every future completed PPO episode

The original PPO trainer samples saved attempts. The optional wrapper records every completed training episode:

```bash
python -m notebooks.train_record_all --output runs/recorded-training --steps 150000 --budget 10000000 --seed 101 --n-envs 1 --recording-bytes 2000000000
```

Use a new output directory. This example starts a fresh PPO run; it is a recording option, not a recommendation to resume the failed policy or train the imitation checkpoint with the original PPO trainer. For an otherwise compatible PPO checkpoint, add the original trainer's `--resume CHECKPOINT_DIRECTORY` option. On Kaggle, call `run_module("notebooks.train_record_all", *arguments)` instead of the original training module after including this wrapper in the source bundle.

Files go into `all-training-traces`, one NPZ per completed episode, named with global transition and vector environment index. The wrapper saves state samples, applied and raw actions, rewards, events and course/episode metadata using the existing trace format. It also writes `recording-policy.json`. Full paths, orientations, gate layout, stage, outcome and timestamps are necessary for montage rendering; actions/rewards preserve diagnostics. Keep model checkpoints and campaign/evaluation logs alongside the traces to associate attempts with learning progress.

The recording quota is a soft byte budget. When reached, the wrapper saves all episodes completed in that vector step and stops training; the trainer's finally block saves a checkpoint. That final vector step can exceed the quota. Unfinished episodes are not recorded on interruption. No completed episodes are silently dropped to stay under quota. File-system errors stop the run; existing same-named trace files are never overwritten. This wrapper does not promise exact replay of an interrupted optimizer rollout.

Verification includes a 256-step fresh smoke run: four completed episodes, four saved NPZ files, and a final checkpoint at step 256. A one-byte quota test stopped at the first completed episode (step 60), retained its NPZ and saved a checkpoint. This checks recording behavior, not policy quality.

## Web verification

The production build and ESLint pass. The 86-test web unit/integration suite passed, including Python contract parity; an additional schema-compatibility regression test subsequently passed with the four editor/montage tests. Browser verification covered numeric height/heading edits, dragging the Z handle, an actual edited-course ONNX success (3.11 seconds, one gate passed), 14 overlaid PPO hover attempts, 18 overlaid one-gate attempts, opaque paths, gate overlays, and replay restart after completion. This edited-flight success is an example, not a guarantee for arbitrary layouts.
