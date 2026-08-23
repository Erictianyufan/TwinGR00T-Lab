# Bimanual showcase experiment checkpoint — 2026-08-21

## Scope decision

- Pause the Unitree G1 / humanoid branch.
- Continue with ALOHA bimanual manipulation only.
- Final delivery target: 4–6 clearly differentiated task demos using one consistent 1080p indoor, five-camera presentation layout.

## Completed and validated

1. `stack_bowls_three`
   - Isaac Lab live physics replay completed.
   - 600 frames, 31.25 FPS, 19.2 seconds, 1920×1080.
   - Cameras: third/overall, front, head, left wrist, right wrist.
   - Remote video: `artifacts/videos/showcase/01_stack_bowls_three.mp4`
   - Remote manifest: `artifacts/videos/showcase/01_stack_bowls_three.json`

2. `stack_bowls_two`
   - RoboTwin episode 0 passed the Isaac Lab success predicate.
   - Isaac Lab live physics replay completed.
   - 156 frames, 31.25 FPS, about 5 seconds, 1920×1080.
   - Remote video: `artifacts/videos/showcase/02_stack_bowls_two.mp4`
   - Remote manifest: `artifacts/videos/showcase/02_stack_bowls_two.json`

## Implemented infrastructure

- `scripts/replay_live_multiview.py` is now task-generic rather than hard-coded to `stack_bowls_three`.
- It automatically selects the task title/instruction and writes `task_key` to the JSON manifest.
- The presentation scene uses consistent indoor walls, table material, robot material, key/fill/rim lighting, and five camera panels.
- Environment compatibility overrides are preserved in `remote_overrides/`.
- Official RoboTwin 2.0 clean-50 archives are cached remotely for:
  - `stack_bowls_two`
  - `stack_blocks_two`
  - `stack_blocks_three`
  - `blocks_ranking_rgb`
  - `handover_block`

## Tested but not accepted as final demos

- `handover_block`: episodes 0–9 replayed, but all failed the Isaac Lab success predicate.
- `stack_blocks_two`: episodes 0–9 replayed; all failed, including after correcting the block stack height from 5 cm to 4 cm.
- `stack_blocks_three`: episodes 0–9 replayed; all failed after the same stack-height correction.

These failures are treated as SAPIEN-to-Isaac-Lab adapter/initial-state compatibility issues. They must not be labeled as successful demos until the final object state is visibly and numerically validated.

## Resume order

1. Run and validate `blocks_ranking_rgb` first; it is the strongest candidate for the third distinct task.
2. Diagnose missing clean-dataset object initial poses for block/handover tasks. Add final object-pose diagnostics before changing any tolerance.
3. If block replay becomes physically valid, render `stack_blocks_two` and `stack_blocks_three` with the common five-camera recorder.
4. If only three unique adapters are stable, add visually distinct, success-validated variants from separate official bowl episodes rather than presenting failed trajectories as successes.
5. Extract representative preview frames, inspect all five views, download videos/manifests, and assemble a 4–6 clip contact sheet/final montage.

## Pause state

- No `record.py` or `replay_live_multiview.py` process was running when this checkpoint was written.
- Existing datasets, converted demonstrations, logs, videos, and compatibility patches remain intact on the cloud machine.
