# Bimanual showcase experiment checkpoint — 2026-08-23

## Delivery state

The ALOHA-only showcase target is complete with five success-validated tasks. Each task was replayed through Isaac Lab physics and recorded at 1920×1080 with a consistent indoor scene and five-camera layout.

| Order | Task | Frames | Duration | Success predicate |
|---:|---|---:|---:|---|
| 1 | Three-bowl stacking | 600 | 19.200 s | Pass |
| 2 | Two-bowl stacking | 376 | 7.520 s | Pass |
| 3 | Two-block stacking | 156 | 4.992 s | Pass |
| 4 | RGB block ranking | 239 | 7.648 s | Pass |
| 5 | Three-block stacking | 236 | 7.552 s | Pass |

The five clips are also combined into `artifacts/videos/showcase/00_bimanual_showcase_five_tasks.mp4`, a 44.92-second, 50 FPS presentation reel with 0.5-second cross-fades.

## Two-bowl repair

The first two-bowl export was invalid: both bowl roots had fallen below the table (`z=-1.167 m` and `z=-1.196 m`) because clean `scene_info.json` omitted object poses. The replacement conversion estimates both bowl centers from the archived first head-camera RGB frame, back-projects them through the saved camera calibration, and restores them at `z=0.766 m`. Episode 0 then passed the unchanged Isaac Lab stacking success predicate.

The clean converter records every second control step, so the original 156-frame presentation also changed joint targets too abruptly. The replacement video restores the 50 Hz timeline with two-times linear action interpolation, plus 0.5 s pre-roll and 0.8 s post-roll. Its 376 frames keep both bowls visible and make the arm motion substantially smoother.

## Root cause and fix for block-task failures

The RoboTwin clean archives do not store `object_info` poses in `scene_info.json`. The Isaac Lab converter therefore spawned all procedural blocks at their default pose, causing the original expert actions to miss the objects.

The converter now:

1. Reads the official per-episode `seed.txt`.
2. Reproduces RoboTwin's clean-data random block sampler and restores x/y/yaw.
3. Falls back to RGB segmentation and camera-calibration back-projection when a seed is unavailable.
4. Uses the original 5 cm block size and 5 cm stacking offset for the stacking tasks.
5. Preserves the original success predicate; failed trajectories are not mislabeled as successes.

Compatibility changes are preserved in `remote_overrides/record.py` and the task-specific environment overrides. The diagnostic utility is `scripts/estimate_block_initial_poses.py`.

## Pause state

- No conversion or rendering process is running.
- Both RTX 4090 GPUs are idle.
- All five videos, previews, JSON manifests, converted demonstrations, source archives, and logs remain on the cloud machine.
- The humanoid/G1 branch remains intentionally paused.
