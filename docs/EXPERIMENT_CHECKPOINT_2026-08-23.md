# Bimanual showcase experiment checkpoint — 2026-08-23

## Delivery state

The ALOHA-only showcase target is complete with five success-validated tasks. Each task was replayed through Isaac Lab physics and recorded at 1920×1080 with a consistent indoor scene and five-camera layout.

| Order | Task | Frames | Duration | Success predicate |
|---:|---|---:|---:|---|
| 1 | Three-bowl stacking | 600 | 19.200 s | Pass |
| 2 | Two-bowl stacking | 376 | 7.520 s | Pass |
| 3 | Two-block stacking | 386 | 7.720 s | Task pass + post-settle stability pass |
| 4 | RGB block ranking | 239 | 7.648 s | Pass |
| 5 | Three-block stacking | 552 | 11.040 s | Task pass + post-settle stability pass |

The five clips are also combined into `artifacts/videos/showcase/00_bimanual_showcase_five_tasks.mp4`, a 51.16-second, 50 FPS presentation reel with 0.5-second cross-fades.

## Block-stack stability repair

The original stack predicate could pass while a released block was tilted or
still moving. Both canonical block-stack clips now include 0.5 s pre-roll, 1.0
s post-roll, interpolated 50 Hz action targets, and a post-settle acceptance
gate. The two-block result finished with 10.00 mm maximum adjacent XY offset,
upright cosine 1.0, 0.00346 m/s maximum linear speed, and 0.03802 rad/s maximum
angular speed. The three-block result finished with 13.49 mm offset, upright
cosine 1.0, 0.00817 m/s, and 0.04486 rad/s. Both passed the documented limits in
`docs/block-stability.md`; the earlier canonical clips are retained only in the
cloud artifact archive.

## Warehouse parcel-sorting showcase

`artifacts/videos/parcel_sorting_demo_v1.mp4` is a 420-frame, 14-second,
1920x1080 multi-camera demonstration. It resolves the Isaac 5.1 Digital Twin
Warehouse and `ConveyorBelt_A01.usd`, shows three parcel models, two destination
bins, and a cooperative left-to-right transfer. The adjacent manifest labels
the robot actions as recorded RoboTwin actions and the parcel motion as
deterministic presentation choreography; it does not claim GR00T inference or
surface-velocity conveyor physics.

The source tree is published at
`https://github.com/Erictianyufan/TwinGR00T-Lab` after secret scanning, 36 unit
tests, and Ruff validation.

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

## Current state

- No conversion or rendering process is running.
- Both RTX 4090 GPUs are idle.
- All five task videos, the rebuilt reel, the parcel demo, previews, JSON manifests, converted demonstrations, source archives, and logs remain on the cloud machine.
- The humanoid/G1 branch remains intentionally paused.
