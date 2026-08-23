# RoboTwin upstream contribution audit

This document separates reusable RoboTwin improvements from TwinGR00T Lab's
project-specific presentation code. The audit compares this repository against
RoboTwin `main` at `3095469` and the `IsaacLab-Arena` branch at `e19d288` on
2026-08-24.

## Community status

- Published the IsaacLab-Arena scene-reconstruction evidence as a
  [comment on RoboTwin #288](https://github.com/RoboTwin-Platform/RoboTwin/issues/288#issuecomment-5387649341)
  on 2026-08-24.
- Opened [RoboTwin #491](https://github.com/RoboTwin-Platform/RoboTwin/issues/491),
  proposing configurable post-settle stability for block-stack success.
- No upstream code PR has been opened yet. The next patch depends on maintainer
  feedback about data-export ownership and SAPIEN-specific threshold calibration.

## Contribution candidates

| Candidate | Evidence in this project | Upstream fit | Recommended route |
|---|---|---|---|
| Deterministic scene reconstruction for IsaacLab-Arena replay | Clean archives without `object_info` caused bowls to spawn below the table and block actions to miss. Seed-based reconstruction restored four tasks; calibrated RGB back-projection provides a fallback. | High for the `IsaacLab-Arena` branch. The durable fix should persist initial object IDs, poses, and quaternions instead of maintaining task-specific random samplers. | Evidence submitted in the [RoboTwin #288 comment](https://github.com/RoboTwin-Platform/RoboTwin/issues/288#issuecomment-5387649341), explicitly distinguishing it from the filesystem-order fix in PR #490. Awaiting maintainer guidance on export versus replay ownership. |
| Post-settle validation for stacking | The original positional predicate accepted tilted or moving blocks. A final-action hold plus position, uprightness, linear-speed, and angular-speed checks rejected the false positive and accepted stable two- and three-block replays. | High. RoboTwin `main` still checks block center positions and open grippers but not orientation or velocity. | Proposal submitted as [RoboTwin #491](https://github.com/RoboTwin-Platform/RoboTwin/issues/491). After maintainers confirm thresholds and timing, implement a minimal shared helper and task tests. |
| Contact-physics evidence for grasping | `physics_grasp.py` checks contact force, lift, continuous hand-relative pose, release, and final settling; dependency-light tests reject pose animation without contact. | Medium to high as an optional evaluator. It is broader than a single task and should not silently change leaderboard semantics. | Include it as a follow-on design in the stability issue; propose an opt-in trace/evidence layer before a code PR. |
| RoboTwin-to-GR00T N1.7 transport and data contract | The repository implements a 14-D logical ALOHA contract, LeRobot v2 export, modality metadata, and policy-server transport. | Low as a new adapter: current RoboTwin `main` routes policies through XPolicyLab, which already contains `policy/GR00T_N17`. | Do not submit a duplicate adapter. A future contribution should add missing validation or a verified checkpoint/result to the existing XPolicyLab adapter. |
| Multi-camera indoor and warehouse renderers | Five-camera videos, indoor materials, warehouse set dressing, and a conveyor presentation are useful for communication. | Low for the benchmark core. The warehouse demo reuses recorded RoboTwin actions and is not a new benchmark task or learned-policy result. | Keep in TwinGR00T Lab unless RoboTwin requests a visualization example. |
| Task-specific environment overrides | The overrides force ALOHA defaults, change sizes or physics values, and remove other embodiments. | Low: they alter upstream defaults and reduce embodiment coverage. | Do not upstream wholesale. Extract only independently justified fixes. |

## Reproducible evidence

### Scene reconstruction

- Affected clean archives: `stack_bowls_two`, `stack_blocks_two`,
  `stack_blocks_three`, and `blocks_ranking_rgb`.
- The first invalid two-bowl conversion placed roots at `z=-1.167 m` and
  `z=-1.196 m`; the repaired conversion restored both to `z=0.766 m` and
  passed the unchanged Isaac Lab task predicate.
- Block reconstruction reads the archived `seed.txt`, reproduces the task's
  rejection sampler, and restores x/y/yaw. RGB segmentation plus camera
  calibration is used only when deterministic state is unavailable.
- Implementation: `remote_overrides/record.py` and
  `scripts/estimate_block_initial_poses.py`.

The preferred upstream contract is explicit initial scene state in every
replayable episode. Seed replay is a compatibility fallback because random
number consumption, asset enumeration, or task code can change between
versions.

### Stable stack acceptance

The accepted two-block replay ended with a 10.00 mm maximum adjacent XY offset,
upright cosine 1.0, 0.00346 m/s maximum linear speed, and 0.03802 rad/s maximum
angular speed. The accepted three-block replay measured 13.49 mm, 1.0,
0.00817 m/s, and 0.04486 rad/s respectively. The gate holds the final action
for 30 control steps and uses the documented limits in `block-stability.md`.

This is evaluation evidence from the pinned IsaacLab-Arena runtime, not a claim
that the same thresholds have already been validated on RoboTwin's current
SAPIEN `main` branch.

## Published community submissions

### [Comment on issue #288](https://github.com/RoboTwin-Platform/RoboTwin/issues/288#issuecomment-5387649341)

```markdown
I reproduced a related but separate replay-fidelity failure on the
`IsaacLab-Arena` branch at `e19d288`. PR #490 addresses filesystem-dependent
model selection on `main`; the clean Arena archives I tested can also omit
`object_info` from `scene_info.json`, so `script/record.py` loads an empty
object map and the procedural objects remain at their default poses.

Affected clean tasks in my test were `stack_bowls_two`, `stack_blocks_two`,
`stack_blocks_three`, and `blocks_ranking_rgb`. In the invalid two-bowl replay,
the roots ended at `z=-1.167 m` and `z=-1.196 m`. Restoring the initial poses
placed both at `z=0.766 m`, after which episode 0 passed the unchanged Isaac Lab
success predicate. For block tasks I reproduced the task rejection sampler
from the archived `seed.txt`; when a seed is unavailable I use first-frame RGB
plus saved camera calibration as a diagnostic fallback.

Evidence and implementation:

- experiment checkpoint: https://github.com/Erictianyufan/TwinGR00T-Lab/blob/c702fb2a51c456dee65d8f1311121591a4496d0c/docs/EXPERIMENT_CHECKPOINT_2026-08-23.md
- compatibility code: https://github.com/Erictianyufan/TwinGR00T-Lab/blob/c702fb2a51c456dee65d8f1311121591a4496d0c/remote_overrides/record.py

My recommendation is to persist the selected object IDs and complete initial
poses (position + quaternion) in each exported episode. Seed reconstruction is
useful for old archives, but it remains sensitive to RNG consumption and task
code changes. If maintainers prefer the compatibility path in `record.py`, I
can extract a small PR without the project-specific vision heuristics.
```

### [RoboTwin issue #491](https://github.com/RoboTwin-Platform/RoboTwin/issues/491)

**Title:** `[Proposal] Require post-settle stability for block-stack success`

```markdown
## Problem

The block-stack success functions on RoboTwin `main` check block-center
positions and open grippers, but do not check block orientation or residual
linear/angular velocity. A trajectory can therefore be accepted at a transient
instant while a block is tilted or the tower is already falling.

I observed this independently while replaying `stack_blocks_two` and
`stack_blocks_three` through the RoboTwin `IsaacLab-Arena` branch. The task
predicate passed, but the top block visibly continued to fall. Holding the last
action for a short settling window and validating uprightness and velocity
removed the false positive.

## Proposed behavior

Keep the existing geometric task condition, then require a configurable
post-settle check:

1. both grippers are open and no longer support a block;
2. adjacent block centers meet the task XY/Z tolerance;
3. each block's local Z axis is sufficiently aligned with world Z;
4. maximum block linear and angular speed are below thresholds for a short
   consecutive window.

The thresholds should be task/runtime configuration rather than hard-coded
leaderboard changes. An opt-in evidence record could expose the measured
position, orientation, velocity, and contact values without changing existing
scores immediately.

## Reproduction evidence

On the pinned Arena runtime, accepted stable replays measured:

| task | max adjacent XY | min upright cosine | max linear speed | max angular speed |
|---|---:|---:|---:|---:|
| two blocks | 10.00 mm | 1.0 | 0.00346 m/s | 0.03802 rad/s |
| three blocks | 13.49 mm | 1.0 | 0.00817 m/s | 0.04486 rad/s |

Protocol and code:

- https://github.com/Erictianyufan/TwinGR00T-Lab/blob/c702fb2a51c456dee65d8f1311121591a4496d0c/docs/block-stability.md
- https://github.com/Erictianyufan/TwinGR00T-Lab/blob/c702fb2a51c456dee65d8f1311121591a4496d0c/src/twingr00t_lab/physics_grasp.py

The numbers above validate the Arena replay path only; I have not yet calibrated
the same thresholds on current SAPIEN `main`. If this direction is acceptable,
I can submit a small shared stability helper and task-level regression tests.
```

## Merge prerequisites

- Get maintainer agreement on whether scene state should be fixed at export or
  reconstructed by the Arena replayer.
- Calibrate any stability thresholds independently for SAPIEN and Isaac Lab;
  do not copy presentation thresholds into leaderboard scoring unchanged.
- Add an explicit license to TwinGR00T Lab before asking maintainers to reuse
  whole modules from this repository. A direct RoboTwin patch can instead be
  contributed under RoboTwin's MIT license.
- Run the upstream task or simulator tests for every code PR; the local
  dependency-light tests alone are not sufficient.
