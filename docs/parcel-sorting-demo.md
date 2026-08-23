# Isaac warehouse contact-physics parcel demo

This verified presentation places the ALOHA bimanual workcell at a visible
Isaac Simple Warehouse logistics station. The set includes the official Isaac
warehouse, conveyor, pallet, and cardboard-box assets plus a camera-safe storage
rack. Three parcel rigid bodies first travel along a visible feed belt, stop at
the pickup coordinates, and are then picked, routed to marked destinations,
released, and allowed to settle under PhysX.

The overall view contains a compact feed conveyor with a dark belt, orange
guards, support legs, lane markers, and all three moving parcels. Isaac Sim 5.1
GPU/Fabric does not execute the runtime-authored SurfaceVelocity API reliably
for this RoboTwin table. The verified implementation therefore uses bounded,
closed-loop tangential forces during the feed phase. Mass, friction, contact,
inertia, gravity, and settling remain active; parcel poses are never animated.

The execution contract is deliberately strict:

- parcels retain the RoboTwin task's dynamic rigid-body, collision, friction,
  mass, and gravity configuration;
- warehouse set dressing is visual-only: its collision and rigid-body APIs are
  disabled so racks, pallets, and background cartons cannot disturb the task;
- conveyor traction is enabled only during the feed interval and is cleared
  before the recorded manipulation begins;
- parcel world poses are written only during initial-state restoration and are
  never animated during the replay;
- grasping uses native collision and friction between the parcels and four
  ALOHA finger bodies, with no fixed joint or scripted attachment;
- actions are continuously resampled from a successful recorded RoboTwin
  sequence;
- success-triggered environment auto-reset is disabled, allowing the final
  grasp to finish its release and settling tail without a robot pose jump;
- five cameras show the overall workcell, front view, head view, and both wrist
  views, with live filtered contact-force diagnostics;
- the adjacent trace stores object motion, conveyor traction, true wrist poses,
  and filtered per-hand contact forces for review.

The verified run passes the feed and all three grasp checks. Parcels travel
0.414 m, arrive within 5.9 mm, and stop below 0.08 mm/s before manipulation.
Peak filtered grasp forces are 139–171 N, lifts are 7.9–8.7 cm, maximum
gripper-local translation drift is 5.7 mm, and released objects again finish
below 0.08 mm/s. The controller is recorded RoboTwin joint and gripper targets;
this artifact does not claim GR00T inference or native surface-velocity physics.

## Render command

```bash
source /home/ubuntu/env_robotwin/bin/activate
python scripts/parcel_sorting_physics_demo.py \
  vendor/RoboTwin/data/blocks_ranking_rgb_posefix/data/demo_0.hdf5 \
  --output artifacts/videos/parcel_sorting_physics_verified.mp4 \
  --headless --device cuda:0 --rendering_mode quality \
  --environment manip_eval_tasks.examples.manipulation.blocks_ranking_rgb_environment:BlocksRankingRgbEnvironment \
  blocks_ranking_rgb --embodiment aloha --enable_cameras True
```

The output is a 15.6-second, five-camera H.264 video, an adjacent JSON manifest,
and a compressed `.trace.npz`. A run fails unless every parcel has finger
contact, a meaningful lift, a continuous held interval, no more than 2.5 cm of
gripper-local relative translation drift, and low velocity after release. Feed
acceptance additionally requires real displacement, bounded arrival error and
lateral drift, and a stable handoff to manipulation.

## Legacy comparison

`scripts/parcel_sorting_demo.py` is retained only as the earlier deterministic
visualization. It writes parcel poses along authored paths and must not be used
as evidence of physical grasping.
