# Isaac warehouse contact-physics parcel demo

This verified presentation places the ALOHA bimanual workcell at a visible
Isaac Simple Warehouse logistics station. The set includes the official Isaac
warehouse, conveyor, pallet, and cardboard-box assets plus a camera-safe storage
rack. Three parcel rigid bodies are picked, transported, routed to marked
destinations, released, and allowed to settle under PhysX.

The execution contract is deliberately strict:

- parcels retain the RoboTwin task's dynamic rigid-body, collision, friction,
  mass, and gravity configuration;
- warehouse set dressing is visual-only: its collision and rigid-body APIs are
  disabled so racks, pallets, cartons, and the conveyor cannot disturb the
  verified manipulation physics;
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
- the adjacent trace stores object motion, true wrist poses, and filtered
  per-hand contact forces for independent review.

The verified run passes all three grasp checks. Peak filtered forces are
151–160 N, lifts are 8.0–8.6 cm, maximum gripper-local translation drift is
4.4 mm, and the released objects finish below 0.14 mm/s linear speed. The
controller is recorded RoboTwin joint and gripper targets; this artifact does
not claim GR00T inference or conveyor surface-velocity physics.

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

The output is an 11-second, five-camera H.264 video, an adjacent JSON manifest,
and a compressed `.trace.npz`. A run fails unless every parcel has finger
contact, a meaningful lift, a continuous held interval, no more than 2.5 cm of
gripper-local relative translation drift, and low linear/angular velocity after
release. The checked artifact is substantially tighter than that limit.

## Legacy comparison

`scripts/parcel_sorting_demo.py` is retained only as the earlier deterministic
visualization. It writes parcel poses along authored paths and must not be used
as evidence of physical grasping.
